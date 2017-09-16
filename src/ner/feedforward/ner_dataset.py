from ner_sentence import Sentence, UNK
import utils
import collections
import codecs
import re
import time
import token
import os
import pickle
import utils


class Dataset(object):
  """A class for handling CoNLL data sets."""

  def __init__(self, train_datapath="train.conllu", dev_datapath=None, test_datapath=None, format_list=["FORM", "NER"], data_output=""):
    self.format_list = format_list
    self.comment_sign = ""
    self.word2id = {}
    self.tag2id = {}
    self.prefix2id = {}
    self.suffix2id = {}
    self.index = {}
    self.data_path = {}
    if train_datapath is not None:
      self.data_path['train'] = train_datapath
    if dev_datapath is not None:
      self.data_path['dev'] = dev_datapath
    if test_datapath is not None:
      self.data_path['test'] = test_datapath
    self.index['train'] = -1
    self.index['dev'] = -1
    self.index['test'] = -1
    self.rare_words = []
    self.data_output = data_output

  def _parse_dataset(self, fgen):
    word_count = collections.defaultdict(lambda: 0)
    tag_count = collections.defaultdict(lambda: 0)
    prefix_count = collections.defaultdict(lambda: 0)
    suffix_count = collections.defaultdict(lambda: 0)
    sentence_list = {}
    sentence_list['train'] = []
    sentence_list['dev'] = []
    sentence_list['test'] = []
    format_len = len(self.format_list)
    column_list = {}
    for field in self.format_list:
      column_list[field] = []

    for name, path in self.data_path.items():
      with open(path) as f:
        for line in f:
          entity = line.split()
          if len(entity) == format_len and entity[0] != self.comment_sign and '-DOCSTART-' not in line:
            for i in range(format_len):
              column_list[self.format_list[i]].append(entity[i])
            if fgen:
              word = column_list["FORM"][-1]
              ner = column_list["NER"][-1]
              word_count[word] += 1
              tag_count[ner] += 1
              prefix_count[word[2:]] += 1
              suffix_count[word[:-2]] += 1
              prefix_count[word[3:]] += 1
              suffix_count[word[:-3]] += 1
          else:
            if column_list[self.format_list[-1]] != []:
              sentence_list[name].append(Sentence(column_list, self.format_list))
            column_list = {}
            for field in self.format_list:
              column_list[field] = []

    return prefix_count, suffix_count, word_count, tag_count, sentence_list

  def load_dataset(self, word2id=None, tag2id=None, prefix2id=None, suffix2id=None, fgen=True):
    '''
    dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
    '''
    start_time = time.time()
    if fgen:
      self.tokens_mapped_to_unk = []
      prefix_count, suffix_count, word_count, tag_count, self.sentence_list = self._parse_dataset(fgen)
      word_count = utils.order_dictionary(word_count, 'value_key', reverse = True)
      tag_count = utils.order_dictionary(tag_count, 'key', reverse = False)
      prefix_count = utils.order_dictionary(prefix_count, 'value_key', reverse = True)
      suffix_count = utils.order_dictionary(suffix_count, 'value_key', reverse = True)
      
      pid = 0
      self.prefix2id["-padding-"] = pid
      pid += 1
      for pre, count in prefix_count.items():
        self.prefix2id[pre] = pid
        pid += 1
      self.prefix2id['-UNK-'] = pid
      pickle.dump(self.prefix2id, open(os.path.join(self.data_output, "ner_prefix2id"), 'wb'))

      sid = 0
      self.suffix2id["-padding-"] = sid
      sid += 1
      for suf, count in suffix_count.items():
        self.suffix2id[suf] = sid
        sid += 1
      self.suffix2id['-UNK-'] = sid
      pickle.dump(self.suffix2id, open(os.path.join(self.data_output, "ner_suffix2id"), 'wb'))
      
      wid = 0
      self.word2id["-padding-"] = wid
      wid += 1
      for word, count in word_count.items():
        self.word2id[word] = wid
        wid += 1
      self.word2id['-UNK-'] = wid
      pickle.dump(self.word2id, open(os.path.join(self.data_output, "ner_word2id"), 'wb'))

      # Ensure that both B- and I- versions exist for ach label
      labels_without_bio = set()
      for label, count in tag_count.items():
        new_label = utils.remove_bio_from_label_name(label)
        labels_without_bio.add(new_label)

      prefixes = ['B-', 'I-']
      nid = 0
      self.tag2id['O'] = nid
      nid += 1
      for label in labels_without_bio:
        if label == 'O':
          continue
        for prefix in prefixes:
          l = prefix + label
          self.tag2id[l] = nid
          nid += 1
      pickle.dump(self.tag2id, open(os.path.join(self.data_output, "ner_tag2id"), 'wb'))

    else:
      self.word2id = word2id
      self.tag2id = tag2id
      self.prefix2id = prefix2id
      self.suffix2id = suffix2id
      _, _, _, _, self.sentence_list = self._parse_dataset(fgen)

    for name, sent_list in self.sentence_list.items():
      for sent in sent_list:
        sent.gen_id_list(self.word2id, self.tag2id)
        sent.gen_sent_features(self.word2id, self.prefix2id, self.suffix2id)

    self.number_of_classes = len(self.tag2id)
    self.vocabulary_size = len(self.word2id)
    self.prefix_size = len(self.prefix2id)
    self.suffix_size = len(self.suffix2id)

    self.id2word = utils.reverse_dictionary(self.word2id)
    self.id2tag = utils.reverse_dictionary(self.tag2id)

    elapsed_time = time.time() - start_time
    print('loading dataset done ({0:.2f} seconds)'.format(elapsed_time))

  def get_sent_num(self, name):
    return len(self.sentence_list[name])

  def get_next_sent(self, name):
    if self.has_next_sent(name):
      self.index[name] += 1
      return self.sentence_list[name][self.index[name]]
    raise IndexError("Run out of data while calling get_next_sent()")

  def has_next_sent(self, name):
    index = self.index[name] + 1
    if index >= len(self.sentence_list[name]):
      return False
    else:
      return True

  def reset_index(self, name):
    self.index[name] = -1

if __name__ == '__main__':
  data_path1 = "/cs/natlang-data/CoNLL/CoNLL-2003/eng.train"
  data_path2 = "/cs/natlang-data/CoNLL/CoNLL-2003/eng.testa"
  data_path3 = "/cs/natlang-data/CoNLL/CoNLL-2003/eng.testb"
  ds = Dataset(data_path1, data_path2, data_path3, format_list=["FORM", "a", "b", "NER"])
  ds.load_dataset()
  print ds.get_sent_num('train')
  print ds.get_sent_num('dev')
  print ds.get_sent_num('test')
  