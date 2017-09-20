from ner_sentence import Sentence
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

  def __init__(self, train_datapath="train.conllu", dev_datapath=None, test_datapath=None, use_char=False, bpe=None):
    self.format_list = ["FORM",
                        "POSTAG",
                        "CHUNK",
                        "NER"]
    self.comment_sign = ""
    self.char_index = {}
    self.word_index = {}
    self.tag_index = {}
    self.ner_index = {}
    self.prefix_index = {}
    self.suffix_index = {}
    self.use_char = use_char
    self.index = {}
    self.data_path = {}
    if train_datapath is not None:
      self.data_path['train'] = train_datapath
    if dev_datapath is not None:
      self.data_path['dev'] = dev_datapath
    self.data_path['test'] = test_datapath
    self.index['train'] = -1
    self.index['dev'] = -1
    self.index['test'] = -1
    self.bpe = None

  def _parse_dataset(self, fgen):
    word_count = collections.defaultdict(lambda: 0)
    tag_count = collections.defaultdict(lambda: 0)
    char_count = collections.defaultdict(lambda: 0)
    ner_count = collections.defaultdict(lambda: 0)
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
      f = codecs.open(path, 'r', 'UTF-8')
      for line in f:
        entity = line.split()
        if len(entity) == format_len and entity[0] != self.comment_sign and '-DOCSTART-' not in line:
          for i in range(format_len):
            column_list[self.format_list[i]].append(str(entity[i]))
          if fgen:
            word = column_list["FORM"][-1]
            tag = column_list["POSTAG"][-1]
            ner = column_list["NER"][-1]
            word_count[word] += 1
            tag_count[tag] += 1
            ner_count[ner] += 1
            prefix_count[word[2:]] += 1
            suffix_count[word[:-2]] += 1
            prefix_count[word[3:]] += 1
            suffix_count[word[:-3]] += 1
            for char in word:
              char_count[char] += 1
        else:
          if column_list[self.format_list[0]] != []:
            sentence_list[name].append(Sentence(column_list, self.format_list))
          column_list = {}
          for field in self.format_list:
            column_list[field] = []

    return prefix_count, suffix_count, word_count, tag_count, char_count, ner_count, sentence_list

  def load_dataset(self, word_index=None, tag_index=None, char_index=None, ner_index=None, prefix_index=None, suffix_index=None, fgen=True):
    '''
    dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
    '''
    start_time = time.time()
    if fgen:
      self.tokens_mapped_to_unk = []
      prefix_count, suffix_count, word_count, tag_count, char_count, ner_count, self.sentence_list = self._parse_dataset(fgen)
      word_count = utils.order_dictionary(word_count, 'value_key', reverse = True)
      tag_count = utils.order_dictionary(tag_count, 'key', reverse = False)
      char_count = utils.order_dictionary(char_count, 'value', reverse = True)
      prefix_count = utils.order_dictionary(prefix_count, 'value_key', reverse = True)
      suffix_count = utils.order_dictionary(suffix_count, 'value_key', reverse = True)

      pid = 0
      self.prefix_index["-padding-"] = pid
      pid += 1
      for pre, count in prefix_count.items():
        self.prefix_index[pre] = pid
        pid += 1
      self.prefix_index['-UNK-'] = pid
      pickle.dump(self.prefix_index, open("ner_prefix_index", 'wb'))

      sid = 0
      self.suffix_index["-padding-"] = sid
      sid += 1
      for suf, count in suffix_count.items():
        self.suffix_index[suf] = sid
        sid += 1
      self.suffix_index['-UNK-'] = sid
      pickle.dump(self.suffix_index, open("ner_suffix_index", 'wb'))

      wid = 0
      self.word_index["-padding-"] = wid
      wid += 1
      for word, count in word_count.items():
        self.word_index[word] = wid
        wid += 1
        if count <=1:
          self.rare_words.append(word)
      self.word_index['-UNK-'] = wid
      pickle.dump(self.word_index, open("ner_word_index", 'wb'))

      # Ensure that both B- and I- versions exist for ach label
      labels_without_bio = set()
      for label, count in ner_count.items():
        new_label = utils.remove_bio_from_label_name(label)
        labels_without_bio.add(new_label)
      print labels_without_bio
      prefixes = ['B-', 'I-']
      nid = 0
      self.ner_index['O'] = nid
      nid += 1
      for label in labels_without_bio:
        if label == 'O':
          continue
        for prefix in prefixes:
          l = prefix + label
          self.ner_index[l] = nid
          nid += 1
      pickle.dump(self.ner_index, open("ner_index", 'wb'))

      tid = 0
      self.tag_index["-padding-"] = tid
      tid += 1
      for tag, count in tag_count.items():
        self.tag_index[tag] = tid
        tid += 1
      pickle.dump(self.tag_index, open("ner_tag_index", 'wb'))

      cid = 0
      self.char_index["-padding-"] = cid
      cid += 1
      for char, count in char_count.items():
        self.char_index[char] = cid
        cid += 1
      self.char_index['-UNK-'] = cid
      pickle.dump(self.char_index, open("ner_char_index", 'wb'))
    else:
      self.word_index = word_index
      self.tag_index = tag_index
      self.char_index = char_index
      self.ner_index = ner_index
      self.prefix_index = prefix_index
      self.suffix_index = suffix_index
      _, _, _, _, _, _, self.sentence_list = self._parse_dataset(fgen)

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
  data_path1 = "/cs/natlang-user/vivian/wsj-conll/train.conllu"
  data_path2 = "/cs/natlang-user/vivian/wsj-conll/dev.conllu"
  data_path3 = "/cs/natlang-user/vivian/wsj-conll/test.conllu"
  ds = Dataset(data_path1, data_path2, data_path3, use_char=True)
  ds.load_dataset()
  train_file = open("pos.train", "w")
  count = 0
  while ds.has_next_sent('train'):
    sent = ds.get_next_sent('train')
    words = sent.get_word_list()
    ner = sent.get_ner_list()
    for idx, w in enumerate(words):
      train_file.write(w+" "+ner[idx])
      train_file.write("\n")
    train_file.write("\n")
    count += 1
  train_file.close()
  assert count == ds.get_sent_num('train')
  count = 0
  dev_file = open("pos.dev", "w") 
  while ds.has_next_sent('dev'):
    sent = ds.get_next_sent('dev')
    words = sent.get_word_list()
    ner = sent.get_ner_list()
    for idx, w in enumerate(words):
      dev_file.write(w+" "+ner[idx])
      dev_file.write("\n")
    dev_file.write("\n")
    count += 1
  dev_file.close()
  assert count == ds.get_sent_num('dev')
  count = 0
  dev_file = open("test.testb", "w") 
  while ds.has_next_sent('test'):
    sent = ds.get_next_sent('test')
    words = sent.get_word_list()
    ner = sent.get_ner_list()
    for idx, w in enumerate(words):
      dev_file.write(w+" "+ner[idx])
      dev_file.write("\n")
    dev_file.write("\n")
    count += 1
  dev_file.close()
  assert count == ds.get_sent_num('test')



