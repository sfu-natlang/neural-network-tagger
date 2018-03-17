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

  def __init__(self, train_datapath="train.conllu", dev_datapath=None, test_datapath=None, use_char=False):
    self.format_list = ["FORM",
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
    if test_datapath is not None:
      self.data_path['test'] = test_datapath
    self.index['train'] = -1
    self.index['dev'] = -1
    self.index['test'] = -1
    self.rare_words = []

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
      with open(path) as f:
        for line in f:
          entity = line.split()
          if (len(entity) == 0 or line.startswith("-DOCSTART-")):
            if column_list[self.format_list[-1]] != []:
              sentence_list[name].append(Sentence(column_list, self.format_list))
            column_list = {}
            for field in self.format_list:
              column_list[field] = []
          else:
            #print entity
            for i in range(format_len):
              column_list[self.format_list[i]].append(entity[i])
            if fgen:
              word = column_list["FORM"][-1]
              ner = column_list["NER"][-1]
              word_count[word] += 1
              ner_count[ner] += 1
              prefix_count[word[2:]] += 1
              suffix_count[word[:-2]] += 1
              prefix_count[word[3:]] += 1
              suffix_count[word[:-3]] += 1
              for char in word:
                char_count[char] += 1

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
      pickle.dump(self.prefix_index, open(os.path.join("lstm_ner_models", "ner_prefix_index"), 'wb'))

      sid = 0
      self.suffix_index["-padding-"] = sid
      sid += 1
      for suf, count in suffix_count.items():
        self.suffix_index[suf] = sid
        sid += 1
      self.suffix_index['-UNK-'] = sid
      pickle.dump(self.suffix_index, open(os.path.join("lstm_ner_models", "ner_suffix_index"), 'wb'))
      
      wid = 0
      self.word_index["-padding-"] = wid
      wid += 1
      for word, count in word_count.items():
        self.word_index[word] = wid
        wid += 1
        if count <=1:
          self.rare_words.append(word)
      self.word_index['-UNK-'] = wid
      pickle.dump(self.word_index, open(os.path.join("lstm_ner_models", "ner_word_index"), 'wb'))

      # Ensure that both B- and I- versions exist for ach label
      labels_without_bio = set()
      for label, count in ner_count.items():
        new_label = utils.remove_bio_from_label_name(label)
        labels_without_bio.add(new_label)

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
      pickle.dump(self.ner_index, open(os.path.join("lstm_ner_models", "ner_index"), 'wb'))
      '''
      tid = 0
      self.tag_index["-padding-"] = tid
      tid += 1
      for tag, count in tag_count.items():
        self.tag_index[tag] = tid
        tid += 1
      pickle.dump(self.tag_index, open("ner_tag_index", 'wb'))
      '''
      cid = 0
      self.char_index["-padding-"] = cid
      cid += 1
      for char, count in char_count.items():
        self.char_index[char] = cid
        cid += 1
      self.char_index['-UNK-'] = cid
      pickle.dump(self.char_index, open(os.path.join("lstm_ner_models", "ner_char_index"), 'wb'))
    else:
      self.word_index = word_index
      self.tag_index = tag_index
      self.char_index = char_index
      self.ner_index = ner_index
      self.prefix_index = prefix_index
      self.suffix_index = suffix_index
      _, _, _, _, _, _, self.sentence_list = self._parse_dataset(fgen)

    for name, sent_list in self.sentence_list.items():
      for sent in sent_list:
        sent.gen_id_list(self.word_index, self.char_index, self.ner_index, None)
        if not self.use_char:
          sent.gen_sent_features(self.word_index, prefix_map=self.prefix_index, suffix_map=self.suffix_index)

    self.number_of_classes = len(self.ner_index)
    self.vocabulary_size = len(self.word_index)
    if self.char_index != None:
      self.alphabet_size = len(self.char_index)
    else:
      self.alphabet_size = 0
    #self.pos_classes = len(self.tag_index)
    self.number_of_boi = 3
    self.number_of_type = 4
    self.prefix_size = len(self.prefix_index)
    self.suffix_size = len(self.suffix_index)

    if self.char_index != None:
      self.char_map = utils.reverse_dictionary(self.char_index)
    self.word_map = utils.reverse_dictionary(self.word_index)
    #self.tag_map = utils.reverse_dictionary(self.tag_index)
    self.ner_map = utils.reverse_dictionary(self.ner_index)

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
  test_data_path = "/cs/natlang-user/vivian/eng.testb"
  dataset = Dataset(test_data_path)
  dataset.load_dataset()
  name = "train"
  output_file = codecs.open("output", 'w', 'UTF-8')
  while dataset.has_next_sent(name):
    sent = dataset.get_next_sent(name)
    words = sent.get_word_list()
    output_string = ""
    for a in words:
      output_string += a+ '\n'
    output_file.write(output_string+'\n')
  output_file.close()

