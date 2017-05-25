from ner_sentence import Sentence
import utils
import collections
import codecs
import re
import time
import token
import os
import pickle


class Dataset(object):
  """A class for handling pos data sets."""

  def __init__(self, datapath="train.conllu"):
    self.format_list = ["FORM",
                        "POSTAG",
                        "CHUNK",
                        "NER"]
    self.comment_sign = ""
    self.char_map = []
    self.word_map = []
    self.tag_map = []
    self.ner_map = []
    self.index = -1
    self.datapath = datapath

  def _parse_dataset(self):
    word_count = collections.defaultdict(lambda: 0)
    tag_count = collections.defaultdict(lambda: 0)
    char_count = collections.defaultdict(lambda: 0)
    ner_count = collections.defaultdict(lambda: 0)
    sentence_list = []
    format_len = len(self.format_list)
    column_list = {}
    for field in self.format_list:
      column_list[field] = []
    f = codecs.open(self.datapath, 'r', 'UTF-8')
    for line in f:
      entity = line.split()
      if len(entity) == format_len and entity[0] != self.comment_sign and '-DOCSTART-' not in line:
        for i in range(format_len):
          column_list[self.format_list[i]].append(str(entity[i]))
        word = column_list["FORM"][-1]
        tag = column_list["POSTAG"][-1]
        ner = column_list["NER"][-1]
        word_count[word] += 1
        tag_count[tag] += 1
        ner_count[ner] += 1
        for char in word:
          char_count[char] += 1
      else:
        if column_list[self.format_list[0]] != []:
          sentence_list.append(Sentence(column_list, self.format_list))
        column_list = {}
        for field in self.format_list:
          column_list[field] = []

    return word_count, tag_count, char_count, ner_count, sentence_list

  def load_dataset(self, word_map=None, tag_map=None, char_map=None, ner_map=None):
    '''
    dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
    '''
    start_time = time.time()
    if word_map is None:
      self.tokens_mapped_to_unk = []
      self.UNK = 'UNK'
      word_count, tag_count, char_count, ner_count, self.sentence_list = self._parse_dataset()

      word_count = utils.order_dictionary(word_count, 'value_key', reverse = True)
      tag_count = utils.order_dictionary(tag_count, 'key', reverse = False)
      ner_count = utils.order_dictionary(ner_count, 'key', reverse = False)
      char_count = utils.order_dictionary(char_count, 'value', reverse = True)
      self.word_map.append("-padding-")
      for word, count in word_count.items():
        self.word_map.append(word)
      self.word_map.append(self.UNK)
      #pickle.dump(self.word_map, open("word_map", 'wb'))
      self.word_size = len(self.word_map)
      self.ner_map.append("-padding-")
      for ner, count in ner_count.items():
        self.ner_map.append(ner)
      self.ner_size = len(self.ner_map)
      self.tag_map.append("-padding-")
      for tag, count in tag_count.items():
        self.tag_map.append(tag)
      #pickle.dump(self.tag_map, open("tag_map", 'wb'))
      self.tag_size = len(self.tag_map)
      self.char_map.append("-padding-")
      for char, count in char_count.items():
        self.char_map.append(char)
      self.char_map.append(self.UNK)
      #pickle.dump(self.char_map, open("char_map", 'wb'))
      self.char_size = len(self.char_map)
    else:
      self.word_map = word_map
      self.tag_map = tag_map
      self.char_map = char_map
      self.tag_map = tag_map
      self.char_map = char_map
      self.ner_map = ner_map
      self.word_size = len(self.word_map)
      self.tag_size = len(self.tag_map)
      self.char_size = len(self.char_map)
      self.ner_size = len(self.ner_map)
      word_count, tag_count, char_count, ner_count, self.sentence_list = self._parse_dataset()
    for sent in self.sentence_list:
      sent.gen_id_list(self.word_map, self.tag_map, self.char_map, self.ner_map)
      self.number_of_classes = len(self.ner_map)
      self.vocabulary_size = len(self.word_map)
      self.alphabet_size = len(self.char_map)
      self.pos_classes = len(self.tag_map)

    elapsed_time = time.time() - start_time
    print('done ({0:.2f} seconds)'.format(elapsed_time))

  def get_sent_num(self):
    return len(self.sentence_list)

  def get_next_sent(self):
    if self.has_next_sent():
      self.index += 1
      return self.sentence_list[self.index]
    raise IndexError("Run out of data while calling get_next_sent()")

  def has_next_sent(self):
    index = self.index + 1
    if index >= len(self.sentence_list):
      return False
    else:
      return True

  def reset_index(self):
    self.index = -1

if __name__ == '__main__':
  data_path = "/cs/natlang-data/CoNLL/CoNLL-2003/eng.train"
  ds = Dataset(data_path)
  ds.load_dataset()
  print ds.get_sent_num()
  print len(ds.word_map)
  print len(ds.tag_map)
  print len(ds.char_map)
  print len(ds.ner_map)

  data_path2 = "/cs/natlang-data/CoNLL/CoNLL-2003/eng.testa"
  test_data = Dataset(data_path2)
  test_data.load_dataset(ds.word_map, ds.tag_map, ds.char_map, ds.ner_map)
  while test_data.has_next_sent():
    sent = test_data.get_next_sent()
    print sent.get_word_list(), sent.word_ids
    print sent.pos_ids
    print sent.char_lists
    print sent.ner_ids
    break
  print test_data.get_sent_num()

