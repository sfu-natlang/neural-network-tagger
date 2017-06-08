from sentence import Sentence
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
    self.format_list = ["ID",
                        "FORM",
                        "LEMMA",
                        "GPOS",
                        "PPOS",
                        "SPLIT_FORM",
                        "SPLIT_LEMMA",
                        "PPOSS",
                        "HEAD",
                        "DEPREL"]
    self.comment_sign = ""
    self.char_map = []
    self.word_map = []
    self.tag_map = []
    self.prefix_map = []
    self.suffix_map = []
    self.index = -1
    self.datapath = datapath

  def _parse_dataset(self, gen_feature):
    prefix_count = collections.defaultdict(lambda: 0)
    suffix_count = collections.defaultdict(lambda: 0)
    word_count = collections.defaultdict(lambda: 0)
    tag_count = collections.defaultdict(lambda: 0)
    char_count = collections.defaultdict(lambda: 0)
    sentence_list = []
    format_len = len(self.format_list)
    column_list = {}
    for field in self.format_list:
      column_list[field] = []
    f = codecs.open(self.datapath, 'r', 'UTF-8')
    for line in f:
      entity = line.split()
      if len(entity) == format_len and entity[0] != self.comment_sign:
        for i in range(format_len):
          column_list[self.format_list[i]].append(str(entity[i]))
        word = column_list["FORM"][-1]
        tag = column_list["PPOS"][-1]
        word_count[word] += 1
        tag_count[tag] += 1
        if gen_feature:
          prefix_count[word[2:]] += 1
          suffix_count[word[:-2]] += 1
          prefix_count[word[3:]] += 1
          suffix_count[word[:-3]] += 1
        else:
          for char in word:
            char_count[char] += 1
      else:
        if column_list[self.format_list[0]] != []:
          sentence_list.append(Sentence(column_list, self.format_list))
        column_list = {}
        for field in self.format_list:
          column_list[field] = []

    return prefix_count, suffix_count, word_count, tag_count, char_count, sentence_list

  def load_dataset(self, word_map=None, tag_map=None, char_map=None, prefix_map=None, suffix_map=None, gen_feature=False):
    '''
    dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
    '''
    start_time = time.time()
    if word_map is None:
      self.tokens_mapped_to_unk = []
      self.UNK = 'UNK'
      prefix_count, suffix_count, word_count, tag_count, char_count, self.sentence_list = self._parse_dataset(gen_feature)

      word_count = utils.order_dictionary(word_count, 'value_key', reverse = True)
      tag_count = utils.order_dictionary(tag_count, 'key', reverse = False)
      char_count = utils.order_dictionary(char_count, 'value', reverse = True)
      prefix_count = utils.order_dictionary(prefix_count, 'value_key', reverse = True)
      suffix_count = utils.order_dictionary(suffix_count, 'value_key', reverse = True)
      if gen_feature:
        for pre, count in prefix_count.items():
          self.prefix_map.append(pre)
        self.prefix_map.append(self.UNK)
        pickle.dump(self.word_map, open("prefix_map", 'wb'))

        for suf, count in suffix_count.items():
          self.suffix_map.append(suf)
        self.suffix_map.append(self.UNK)
        pickle.dump(self.word_map, open("suffix_map", 'wb'))
      else:
        for char, count in char_count.items():
          self.char_map.append(char)
        self.char_map.append(self.UNK)
        pickle.dump(self.char_map, open("char_map", 'wb'))

      #self.word_map.append("-padding-")
      for word, count in word_count.items():
        self.word_map.append(word)
      self.word_map.append(self.UNK)
      pickle.dump(self.word_map, open("word_map", 'wb'))
      #self.tag_map.append("-padding-")
      for tag, count in tag_count.items():
        self.tag_map.append(tag)
      pickle.dump(self.tag_map, open("tag_map", 'wb'))
      self.char_map.append("-padding-")

    else:
      self.word_map = word_map
      self.tag_map = tag_map
      self.char_map = char_map
      self.tag_map = tag_map
      self.char_map = char_map
      self.prefix_map = prefix_map
      self.suffix_map = suffix_map
      _,_,_,_,_,self.sentence_list = self._parse_dataset(gen_feature)

    if gen_feature:
      for sent in self.sentence_list:
        sent.gen_id_list(self.word_map, self.tag_map, self.char_map)
        sent.gen_sent_features(self.word_map, self.prefix_map, self.suffix_map)
      self.prefix_size = len(self.prefix_map)
      self.suffix_size = len(self.suffix_map)
    else:
      for sent in self.sentence_list:
        sent.gen_id_list(self.word_map, self.tag_map, self.char_map)

    self.number_of_classes = len(self.tag_map)
    self.vocabulary_size = len(self.word_map)
    self.alphabet_size = len(self.char_map)


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
  data_path = "test.conllu"
  ds = Dataset(data_path)
  ds.load_dataset()
  print ds.get_sent_num()
  print len(ds.word_map)
  print len(ds.tag_map)
  print len(ds.char_map)

  word_map = pickle.load(open(word_map, rb))

