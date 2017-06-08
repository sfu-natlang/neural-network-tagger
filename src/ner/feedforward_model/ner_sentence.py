import os
import sys
import inspect
import logging
import copy
import string
import utils
import sklearn.preprocessing
from pos_feat import Pos_Feat


class Sentence():
  '''
  A data structure that represent a sentence 
  '''
  def __init__(self, column_list={}, field_name_list=[]):
    '''
    initialize a sentence
    :param column_list: a dict of data column
    :type column_list: dict(list)

    :param field_name_list: a list of the data columns
    :type field_name_list: list(str)
    '''
    self.column_list = column_list
    self.field_name_list = field_name_list
    self.index = -1
    self.cur_tags = []
    self.output_tags = []
    self.epoch = -1

  def has_next_state(self):
    index = self.state + 1
    if index >= len(self.get_word_list()):
      return False
    else:
      return True

  def get_next_state(self):
    if self.has_next_state():
      self.state += 1
      return self.state
    raise IndexError("Run out of data while calling get_next_sent()")

  def reset_state(self):
    self.state = -1

  def gen_sent_features(self, word_map, pos_map, prefix_map, suffix_map):
    feat_gen = Pos_Feat(self.get_word_list(), self.get_pos_list())
    self.word_features, self.cap_features = feat_gen.gen_word_features(word_map)
    self.prefix_features = feat_gen.gen_prefix_features(prefix_map)
    self.suffix_features = feat_gen.gen_suffix_features(suffix_map)
    self.pos_features = feat_gen.gen_pos_features(pos_map)
    #self.other_features = feat_gen.gen_other_features()

  def gen_id_list(self, word_map, pos_map, char_map, ner_map):
    self.char_lists = self.gen_char_list(char_map)
    self.word_ids, self.word_lengths = self.gen_wordid_list(word_map)
    self.ner_ids = self.gen_nerid_list(ner_map)
    self.pos_ids = self.gen_posid_list(pos_map)

  def gen_char_list(self, char_map):
    sent_char_lists = []
    word_list = self.get_word_list()
    for w in word_list:
      char_list = []
      for c in w:
        char_list.append(utils.get_index(c, char_map))
      sent_char_lists.append(char_list)
    return sent_char_lists

  def gen_wordid_list(self, word_map):
    word_ids = []
    word_lengths = []
    for word in self.get_word_list():
      if utils._contains_digits(word):
        word = utils.replace_digits(word)
      word_ids.append(utils.get_index(word, word_map))
      word_lengths.append(len(word))
    return word_ids, word_lengths

  def gen_posid_list(self, tag_map):
    tag_ids = []
    for tag in self.get_pos_list():
      tag_ids.append(utils.get_index(tag, tag_map))
    return tag_ids

  def gen_nerid_list(self, tag_map):
    tag_ids = []
    for tag in self.get_ner_list():
      tag_ids.append(utils.get_index(tag, tag_map))
    return tag_ids

  def _gen_word_features(self, wordMap):
    sent = self._fetch_column("FORM")
    sent_cap_features = []
    for i in range(len(sent)):
      if sent[i].isupper():
        sent_cap_features.append(1)
      else:
        sent_cap_features.append(2)

    return sent_cap_features

  def _gen_prefix_features(self, pMap):
    sent = self._fetch_column("FORM")
    sent_prefix_features = []
    for i in range(len(sent)):
      sent_prefix_features.append(utils.get_index(sent[i][:2], pMap))
    return sent_prefix_features

  def _gen_suffix_features(self, sMap):
    sent = self._fetch_column("FORM")
    sent_suffix_features = []
    for i in range(len(sent)):
      sent_suffix_features.append(utils.get_index(sent[i][-2:], sMap))
    return sent_suffix_features

  def _gen_other_features(self):
    sent = self._fetch_column("FORM")
    sent_other_features = []
    for word in sent:
      if _all_digits(word):
        sent_other_features.append(1)
      elif _contains_digits(word):
        sent_other_features.append(2)
      else:
        sent_other_features.append(3)

    return sent_other_features


  def get_sent_len(self):
    return len(self.column_list["FORM"])

  def _fetch_column(self, field_name):
    '''
    return the column given the field_name
    '''
    if field_name in self.field_name_list:
      return self.column_list[field_name]
    else:
      raise RuntimeError("SENTENCE [ERROR]: '" + field_name + "' is needed in Sentence but it's not in format file")

  def get_next_state(self):
    if self.has_next_state():
      self.index +=1 
      return self.index
    raise IndexError("Run out of sentence state while calling get_next_state()")

  def has_next_state(self):
    index = self.index + 1
    if index >= self.get_sent_len():
      return False
    else:
      return True

  def reset_state(self):
    self.index = -1
    self.output_tags = self.cur_tags
    self.cur_tags = []

  def get_pos_list(self):
    return self._fetch_column("POSTAG")

  def get_ner_list(self):
    return self._fetch_column("NER")

  def get_word_list(self):
    return self._fetch_column("FORM")

  def set_tag(self, tag):
    self.cur_tags.append(tag)

  def set_sent_tags(self, tags):
    sent_len = self.get_sent_len()
    self.output_tags = tags[:sent_len]

  def get_tag_output(self):
    return self.output_tags

  def set_epoch(self, epoch):
    self.epoch = epoch

  def get_epoch(self):
    return self.epoch

  def get_chunks(self):
    return self._fetch_column("CHUNKS")


