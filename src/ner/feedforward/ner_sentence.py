import os
import sys
import inspect
import logging
import copy
import string
import utils
from feat_gen import Feat_Gen

UNK = "-UNK-"

class Sentence(object):
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
    self.feat_gen = Feat_Gen(self.get_word_list())

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

  def gen_sent_features(self, word_map, prefix_map=None, suffix_map=None):
    self.word_features, self.cap_features, self.prefix_features, self.suffix_features, self.other_features = \
        self.feat_gen.gen_word_features(word_map, prefix_map=prefix_map, suffix_map=suffix_map)

  def get_history_features(self, unk_tag_id):
    return self.feat_gen.get_history_features(self.cur_tags, unk_tag_id)

  def gen_id_list(self, word2id, ner2id, rare_words=None):
    self.word_ids, self.word_lengths = self.gen_wordid_list(word2id, rare_words)
    self.ner_ids = self.gen_tagid_list(ner2id)
    
  def gen_wordid_list(self, word2id, rare_words):
    word_ids = []
    word_lengths = []
    for word in self.get_word_list():
      if word not in word2id:
        word = UNK
      word_ids.append(word2id[word])
      word_lengths.append(len(word))
    return word_ids, word_lengths

  def gen_tagid_list(self, tag2id):
    tag_ids = []
    for tag in self.get_ner_list():
      tag_ids.append(tag2id[tag])
    return tag_ids

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


