import os
import sys
import inspect
import logging
import copy
import string

def get_index(word, wordMap):
  if word in wordMap:
    return wordMap.index(word)
  else:
    return 0

def _all_digits(s):
  return all(char.isdigit() for char in s)

def _contains_digits(s):
  return any(char.isdigit() for char in s)

def _contains_hyphen(s):
  return any(char == "-" for char in s)

def _contains_upper(s):
  return any(char.isupper() for char in s)

def _contains_punc(s):
  return any(char in string.punctuation for char in s)

def process_seg_sent(sent):
  output = []
  for word in sent:
    if word[-2:] == "@@":
      output.append(word[:-2])
    else:
      output.append(word)
  return output

class Sentence():
  '''
  A data structure that represent a sentence 
  '''
  def __init__(self, column_list={}, field_name_list=[], word_map=None, tag_map=None, pre_map=None, suf_map=None, bpe = None):
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
    if bpe is not None:
      self.seg_word_list = []
      self.origin_word_list = []
      self.origin_tag_list = self._fetch_column("PPOS")
      self._bpe_word_list(bpe)
    if pre_map is not None:
      self.prefix_features = self._gen_prefix_features(pre_map)
    if suf_map is not None:
      self.suffix_features = self._gen_suffix_features(suf_map)
    self.word_features, self.cap_features = self._gen_word_features(word_map)
    self.other_features = self._gen_other_features()
    self.tag_map = tag_map

  def _bpe_word_list(self, bpe):
    word_list, self.column_list["PPOS"] = bpe.segment(self.column_list["FORM"], self.column_list["PPOS"])
    self.seg_word_list = word_list
    self.column_list["FORM"] = process_seg_sent(word_list)

  def _gen_word_features(self, wordMap):
    sent = self._fetch_column("FORM")
    sent_word_features = []
    sent_cap_features = []
    for i in range(len(sent)):
      word_features = []
      cap_features = []
      if i-4 < 0:
        word_features.append(1)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i-4], wordMap))
        if sent[i-4].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-3 < 0:
        word_features.append(1)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i-3], wordMap))
        if sent[i-3].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-2 < 0:
        word_features.append(1)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i-2], wordMap))
        if sent[i-2].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-1 < 0:
        word_features.append(1)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i-1],wordMap))
        if sent[i-1].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      word_features.append(get_index(sent[i], wordMap))
      if sent[i].isupper():
        cap_features.append(1)
      else:
        cap_features.append(2)

      sent_len = len(sent)

      if i+1 >= sent_len:
        word_features.append(2)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i+1], wordMap))
        if sent[i+1].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i+2 >= sent_len:
        word_features.append(2)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i+2], wordMap))
        if sent[i+2].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i+3 >= sent_len:
        word_features.append(2)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i+3], wordMap))
        if sent[i-4].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      sent_word_features.append(word_features)
      sent_cap_features.append(cap_features)
    return sent_word_features, sent_cap_features

  def _gen_prefix_features(self, pMap):
    sent = self._fetch_column("FORM")
    sent_prefix_features = []
    for i in range(len(sent)):
      p_features = []
      if i-4 < 0:
        p_features.append(1)
      else:
        p_features.append(get_index(sent[i-4][:2], pMap))

      if i-3 < 0:
        p_features.append(1)
      else:
        p_features.append(get_index(sent[i-3][:2], pMap))

      if i-2 < 0:
        p_features.append(1)
      else:
        p_features.append(get_index(sent[i-2][:2], pMap))  

      if i-1 < 0:
        p_features.append(1)
      else:
        p_features.append(get_index(sent[i-1][:2], pMap))

      if i+1 >= len(sent):
        p_features.append(1)
      else:
        p_features.append(get_index(sent[i+1][:2], pMap))

      if i+2 >= len(sent):
        p_features.append(1)
      else:
        p_features.append(get_index(sent[i+2][:2], pMap))

      if i+3 >= len(sent):
        p_features.append(1)
      else:
        p_features.append(get_index(sent[i+3][:2], pMap))

      p_features.append(get_index(sent[i][:2], pMap))

      sent_prefix_features.append(p_features)
    return sent_prefix_features

  def _gen_suffix_features(self, sMap):
    sent = self._fetch_column("FORM")
    sent_suffix_features = []
    for i in range(len(sent)):
      s_features = []
      if i-4 < 0:
        s_features.append(1)
      else:
        s_features.append(get_index(sent[i-4][-2:], sMap))
      if i-3 < 0:
        s_features.append(1)
      else:
        s_features.append(get_index(sent[i-3][-2:], sMap))
      if i-2 < 0:
        s_features.append(1)
      else:
        s_features.append(get_index(sent[i-2][-2:], sMap))
      if i-1 < 0:
        s_features.append(1)
      else:
        s_features.append(get_index(sent[i-1][-2:], sMap))

      if i+1 >= len(sent): 
        s_features.append(1)
      else:
        s_features.append(get_index(sent[i+1][-2:], sMap))

      if i+2 >= len(sent): 
        s_features.append(1)
      else:
        s_features.append(get_index(sent[i+2][-2:], sMap))

      if i+3 >= len(sent): 
        s_features.append(1)
      else:
        s_features.append(get_index(sent[i+3][-2:], sMap))

      s_features.append(get_index(sent[i][-2:], sMap))
      sent_suffix_features.append(s_features)
    return sent_suffix_features

  def _gen_other_features(self):
    sent = self._fetch_column("FORM")
    sent_other_features = []
    for i, word in enumerate(sent):
      other_features = []
      if _all_digits(word):
        other_features.append(2)
      elif _contains_digits(word):
        other_features.append(1)
      else:
        other_features.append(0)

      if not _contains_hyphen(word):
        other_features.append(0)
      else:
        other_features.append(1)
      sent_other_features.append(other_features)

    return sent_other_features


  def gen_tag_features(self, state):
    tag_features = []
    if state - 4 < 0:
      tag_features.append(45)
    else:
      tag_features.append(get_index(self.cur_tags[state-4], self.tag_map))
    if state - 3 < 0:
      tag_features.append(45)
    else:
      tag_features.append(get_index(self.cur_tags[state-3], self.tag_map))
    if state - 2 < 0:
      tag_features.append(45)
    else:
      tag_features.append(get_index(self.cur_tags[state-2], self.tag_map))
    if state - 1 < 0:
      tag_features.append(45)
    else:
      tag_features.append(get_index(self.cur_tags[state-1], self.tag_map))
    return tag_features

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

  def get_tag_list(self):
    return self._fetch_column("PPOS")

  def get_word_list(self):
    return self._fetch_column("FORM")

  def set_tag(self, tag):
    self.cur_tags.append(tag)

  def get_tag_output(self):
    return self.output_tags

  def set_epoch(self, epoch):
    self.epoch = epoch

  def get_epoch(self):
    return self.epoch



