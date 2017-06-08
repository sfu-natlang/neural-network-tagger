import os
import sys
import inspect
import logging
import copy
import string
import sklearn.preprocessing

def get_index(word, wordMap):
  if word in wordMap:
    return wordMap.index(word)
  else:
    return len(wordMap)-1

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


class Sentence():
  '''
  A data structure that represent a sentence 
  '''
  def __init__(self, column_list={}, field_name_list=[], word_map=None, tag_map=None, pre_map=None, suf_map=None):
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
    self.h_emb = []
    self.epoch = -1
    if pre_map is not None:
      self.prefix_features = self._gen_prefix_features(pre_map)
    if suf_map is not None:
      self.suffix_features = self._gen_suffix_features(suf_map)
    self.word_features, self.cap_features = self._gen_word_features(word_map)
    self.other_features = self._gen_other_features()
    self.tag_map = tag_map
    self.word_map = word_map
    self.char_list = []
    self.word_length = []
    self.wordid_list = self._gen_wordid_list()
    self.tagid_list = self._gen_tagid_list()
    self.label_vector = self._gen_label_vector()

  def gen_char_list(self, char_map):
    word_list = self.get_word_list()
    for w in word_list:
      char_list = []
      for c in w:
        char_list.append(get_index(c, char_map))
      self.char_list.append(char_list)

  def _gen_label_vector(self):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(len(self.tag_map)))
    return label_binarizer.transform(self.tagid_list)

  def _gen_wordid_list(self):
    wordid = []
    for word in self._fetch_column("FORM"):
      wordid.append(get_index(word, self.word_map))
      self.word_length.append(len(word))
    return wordid

  def _gen_tagid_list(self):
    tagid = []
    for tag in self._fetch_column("PPOS"):
      tagid.append(get_index(tag, self.tag_map))
    return tagid

  def _gen_word_features(self, wordMap):
    sent = self._fetch_column("FORM")
    sent_word_features = []
    sent_cap_features = []
    for i in range(len(sent)):
      word_features = []
      cap_features = []
      if i-4 < 0:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i-4], wordMap))
        if sent[i-4].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-3 < 0:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i-3], wordMap))
        if sent[i-3].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-2 < 0:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i-2], wordMap))
        if sent[i-2].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-1 < 0:
        word_features.append(0)
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
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i+1], wordMap))
        if sent[i+1].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i+2 >= sent_len:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(get_index(sent[i+2], wordMap))
        if sent[i+2].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i+3 >= sent_len:
        word_features.append(0)
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

  def set_h_emb(self, h):
    self.h_emb.append(h)

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


