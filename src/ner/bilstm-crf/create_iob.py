from ner_sentence import Sentence
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
    self.format_list = ["a","b","c","FORM","POSTAG","d","e","f","h",
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
      f = codecs.open(path, 'r', 'UTF-8')
      tag = 'O'
      for line in f:
        entity = line.split()
        if len(entity) == format_len and entity[0] != self.comment_sign and '-DOCSTART-' not in line:
          for i in [3,4,10]:
            if i == 10:
              if entity[i][0]=='(' and entity[i][-1]==')':
                entity[i] = u'B-'+entity[i][1:-1]
                tag = 'O'
              elif entity[i][0]=='(':
                tag = entity[i][1:-1]
                entity[i] = u'B-'+tag
                tag = u'I-'+tag
              elif entity[i][-1]==')':
                entity[i] = tag
                tag = 'O'
              else:
                entity[i] = tag
            column_list[self.format_list[i]].append(entity[i])
        else:
          if column_list[self.format_list[10]] != []:
            sentence_list[name].append(Sentence(column_list, self.format_list))
          column_list = {}
          for field in self.format_list:
            column_list[field] = []

    return prefix_count, suffix_count, word_count, tag_count, char_count, ner_count, sentence_list

  def load_dataset(self, word_index=None, tag_index=None, char_index=None, ner_index=None, prefix_index=None, suffix_index=None, fgen=False):
    '''
    dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
    '''
    start_time = time.time()
    prefix_count, suffix_count, word_count, tag_count, char_count, ner_count, self.sentence_list = self._parse_dataset(fgen)
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
  data_path1 = "/cs/natlang-data/conll-2012-processed/conll_2012_english_train"
  data_path2 = "/cs/natlang-data/conll-2012-processed/conll_2012_english_dev"
  data_path3 = "/cs/natlang-data/conll-2012-processed/conll_2012_english_test"
  ds = Dataset(data_path1, data_path2, data_path3, use_char=True)
  ds.load_dataset()
  train_file = codecs.open("engonto.train", "w", encoding="utf-8")
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
    if count % 5000 == 0:
      print count
  train_file.close()
  assert count == ds.get_sent_num('train')
  count = 0
  print "dev"
  dev_file = codecs.open("engonto.testa", "w", encoding="utf-8") 
  while ds.has_next_sent('dev'):
    sent = ds.get_next_sent('dev')
    words = sent.get_word_list()
    ner = sent.get_ner_list()
    for idx, w in enumerate(words):
      dev_file.write(w+" "+ner[idx])
      dev_file.write("\n")
    dev_file.write("\n")
    count += 1
    if count % 5000 == 0:
      print count
  dev_file.close()
  assert count == ds.get_sent_num('dev')
  count = 0
  print "test"
  dev_file = codecs.open("engonto.testb", "w", encoding="utf-8") 
  while ds.has_next_sent('test'):
    sent = ds.get_next_sent('test')
    words = sent.get_word_list()
    ner = sent.get_ner_list()
    for idx, w in enumerate(words):
      dev_file.write(w+" "+ner[idx])
      dev_file.write("\n")
    dev_file.write("\n")
    count += 1
    if count % 5000 == 0:
      print count
  dev_file.close()
  assert count == ds.get_sent_num('test')



