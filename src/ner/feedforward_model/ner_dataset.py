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
  """A class for handling CoNLL data sets."""

  def __init__(self, train_datapath="train.conllu", dev_datapath=None, test_datapath=None):
    self.format_list = ["FORM",
                        "POSTAG",
                        "CHUNK",
                        "NER"]
    self.comment_sign = ""
    self.char_map = []
    self.word_map = []
    self.tag_map = []
    self.ner_map = []
    self.prefix_map = []
    self.suffix_map = []
    self.index = {}
    self.data_path = {}
    self.data_path['train'] = train_datapath
    self.data_path['dev'] = dev_datapath
    self.data_path['test'] = test_datapath
    self.index['train'] = -1
    self.index['dev'] = -1
    self.index['test'] = -1

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
          word = column_list["FORM"][-1]
          tag = column_list["POSTAG"][-1]
          ner = column_list["NER"][-1]
          if utils._contains_digits(word):
            word = utils.replace_digits(word)
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

  def load_dataset(self, word_map=None, tag_map=None, char_map=None, ner_map=None, prefix_map=None, suffix_map=None, fgen=False):
    '''
    dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
    '''
    start_time = time.time()
    if word_map is None:
      self.tokens_mapped_to_unk = []
      self.UNK = 'UNK'
      prefix_count, suffix_count, word_count, tag_count, char_count, ner_count, self.sentence_list = self._parse_dataset(fgen)

      word_count = utils.order_dictionary(word_count, 'value_key', reverse = True)
      tag_count = utils.order_dictionary(tag_count, 'key', reverse = False)
      ner_count = utils.order_dictionary(ner_count, 'key', reverse = False)
      char_count = utils.order_dictionary(char_count, 'value', reverse = True)
      prefix_count = utils.order_dictionary(prefix_count, 'value_key', reverse = True)
      suffix_count = utils.order_dictionary(suffix_count, 'value_key', reverse = True)

      self.prefix_map.append("-padding-")
      for pre, count in prefix_count.items():
        self.prefix_map.append(pre)
      self.prefix_map.append(self.UNK)
      pickle.dump(self.prefix_map, open("ner_prefix_map", 'wb'))

      self.suffix_map.append("-padding-")
      for suf, count in suffix_count.items():
        self.suffix_map.append(suf)
      self.suffix_map.append(self.UNK)
      pickle.dump(self.suffix_map, open("ner_suffix_map", 'wb'))
      
      self.word_map.append("-padding-")
      for word, count in word_count.items():
        self.word_map.append(word)
      self.word_map.append(self.UNK)
      pickle.dump(self.word_map, open("ner_word_map", 'wb'))

      # Ensure that both B- and I- versions exist for ach label
      labels_without_bio = set()
      for label, count in ner_count.items():
        new_label = utils.remove_bio_from_label_name(label)
        labels_without_bio.add(new_label)
      print labels_without_bio
      prefixes = ['B-', 'I-']
      self.ner_map.append('O')
      for label in labels_without_bio:
        if label == 'O':
          continue
        for prefix in prefixes:
          l = prefix + label
          self.ner_map.append(l)
      pickle.dump(self.ner_map, open("ner_ner_map", 'wb'))

      self.tag_map.append("-padding-")
      for tag, count in tag_count.items():
        self.tag_map.append(tag)
      pickle.dump(self.tag_map, open("ner_tag_map", 'wb'))

      self.char_map.append("-padding-")
      for char, count in char_count.items():
        self.char_map.append(char)
      self.char_map.append(self.UNK)
      pickle.dump(self.char_map, open("ner_char_map", 'wb'))
    else:
      self.word_map = word_map
      self.tag_map = tag_map
      self.char_map = char_map
      self.tag_map = tag_map
      self.char_map = char_map
      self.ner_map = ner_map
      self.prefix_map = prefix_map
      self.suffix_map = suffix_map
      _, _, _, _, _, _, self.sentence_list = self._parse_dataset(fgen)
    for name, sent_list in self.sentence_list.items():
      for sent in sent_list:
        sent.gen_id_list(self.word_map, self.tag_map, self.char_map, self.ner_map)
        sent.gen_sent_features(self.word_map, self.tag_map, self.prefix_map, self.suffix_map)

    self.number_of_classes = len(self.ner_map)
    self.vocabulary_size = len(self.word_map)
    self.alphabet_size = len(self.char_map)
    self.pos_classes = len(self.tag_map)
    self.prefix_size = len(self.prefix_map)
    self.suffix_size = len(self.suffix_map)

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
  data_path1 = "/cs/natlang-data/CoNLL/CoNLL-2003/eng.testa"
  data_path2 = "/cs/natlang-data/CoNLL/CoNLL-2003/eng.testa"
  data_path3 = "/cs/natlang-data/CoNLL/CoNLL-2003/eng.testa"
  ds = Dataset(data_path1, data_path2, data_path3)
  ds.load_dataset()
  print ds.get_sent_num('train')
  print ds.get_sent_num('dev')
  print ds.get_sent_num('test')
  print len(ds.word_map)
  print len(ds.tag_map)
  print len(ds.char_map)
  print len(ds.ner_map)
  print len(ds.suffix_map)
  print len(ds.prefix_map)


