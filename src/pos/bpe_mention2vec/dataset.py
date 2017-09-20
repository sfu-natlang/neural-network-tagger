import utils
import collections
import codecs
import re
import time
import token
import os
import pickle
from apply_bpe import BPE
import argparse

class Dataset(object):
  """A class for handling pos data sets."""

  def __init__(self, train_datapath=None, dev_datapath=None, test_datapath=None, use_char=False, bpe=None):
    self.format_list = ["ID",
                        "FORM",
                        "LEMMA",
                        "GPOS",
                        "POSTAG",
                        "SPLIT_FORM",
                        "SPLIT_LEMMA",
                        "PPOSS",
                        "HEAD",
                        "DEPREL"]
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
    self.bpe = bpe


  def _parse_dataset(self):
    word_count = collections.defaultdict(lambda: 0)
    tag_count = collections.defaultdict(lambda: 0)
    char_count = collections.defaultdict(lambda: 0)
    prefix_count = collections.defaultdict(lambda: 0)
    suffix_count = collections.defaultdict(lambda: 0)
   
    format_len = len(self.format_list)
    column_list = {}
    for field in self.format_list:
      column_list[field] = []
    start = time.time()
    for name, path in self.data_path.items():
      f = codecs.open(path, 'r', 'UTF-8')
      w = codecs.open(path+"_seg", 'w')
      for line in f:
        entity = line.split()
        if len(entity) == format_len:
          for i in range(format_len):
            column_list[self.format_list[i]].append(str(entity[i]))
          if self.bpe == None:
            word = column_list["FORM"][-1]
            tag = column_list["POSTAG"][-1]
            word_count[word] += 1
            tag_count[tag] += 1
            prefix_count[word[2:]] += 1
            suffix_count[word[:-2]] += 1
            prefix_count[word[3:]] += 1
            suffix_count[word[:-3]] += 1
            for char in word:
              char_count[char] += 1
        else:
          if bpe is not None:
            column_list["FORM"], column_list["POSTAG"] = bpe.segment(column_list["FORM"], column_list["POSTAG"]) 
            for idx, val in enumerate(column_list["FORM"]):
              w.write(val+" "+column_list["POSTAG"][idx])
              w.write("\n")
            w.write("\n")
          column_list = {}
          for field in self.format_list:
            column_list[field] = []
      f.close()
      w.close()
      print time.time() - start
    return prefix_count, suffix_count, word_count, tag_count, char_count

  def load_dataset(self, word_index=None, tag_index=None, char_index=None, ner_index=None, prefix_index=None, suffix_index=None):
    '''
    dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
    '''
    start_time = time.time()
    self.tokens_mapped_to_unk = []
    prefix_count, suffix_count, word_count, tag_count, char_count = self._parse_dataset()
    word_count = utils.order_dictionary(word_count, 'value_key', reverse = True)
    tag_count = utils.order_dictionary(tag_count, 'key', reverse = False)
    char_count = utils.order_dictionary(char_count, 'value', reverse = True)
    prefix_count = utils.order_dictionary(prefix_count, 'value_key', reverse = True)
    suffix_count = utils.order_dictionary(suffix_count, 'value_key', reverse = True)
    if self.bpe is None:
      self.write_vocab(word_count, "word_count")

  def write_vocab(self, vocab, filename):
    print("Writing vocab...")
    with open(filename, "w") as f:
      for key, value in vocab.items():
        f.write("{0} {1}\n".format(key, value))
    print("- done. {} tokens".format(len(vocab)))
def create_parser():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description="learn BPE-based word segmentation")

  parser.add_argument('--eval', '-e', action="store_true", default=False)

  return parser

if __name__ == '__main__':
  parser = create_parser()
  args = parser.parse_args()
  data_path1 = "/cs/natlang-user/vivian/wsj-conll/train.conllu"
  data_path2 = "/cs/natlang-user/vivian/wsj-conll/dev.conllu"
  data_path3 = "/cs/natlang-user/vivian/wsj-conll/test.conllu"
  '''
  if args.eval:
    print "!!!"
  else:
    print "xxx"
  bpe = BPE(codecs.open("codes_file", encoding='utf-8'))
  ds = Dataset(data_path1, data_path2, data_path3, bpe=bpe)
  ds.load_dataset()
  '''

