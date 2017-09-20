import numpy as np
import os
import utils


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
PAD = "$PAD$"
NONE = "O"

class CoNLLDataset(object):
  """
  Class that iterates over CoNLL Dataset

  __iter__ method yields a tuple (words, tags)
    words: list of raw words
    tags: list of raw tags
  If processing_word and processing_tag are not None,
  optional preprocessing is appplied

  Example:
    ```python
    data = CoNLLDataset(filename)
    for sentence, tags in data:
      pass
    ```
  """
  def __init__(self, filename, 
        processing_word=None,
        processing_tag=None, 
        processing_tag_a=None,
        processing_tag_b=None,
        max_iter=None,
        use_char=False):
    """
    Args:
      filename: path to the file
      processing_words: (optional) function that takes a word as input
      processing_tags: (optional) function that takes a tag as input
      max_iter: (optional) max number of sentences to yield
    """
    self.filename = filename
    self.processing_word = processing_word
    self.processing_tag = processing_tag
    self.processing_tag_a = processing_tag_a
    self.processing_tag_b = processing_tag_b
    self.max_iter = max_iter
    self.length = None
    self.use_char = use_char


  def __iter__(self):
    niter = 0
    with open(self.filename) as f:
      words, tags, iob_tags, types, mentions, word_features, char_features = [], [], [], [], [], [], []
      for line in f:
        line = line.strip()
        if (len(line) == 0 or line.startswith("-DOCSTART-")):
          if len(words) != 0:
            niter += 1
            if self.max_iter is not None and niter > self.max_iter:
              break
            mentions = utils.find_mentions(iob_tags)
            word_features, char_features = utils.gen_word_features(words, self.processing_word, self.use_char)
            yield words, tags, iob_tags, types, mentions, word_features, char_features
            words, tags, iob_tags, types, mentions, word_features, char_features = [], [], [], [], [], [], []
        else:
          ls = line.split()
          word, tag = ls[0],ls[-1]
          if tag != 'O':
            tag_a = tag[0]
            tag_b = tag[2:]
          else:
            tag_a = tag
          if self.processing_word is not None:
            word = self.processing_word(word)
          if self.processing_tag is not None:
            tag = self.processing_tag(tag)
          if self.processing_tag_a is not None:
            tag_boudry = self.processing_tag_a(tag_a)
            iob_tags += [tag_boudry]
          if self.processing_tag_b is not None and tag_a == 'B':
            tag_type = self.processing_tag_b(tag_b)
            types += [tag_type]
          words += [word]
          tags += [tag]

  def __len__(self):
    """
    Iterates once over the corpus to set and store length
    """
    if self.length is None:
      self.length = 0
      for _ in self:
        self.length += 1

    return self.length
