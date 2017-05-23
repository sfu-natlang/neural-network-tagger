from sentence import Sentence
import pickle

def readMap(path):
  ret = []
  with open(path, 'rb') as f:
    for idx, line in enumerate(f):
      if idx == 0:
        continue
      ret.append(line.split()[0])
  return ret

def readAffix(path):
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  return ret

class ConllData():
  def __init__(self, data_path='', word_map=None, tag_map=None, pre_map=None, suf_map=None, bpe=None):
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
    self.sentence_list = []
    self._get_sentence_list(data_path, word_map, tag_map, pre_map, suf_map, bpe)
    self.index = -1

  def _get_sentence_list(self, data_path, word_map, tag_map, pre_map, suf_map, bpe):
    format_len = len(self.format_list)
    column_list = {}
    for field in self.format_list:
      column_list[field] = []

    with open(data_path) as dp:
      for line in dp:
        entity = line.split()
        if len(entity) == format_len and entity[0] != self.comment_sign:
          for i in range(format_len):
            column_list[self.format_list[i]].append(str(entity[i].encode('utf-8')))
        else:
          if column_list[self.format_list[0]] != []:
            self.sentence_list.append(Sentence(column_list, self.format_list, word_map, tag_map, pre_map, suf_map, bpe))
          column_list = {}
          for field in self.format_list:
            column_list[field] = []

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
  wordMapPath = "word-map"
  tagMapPath = "tag-map"
  pMapPath2 = "prefix-list"
  sMapPath2 = "suffix-list"
  pMap2 = readAffix(pMapPath2)
  sMap2 = readAffix(sMapPath2)

  wordMap = readMap(wordMapPath)
  tagMap = readMap(tagMapPath)

  wordMap.insert(0,"-start-")
  wordMap.insert(0,"-end-")
  wordMap.insert(0,"-unknown-")

  pMap2.insert(0,"-start-")
  pMap2.insert(0,"-unknown-")
  sMap2.insert(0,"-start-")
  sMap2.insert(0,"-unknown-")

  data_path = "/cs/natlang-user/vivian/wsj-conll/test.conllu"
  conllData = ConllData(data_path, wordMap, tagMap, pMap2, sMap2)
  print conllData.get_sent_num()






