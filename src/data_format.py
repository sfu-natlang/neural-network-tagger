from sentence import Sentence

class ConllData():
  def __init__(self, data_path=''):
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
    self._get_sentence_list(data_path)
    self.index = -1

  def _get_sentence_list(self, data_path):
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
            self.sentence_list.append(Sentence(column_list, self.format_list))
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
  data_path = "test.conllu"
  conllData = ConllData(data_path)
  print conllData.get_sent_num()






