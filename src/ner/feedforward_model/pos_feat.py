import utils

class Pos_Feat():
  '''
  A data structure that represent a sentence 
  '''
  def __init__(self, sent, pos_tags = None):
    self.sent = sent
    self.pos_tags = pos_tags

  def gen_word_features(self, wordMap):
    sent = self.sent
    sent_word_features = []
    sent_cap_features = []
    for i in range(len(sent)):
      word_features = []
      cap_features = []
      if i-4 < 0:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(utils.get_index(sent[i-4], wordMap))
        if sent[i-4].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-3 < 0:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(utils.get_index(sent[i-3], wordMap))
        if sent[i-3].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-2 < 0:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(utils.get_index(sent[i-2], wordMap))
        if sent[i-2].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-1 < 0:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(utils.get_index(sent[i-1],wordMap))
        if sent[i-1].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      word_features.append(utils.get_index(sent[i], wordMap))
      if sent[i].isupper():
        cap_features.append(1)
      else:
        cap_features.append(2)

      sent_len = len(sent)

      if i+1 >= sent_len:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(utils.get_index(sent[i+1], wordMap))
        if sent[i+1].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i+2 >= sent_len:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(utils.get_index(sent[i+2], wordMap))
        if sent[i+2].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i+3 >= sent_len:
        word_features.append(0)
        cap_features.append(0)
      else:
        word_features.append(utils.get_index(sent[i+3], wordMap))
        if sent[i-4].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      sent_word_features.append(word_features)
      sent_cap_features.append(cap_features)
    return sent_word_features, sent_cap_features

  def gen_prefix_features(self, pMap):
    sent = self.sent
    sent_prefix_features = []
    for i in range(len(sent)):
      p_features = []
      if i-4 < 0:
        p_features.append(1)
      else:
        p_features.append(utils.get_index(sent[i-4][:2], pMap))

      if i-3 < 0:
        p_features.append(1)
      else:
        p_features.append(utils.get_index(sent[i-3][:2], pMap))

      if i-2 < 0:
        p_features.append(1)
      else:
        p_features.append(utils.get_index(sent[i-2][:2], pMap))  

      if i-1 < 0:
        p_features.append(1)
      else:
        p_features.append(utils.get_index(sent[i-1][:2], pMap))

      if i+1 >= len(sent):
        p_features.append(1)
      else:
        p_features.append(utils.get_index(sent[i+1][:2], pMap))

      if i+2 >= len(sent):
        p_features.append(1)
      else:
        p_features.append(utils.get_index(sent[i+2][:2], pMap))

      if i+3 >= len(sent):
        p_features.append(1)
      else:
        p_features.append(utils.get_index(sent[i+3][:2], pMap))

      p_features.append(utils.get_index(sent[i][:2], pMap))

      sent_prefix_features.append(p_features)
    return sent_prefix_features

  def gen_suffix_features(self, sMap):
    sent = self.sent
    sent_suffix_features = []
    for i in range(len(sent)):
      s_features = []
      if i-4 < 0:
        s_features.append(1)
      else:
        s_features.append(utils.get_index(sent[i-4][-2:], sMap))
      if i-3 < 0:
        s_features.append(1)
      else:
        s_features.append(utils.get_index(sent[i-3][-2:], sMap))
      if i-2 < 0:
        s_features.append(1)
      else:
        s_features.append(utils.get_index(sent[i-2][-2:], sMap))
      if i-1 < 0:
        s_features.append(1)
      else:
        s_features.append(utils.get_index(sent[i-1][-2:], sMap))

      if i+1 >= len(sent): 
        s_features.append(1)
      else:
        s_features.append(utils.get_index(sent[i+1][-2:], sMap))

      if i+2 >= len(sent): 
        s_features.append(1)
      else:
        s_features.append(utils.get_index(sent[i+2][-2:], sMap))

      if i+3 >= len(sent): 
        s_features.append(1)
      else:
        s_features.append(utils.get_index(sent[i+3][-2:], sMap))

      s_features.append(utils.get_index(sent[i][-2:], sMap))
      sent_suffix_features.append(s_features)
    return sent_suffix_features

  def gen_other_features(self):
    sent = self.sent
    sent_other_features = []
    for i, word in enumerate(sent):
      other_features = []
      if utils._all_digits(word):
        other_features.append(2)
      elif utils._contains_digits(word):
        other_features.append(1)
      else:
        other_features.append(0)

      if not utils._contains_hyphen(word):
        other_features.append(0)
      else:
        other_features.append(1)
      sent_other_features.append(other_features)

    return sent_other_features

  def gen_pos_features(self, pos_map):
    sent_tag_features = []
    sent_len = len(self.pos_tags)
    for i in range(sent_len):
      pos_features = []
      if i-4 < 0:
        pos_features.append(0)
      else:
        pos_features.append(utils.get_index(self.pos_tags[i-4], pos_map))
      if i-3 < 0:
        pos_features.append(0)
      else:
        pos_features.append(utils.get_index(self.pos_tags[i-3], pos_map))

      if i-2 < 0:
        pos_features.append(0)
      else:
        pos_features.append(utils.get_index(self.pos_tags[i-2], pos_map))

      if i-1 < 0:
        pos_features.append(0)
      else:
        pos_features.append(utils.get_index(self.pos_tags[i-1],pos_map))

      pos_features.append(utils.get_index(self.pos_tags[i], pos_map))

      if i+1 >= sent_len:
        pos_features.append(0)
      else:
        pos_features.append(utils.get_index(self.pos_tags[i+1], pos_map))

      if i+2 >= sent_len:
        pos_features.append(0)
      else:
        pos_features.append(utils.get_index(self.pos_tags[i+2], pos_map))

      if i+3 >= sent_len:
        pos_features.append(0)
      else:
        pos_features.append(utils.get_index(self.pos_tags[i+3], pos_map))

      sent_tag_features.append(pos_features)
    return sent_tag_features
