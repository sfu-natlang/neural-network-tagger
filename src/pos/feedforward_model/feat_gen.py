import utils

class Feat_Gen():
  '''
  A data structure that represent a feature generator
  '''
  def __init__(self, sent, pos_tags = None):
    self.sent = sent
    self.pos_tags = pos_tags

  def get_history_features(self, cur_tags):
    history = []
    l = len(cur_tags)
    if l < 4:
      history.extend([45]*(4-l))
      history.extend(cur_tags)
    else:
      history = cur_tags[-4:]
    return history

  def gen_word_features(self, word_map, pos_map, prefix_map = None, suffix_map = None):
    sent = self.sent
    pos_tags = self.pos_tags
    sent_word_features = []
    sent_cap_features = []
    sent_prefix_features = []
    sent_suffix_features = []
    sent_other_features = []
    for i in range(len(sent)):
      word_features = []
      cap_features = []
      prefix_features = []
      suffix_features = []
      pos_features = []
      other_features = []
      if i-4 < 0:
        word_features.append(0)
        cap_features.append(0)
        prefix_features.append(0)
        suffix_features.append(0)
        #pos_features.append(0)
      else:
        word_features.append(word_map[sent[i-4]])
        #pos_features.append(pos_map[pos_tags[i-4]])
        
        pre = sent[i-4][:2]
        if pre in prefix_map:
          prefix_features.append(prefix_map[pre])
        else :
          prefix_features.append(prefix_map['-UNK-'])
        suf = sent[i-4][-2:]
        if suf in suffix_map:
          suffix_features.append(suffix_map[suf])
        else:
          suffix_features.append(suffix_map['-UNK-'])
        if sent[i-4].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)
        

      if i-3 < 0:
        word_features.append(0)
        cap_features.append(0)
        prefix_features.append(0)
        suffix_features.append(0)
        #pos_features.append(0)
      else:
        word_features.append(word_map[sent[i-3]])
        #pos_features.append(pos_map[pos_tags[i-3]])
        pre = sent[i-3][:2]
        if pre in prefix_map:
          prefix_features.append(prefix_map[pre])
        else :
          prefix_features.append(prefix_map['-UNK-'])
        suf = sent[i-3][-2:]
        if suf in suffix_map:
          suffix_features.append(suffix_map[suf])
        else:
          suffix_features.append(suffix_map['-UNK-'])
        if sent[i-3].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-2 < 0:
        word_features.append(0)
        cap_features.append(0)
        prefix_features.append(0)
        suffix_features.append(0)
        #pos_features.append(0)
      else:
        word_features.append(word_map[sent[i-2]])
        #pos_features.append(pos_map[pos_tags[i-2]])
        pre = sent[i-2][:2]
        if pre in prefix_map:
          prefix_features.append(prefix_map[pre])
        else :
          prefix_features.append(prefix_map['-UNK-'])
        suf = sent[i-2][-2:]
        if suf in suffix_map:
          suffix_features.append(suffix_map[suf])
        else:
          suffix_features.append(suffix_map['-UNK-'])
        if sent[i-2].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i-1 < 0:
        word_features.append(0)
        cap_features.append(0)
        prefix_features.append(0)
        suffix_features.append(0)
        #pos_features.append(0)
      else:
        word_features.append(word_map[sent[i-1]])
        #pos_features.append(pos_map[pos_tags[i-1]])
        pre = sent[i-1][:2]
        if pre in prefix_map:
          prefix_features.append(prefix_map[pre])
        else :
          prefix_features.append(prefix_map['-UNK-'])
        suf = sent[i-1][-2:]
        if suf in suffix_map:
          suffix_features.append(suffix_map[suf])
        else:
          suffix_features.append(suffix_map['-UNK-'])
        if sent[i-1].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      word_features.append(word_map[sent[i]])
      #pos_features.append(pos_map[pos_tags[i]])
      pre = sent[i][:2]
      if pre in prefix_map:
        prefix_features.append(prefix_map[pre])
      else :
        prefix_features.append(prefix_map['-UNK-'])
      suf = sent[i][-2:]
      if suf in suffix_map:
        suffix_features.append(suffix_map[suf])
      else:
        suffix_features.append(suffix_map['-UNK-'])
      if sent[i].isupper():
        cap_features.append(1)
      else:
        cap_features.append(2)

      if utils._all_digits(sent[i]):
        other_features.append(2)
      elif utils._contains_digits(sent[i]):
        other_features.append(1)
      else:
        other_features.append(0)

      if not utils._contains_hyphen(sent[i]):
        other_features.append(0)
      else:
        other_features.append(1)

      sent_len = len(sent)

      if i+1 >= sent_len:
        word_features.append(0)
        cap_features.append(0)
        prefix_features.append(0)
        suffix_features.append(0)
        #pos_features.append(0)
      else:
        word_features.append(word_map[sent[i+1]])
        #pos_features.append(pos_map[pos_tags[i+1]])
        pre = sent[i+1][:2]
        if pre in prefix_map:
          prefix_features.append(prefix_map[pre])
        else :
          prefix_features.append(prefix_map['-UNK-'])
        suf = sent[i+1][-2:]
        if suf in suffix_map:
          suffix_features.append(suffix_map[suf])
        else:
          suffix_features.append(suffix_map['-UNK-'])
        if sent[i+1].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i+2 >= sent_len:
        word_features.append(0)
        cap_features.append(0)
        prefix_features.append(0)
        suffix_features.append(0)
        #pos_features.append(0)
      else:
        word_features.append(word_map[sent[i+2]])
        #pos_features.append(pos_map[pos_tags[i+2]])
        pre = sent[i+2][:2]
        if pre in prefix_map:
          prefix_features.append(prefix_map[pre])
        else :
          prefix_features.append(prefix_map['-UNK-'])
        suf = sent[i+2][-2:]
        if suf in suffix_map:
          suffix_features.append(suffix_map[suf])
        else:
          suffix_features.append(suffix_map['-UNK-'])
        if sent[i+2].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      if i+3 >= sent_len:
        word_features.append(0)
        cap_features.append(0)
        prefix_features.append(0)
        suffix_features.append(0)
        #pos_features.append(0)
      else:
        word_features.append(word_map[sent[i+3]])
        #pos_features.append(pos_map[pos_tags[i+3]])
        pre = sent[i+3][:2]
        if pre in prefix_map:
          prefix_features.append(prefix_map[pre])
        else :
          prefix_features.append(prefix_map['-UNK-'])
        suf = sent[i+3][-2:]
        if suf in suffix_map:
          suffix_features.append(suffix_map[suf])
        else:
          suffix_features.append(suffix_map['-UNK-'])
        if sent[i-4].isupper():
          cap_features.append(1)
        else:
          cap_features.append(2)

      sent_word_features.append(word_features)
      sent_cap_features.append(cap_features)
      sent_prefix_features.append(prefix_features)
      sent_suffix_features.append(suffix_features)
      sent_other_features.append(other_features)
    return sent_word_features, sent_cap_features, sent_prefix_features, sent_suffix_features, sent_other_features
