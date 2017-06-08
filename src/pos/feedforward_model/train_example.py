class TrainExample(object):
  def __init__(self, word, features, label):
    self._features = features
    self._label = label
    self._word = word

  def get_features(self):
    return self._features

  def get_label(self):
    return self._label

  def get_word(self):
    return self._word
