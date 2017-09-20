import copy
import collections
import operator
import os
import pickle
import numpy as np

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
PAD = "$PAD$"
NONE = "O"

# special error message
class MyIOError(Exception):
  def __init__(self, filename):
  # custom error message
    message = """
    ERROR: Unable to locate file {}.
    FIX: Have you tried running python build_data.py first?
    This will build vocab file from your train, test and dev sets and
    trimm your word vectors.
    """.format(filename)
    super(MyIOError, self).__init__(message)

def gen_word_features(words, processing_word, use_char=False):
  if not use_char:
    word_ids = words
    pad = 0
    sent_word_features = []
    sent_char_features = []
    for i in range(len(words)):
      word_features = []
      char_features = []
      if i-4 < 0:
        word_features.append(pad)
      else:
        word_features.append(word_ids[i-4])
     
      if i-3 < 0:
        word_features.append(pad)
        char_features.append([pad])
      else:
        word_features.append(word_ids[i-3])
      if i-2 < 0:
        word_features.append(pad)
      else:
        word_features.append(word_ids[i-2])

      if i-1 < 0:
        word_features.append(pad)
      else:
        word_features.append(word_ids[i-1])

      word_features.append(word_ids[i])

      sent_len = len(words)

      if i+1 >= sent_len:
        word_features.append(pad)
      else:
        word_features.append(word_ids[i+1])
      
      if i+2 >= sent_len:
        word_features.append(pad)
      else:
        word_features.append(word_ids[i+2])

      if i+3 >= sent_len:
        word_features.append(pad)
      else:
        word_features.append(word_ids[i+3])
      sent_word_features.append(word_features)
    return sent_word_features, sent_char_features
  word_ids = [w[1] for w in words]
  char_ids = [w[0] for w in words]
  pad = 0
  sent_word_features = []
  sent_char_features = []
  for i in range(len(words)):
    word_features = []
    char_features = []
    if i-4 < 0:
      word_features.append(pad)
      char_features.append([pad])
    else:
      word_features.append(word_ids[i-4])
      char_features.append(char_ids[i-4])
   
    if i-3 < 0:
      word_features.append(pad)
      char_features.append([pad])
    else:
      word_features.append(word_ids[i-3])
      char_features.append(char_ids[i-3])
    if i-2 < 0:
      word_features.append(pad)
      char_features.append([pad])
    else:
      word_features.append(word_ids[i-2])
      char_features.append(char_ids[i-2])

    if i-1 < 0:
      word_features.append(pad)
      char_features.append([pad])
    else:
      word_features.append(word_ids[i-1])
      char_features.append(char_ids[i-1])

    word_features.append(word_ids[i])
    char_features.append(char_ids[i])

    sent_len = len(words)

    if i+1 >= sent_len:
      word_features.append(pad)
      char_features.append([pad])
    else:
      word_features.append(word_ids[i+1])
      char_features.append(char_ids[i+1])
    
    if i+2 >= sent_len:
      word_features.append(pad)
      char_features.append([pad])
    else:
      word_features.append(word_ids[i+2])
      char_features.append(char_ids[i+2])

    if i+3 >= sent_len:
      word_features.append(pad)
      char_features.append([pad])
    else:
      word_features.append(word_ids[i+3])
      char_features.append(char_ids[i+3])
    sent_word_features.append(word_features)
    sent_char_features.extend(char_features)
  return sent_word_features, sent_char_features

def find_labels(iob, mentions, vocab_tags, id2type):
  labels = []
  mid = -1
  for idx, lab in enumerate(iob):
    if lab == 0:
      tag = 'O'
      labels.append(vocab_tags[tag])
    elif lab == 1:
      mid += 1
      tag = 'B-'+id2type[mentions[mid]]
      labels.append(vocab_tags[tag])
    else:
      # TODO: fix the combining step here
      tag = 'I-'+id2type[mentions[mid]]
      labels.append(vocab_tags[tag])
  return labels

def find_mentions(tags):
  mentions = []
  entity = []
  for idx, t in enumerate(tags):
    if t == 'B' or t == 1:
      if len(entity) > 0:
        mentions.append(entity)
        entity = []
      entity.append(idx)
    elif t == 'I' or t == 2:
      entity.append(idx)
    elif len(entity) > 0:
      mentions.append(entity)
      entity = []
  if len(entity) > 0:
    mentions.append(entity)
  return mentions

def get_vocabs(datasets):
  """
  Args:
    datasets: a list of dataset objects
  Return:
    a set of all the words in the dataset
  """
  print("Building vocab...")
  vocab_words = set()
  vocab_tags = set()
  for dataset in datasets:
    for words, tags, _, _, _, _, _ in dataset:
      vocab_words.update(words)
      vocab_tags.update(tags)
  print("- done. {} tokens".format(len(vocab_words)))
  return vocab_words, vocab_tags


def get_char_vocab(dataset):
  """
  Args:
    dataset: a iterator yielding tuples (sentence, tags)
  Returns:
    a set of all the characters in the dataset
  """
  vocab_char = set()
  for words, _, _, _, _, _, _ in dataset:
    for word in words:
      vocab_char.update(word)

  return vocab_char


def get_glove_vocab(filename):
  """
  Args:
    filename: path to the glove vectors
  """
  print("Building vocab...")
  count = 0
  vocab = set()
  with open(filename) as f:
    for line in f:
      word = line.strip().split(' ')[0]
      vocab.add(word)
      count += 1
  print("- done. {} tokens!".format(count))
  return vocab


def write_vocab(vocab, filename):
  """
  Writes a vocab to a file

  Args:
    vocab: iterable that yields word
    filename: path to vocab file
  Returns:
    write a word per line
  """
  print("Writing vocab...")
  with open(filename, "w") as f:
    for i, word in enumerate(vocab):
      if i != len(vocab) - 1:
        f.write("{}\n".format(word))
      else:
        f.write(word)
  print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
  """
  Args:
    filename: file with a word per line
  Returns:
    d: dict[word] = index
  """
  try:
    d = dict()
    with open(filename) as f:
      for idx, word in enumerate(f):
        word = word.strip()
        d[word] = idx


  except IOError:
    raise MyIOError(filename)
  return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
  """
  Saves glove vectors in numpy array

  Args:
    vocab: dictionary vocab[word] = index
    glove_filename: a path to a glove file
    trimmed_filename: a path where to store a matrix in npy
    dim: (int) dimension of embeddings
  """
  embeddings = np.zeros([len(vocab), dim])
  with open(glove_filename) as f:
    for line in f:
      line = line.strip().split(' ')
      word = line[0]
      embedding = [float(x) for x in line[1:]]
      if word in vocab:
        word_idx = vocab[word]
        embeddings[word_idx] = np.asarray(embedding)

  np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
  """
  Args:
    filename: path to the npz file
  Returns:
    matrix of embeddings (np array)
  """
  try:
    with np.load(filename) as data:
      return data["embeddings"]

  except IOError:
    raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,
          lowercase=False, chars=False):
  """
  Args:
    vocab: dict[word] = idx
  Returns:
    f("cat") = ([12, 4, 32], 12345)
         = (list of char ids, word id)
  """
  def f(word):
    # 0. get chars of words
    if vocab_chars is not None and chars == True:
      char_ids = []
      for char in word:
        # ignore chars out of vocabulary
        if char in vocab_chars:
          char_ids += [vocab_chars[char]]

    # 1. preprocess word
    if lowercase:
      word = word.lower()
    if word.isdigit():
      word = NUM

    # 2. get id of word
    if vocab_words is not None:
      if word in vocab_words:
        word = vocab_words[word]
      else:
        word = vocab_words[UNK]

    # 3. return tuple char ids, word id
    if vocab_chars is not None and chars == True:
      return char_ids, word
    else:
      return word

  return f


def _pad_sequences(sequences, pad_tok, max_length):
  """
  Args:
    sequences: a generator of list or tuple
    pad_tok: the char to pad with
  Returns:
    a list of list where each sublist has same length
  """
  sequence_padded, sequence_length = [], []

  for seq in sequences:
    seq = list(seq)
    seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
    sequence_padded +=  [seq_]
    sequence_length += [min(len(seq), max_length)]

  return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
  """
  Args:
    sequences: a generator of list or tuple
    pad_tok: the char to pad with
  Returns:
    a list of list where each sublist has same length
  """
  if nlevels == 1:
    max_length = max(map(lambda x : len(x), sequences))
    sequence_padded, sequence_length = _pad_sequences(sequences,
                      pad_tok, max_length)

  elif nlevels == 2:
    length_word = [max(map(lambda x: len(x), seq)) for seq in sequences if len(seq)>0]
    if len(length_word) == 0:
      max_length_word = 0
    else:
      max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences if len(seq)>0])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
      # all words are same length now
      sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
      sequence_padded += [sp]
      sequence_length += [sl]

    max_length_sentence = max(map(lambda x : len(x), sequences))
    sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word,
                      max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
  
  return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
  """
  Args:
    data: generator of (sentence, tags) tuples
    minibatch_size: (int)
  Returns:
    list of tuples
  """
  x_batch, y_batch, iob_batch, type_batch, mention_batch, wfeat_batch, cfeat_batch = [], [], [], [], [], [], []
  for (x, y, a, b, c, d, e) in data:
    if len(x_batch) == minibatch_size:
      yield x_batch, y_batch, iob_batch, type_batch, mention_batch, wfeat_batch, cfeat_batch
      x_batch, y_batch, iob_batch, type_batch, mention_batch, wfeat_batch, cfeat_batch = [], [], [], [], [], [], []

    if type(x[0]) == tuple:
      x = zip(*x)
    x_batch += [x]
    y_batch += [y]
    iob_batch += [a]
    type_batch += [b]
    mention_batch += [c]
    wfeat_batch += [d]
    cfeat_batch += [e]

  if len(x_batch) != 0:
    left = minibatch_size - len(x_batch)
    for i in range(left):
      if type(x[0]) == tuple:
        x = zip(*x)
      x_batch += [x_batch[0]]
      y_batch += [y_batch[0]]
      iob_batch += [iob_batch[0]]
      type_batch += [type_batch[0]]
      mention_batch += [mention_batch[0]]
      wfeat_batch += [wfeat_batch[0]]
      cfeat_batch += [cfeat_batch[0]]
    yield x_batch, y_batch, iob_batch, type_batch, mention_batch, wfeat_batch, cfeat_batch


def get_chunk_type(tok, idx_to_tag):
  """
  Args:
    tok: id of token, ex 4
    idx_to_tag: dictionary {4: "B-PER", ...}
  Returns:
    tuple: "B", "PER"
  """
  tag_name = idx_to_tag[tok]
  tag_class = tag_name.split('-')[0]
  tag_type = tag_name.split('-')[-1]
  return tag_class, tag_type


def get_chunks(seq, tags):
  """
  Args:
    seq: [4, 4, 0, 0, ...] sequence of labels
    tags: dict["O"] = 4
  Returns:
    list of (chunk_type, chunk_start, chunk_end)

  Example:
    seq = [4, 5, 0, 3]
    tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
    result = [("PER", 0, 2), ("LOC", 3, 4)]
  """
  default = tags[NONE]
  idx_to_tag = {idx: tag for tag, idx in tags.items()}
  chunks = []
  chunk_type, chunk_start = None, None
  for i, tok in enumerate(seq):
    # End of a chunk 1
    if tok == default and chunk_type is not None:
      # Add a chunk.
      chunk = (chunk_type, chunk_start, i)
      chunks.append(chunk)
      chunk_type, chunk_start = None, None

    # End of a chunk + start of a chunk!
    elif tok != default:
      tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
      if chunk_type is None:
        chunk_type, chunk_start = tok_chunk_type, i
      elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
        chunk = (chunk_type, chunk_start, i)
        chunks.append(chunk)
        chunk_type, chunk_start = tok_chunk_type, i
    else:
      pass
  # end condition
  if chunk_type is not None:
    chunk = (chunk_type, chunk_start, len(seq))
    chunks.append(chunk)

  return chunks

def read_pickle_file(path):
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  return ret


def order_dictionary(dictionary, mode, reverse=False):
  '''
  Order a dictionary by 'key' or 'value'.
  mode should be either 'key' or 'value'
  http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
  '''

  if mode =='key':
    return collections.OrderedDict(sorted(dictionary.items(),
                      key=operator.itemgetter(0),
                      reverse=reverse))
  elif mode =='value':
    return collections.OrderedDict(sorted(dictionary.items(),
                      key=operator.itemgetter(1),
                      reverse=reverse))
  elif mode =='key_value':
    return collections.OrderedDict(sorted(dictionary.items(),
                      reverse=reverse))
  elif mode =='value_key':
    return collections.OrderedDict(sorted(dictionary.items(),
                      key=lambda x: (x[1], x[0]),
                      reverse=reverse))
  else:
    raise ValueError("Unknown mode. Should be 'key' or 'value'")

def reverse_dictionary(dictionary):

  if type(dictionary) is collections.OrderedDict:
    return collections.OrderedDict([(v, k) for k, v in dictionary.items()])
  else:
    return {v: k for k, v in dictionary.items()}

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

