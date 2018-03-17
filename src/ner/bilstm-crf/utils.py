import copy
import collections
import operator
import os
import re
import pickle
import codecs
import numpy as np

def get_chunk_type(tok, idx_to_tag):
  """
  Args:
      tok: id of token, ex 4
      idx_to_tag: dictionary {4: "B-PER", ...}
  Returns:
      tuple: "B", "PER"
  """
  if idx_to_tag is not None:
    tok = idx_to_tag[tok]
  tag_class = tok.split('-')[0]
  tag_type = tok.split('-')[-1]
  return tag_class, tag_type

def get_chunks(seq, idx_to_tag=None):

  default = 0
  
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

def load_pretrained_token_embeddings(file_input):
    file_input = codecs.open(file_input, 'r', 'UTF-8')
    count = -1
    token_to_vector = {}
    for cur_line in file_input:
        count += 1
        cur_line = cur_line.strip()
        cur_line = cur_line.split(' ')
        if len(cur_line)==0:continue
        token = cur_line[0]
        vector = np.array([float(x) for x in cur_line[1:]])
        token_to_vector[token] = vector
    file_input.close()
    return token_to_vector

def create_boundry_list(iob_list):
  boundry_list = []
  start = 0
  length = 0
  for i, tag in enumerate(iob_list):
    if tag == 0:
      if i > 0 and iob_list[i-1] != 0:
        boundry_list.append([start, length])
        length = 0
    elif tag == 1:
      if i > 0 and iob_list[i-1] != 0:
        boundry_list.append([start, length])
        length = 0
      start = i
      length = 1
    else:
      length += 1
  return boundry_list


def create_mention_list(boundry_list):
  mention_list = []
  for boundry in boundry_list:
    mention = []
    start = boundry[0]
    for i in range(boundry[1]):
      mention.append(start+i)
    mention_list.append(mention)
  return mention_list

def iob_type(ner_tags):
  iob_list = []
  type_list = []
  boundry_list = []
  boundry_start = 0
  boundry = 0
  prev_type = ""
  tag_map = {"LOC":0, "PER":1, "ORG":2, "MISC":3}
  iob_map = {"O":0, "B":1, "I":2}
  for i, tag in enumerate(ner_tags):
    if tag == 'O':
      iob_list.append(0)
      if i > 0 and ner_tags[i-1] != 'O':
        boundry_list.append([boundry_start, boundry])
        type_list.append(tag_map[prev_type])
        boundry = 0
      continue

    split = tag.split('-')
    if split[0] == 'B':
      if i > 0 and ner_tags[i-1] != 'O':
        boundry_list.append([boundry_start, boundry])
        type_list.append(tag_map[prev_type])
      iob_list.append(1)
      boundry = 1
      boundry_start = i
      prev_type = tag[2:]
    else:
      iob_list.append(2)
      boundry += 1
  if boundry > 0:
    type_list.append(tag_map[prev_type])
    boundry_list.append([boundry_start, boundry])
      
  return iob_list, type_list, boundry_list

def iob(ner_tags):
  for i, tag in enumerate(ner_tags):
    if tag == 'O':
      continue
    split = tag.split('-')
    if len(split) != 2 or split[0] not in ['I', 'B']:
      return False
    if split[0] == 'B':
      continue
    elif i == 0 or ner_tags[i - 1] == 'O':  # conversion to IOB
      ner_tags[i] = 'B' + tag[1:]
    elif ner_tags[i - 1][1:] == tag[1:]:
      continue
    else:  # conversion to IOB
      ner_tags[i] = 'B' + tag[1:]
  return True

def iob2(ner_tags, ner_map):
  for i, tag_index in enumerate(ner_tags):
    tag = ner_map[tag_index]
    if tag == 'O':
      continue
    split = tag.split('-')
    if len(split) != 2 or split[0] not in ['I', 'B']:
      return False
    if split[0] == 'I':
      continue
    elif i == 0 or ner_tags[i - 1] == 'O':  # conversion to IOB
      ner_tags[i] = 'I' + tag[1:]
  return True


def read_pickle_file(path):
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  return ret

def replace_digits(s):
  return re.sub('\d', '0', s)

def pad_lists(lists, padding=0):
  if len(lists)==0:
    return [[]]
  maxlen = max(map(len, lists))
  padded = []

  for l in lists:
    padded.append(l + [padding]*(maxlen - len(l)))

  return padded

def get_index(word, wordMap):
  if word in wordMap:
    return wordMap.index(word)
  else:
    return len(wordMap)-1

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

def remove_bio_from_label_name(label_name):
  if label_name[:2] in ['B-', 'I-', 'E-', 'S-']:
    new_label_name = label_name[2:]
  else:
    new_label_name = label_name
  return new_label_name

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

if __name__ == '__main__':

  ex = [0, 0, 1, 2, 2, 0, 0, 1, 0, 1, 2, 0]
  print create_boundry_list(ex)
