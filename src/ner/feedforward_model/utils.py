import copy
import collections
import operator
import os
import re
import pickle

def read_pickle_file(path):
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  return ret

def replace_digits(s):
  return re.sub('\d', '0', s)

def pad_lists(lists, padding=0):
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