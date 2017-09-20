import copy
import collections
import operator
import os
import pickle

def read_pickle_file(path):
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  return ret

def pad_lists(lists, padding=0):
  maxlen = max(map(len, lists))
  padded = []

  for l in lists:
    padded.append(l + [padding]*(maxlen - len(l)))

  return padded


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