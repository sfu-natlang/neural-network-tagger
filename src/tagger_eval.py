
"""A program to annotate and evaluate a tensorflow neural net tagger."""


import os
import os.path
import time
import tempfile
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from graph_builder import GreedyTagger
from train_example import TrainExample

import pickle
from data.data_pool import DataPool

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', 'models/model', 'Path to model parameters.')
flags.DEFINE_string('input', 'stdin',
                    'Name of the context input to read data from.')
flags.DEFINE_string('output', 'stdout',
                    'Name of the context input to write data to.')
flags.DEFINE_string('hidden_layer_sizes', '128',
                    'Comma separated list of hidden layer sizes.')
flags.DEFINE_integer('batch_size', 32,
                     'Number of sentences to process in parallel.')
flags.DEFINE_integer('max_steps', 1000, 'Max number of steps to take.')
flags.DEFINE_bool('slim_model', False,
                  'Whether to expect only averaged variables.')

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

def loadBatch(batch_size, index, tokens, epochs):
  features = []
  tags = []
  token_size = len(tokens)
  word_features = []
  other_features = []
  prefix_features = []
  suffix_features = []
  prefix_features3 = []
  suffix_features3 = []
  words = []
  for i in range(batch_size):
    if index == token_size:
      index = 0
      epochs += 1
    token = tokens[index]
    words.append(token.get_word())
    word_features.extend(token.get_features()[0])
    other_features.extend(token.get_features()[1])
    prefix_features.extend(token.get_features()[2])
    suffix_features.extend(token.get_features()[3])
    prefix_features3.extend(token.get_features()[4])
    suffix_features3.extend(token.get_features()[5])
    tags.append(token.get_label())
    index += 1
  features.append(','.join(str(e) for e in word_features))
  features.append(','.join(str(e) for e in other_features))      
  features.append(','.join(str(e) for e in prefix_features))
  features.append(','.join(str(e) for e in suffix_features))
  #features.append(','.join(str(e) for e in prefix_features3))
  #features.append(','.join(str(e) for e in suffix_features3))  
  return index, epochs, features, tags, words

def get_index(word, wordMap):
  if word in wordMap:
    return wordMap.index(word)
  else:
    return 0

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

def gen_prefix_feature3(sent, i, pMap, p_features):
  if i-4 < 0:
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i-4][:3], pMap))

  if i-3 < 0:
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i-3][:3], pMap))

  if i-2 < 0:
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i-2][:3], pMap))  

  if i-1 < 0:
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i-1][:3], pMap))

  if i+1 >= len(sent):
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i+1][:3], pMap))

  if i+2 >= len(sent):
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i+2][:3], pMap))

  if i+3 >= len(sent):
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i+3][:3], pMap))

  p_features.append(get_index(sent[i][:3], pMap))

def gen_suffix_feature3(sent, i, sMap, s_features):
  if i-4 < 0:
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i-4][-3:], sMap))
  if i-3 < 0:
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i-3][-3:], sMap))
  if i-2 < 0:
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i-2][-3:], sMap))
  if i-1 < 0:
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i-1][-3:], sMap))

  if i+1 >= len(sent): 
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i+1][-3:], sMap))

  if i+2 >= len(sent): 
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i+2][-3:], sMap))

  if i+3 >= len(sent): 
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i+3][-3:], sMap))

  s_features.append(get_index(sent[i][-3:], sMap))

def gen_prefix_feature2(sent, i, pMap, p_features):
  if i-4 < 0:
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i-4][:2], pMap))

  if i-3 < 0:
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i-3][:2], pMap))

  if i-2 < 0:
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i-2][:2], pMap))  

  if i-1 < 0:
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i-1][:2], pMap))

  if i+1 >= len(sent):
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i+1][:2], pMap))

  if i+2 >= len(sent):
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i+2][:2], pMap))

  if i+3 >= len(sent):
    p_features.append(1)
  else:
    p_features.append(get_index(sent[i+3][:2], pMap))

  p_features.append(get_index(sent[i][:2], pMap))

def gen_suffix_feature2(sent, i, sMap, s_features):
  if i-4 < 0:
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i-4][-2:], sMap))
  if i-3 < 0:
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i-3][-2:], sMap))
  if i-2 < 0:
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i-2][-2:], sMap))
  if i-1 < 0:
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i-1][-2:], sMap))

  if i+1 >= len(sent): 
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i+1][-2:], sMap))

  if i+2 >= len(sent): 
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i+2][-2:], sMap))

  if i+3 >= len(sent): 
    s_features.append(1)
  else:
    s_features.append(get_index(sent[i+3][-2:], sMap))

  s_features.append(get_index(sent[i][-2:], sMap))


def gen_other_features(other_features, word):
  if _all_digits(word):
    other_features.append(2)
  elif _contains_digits(word):
    other_features.append(1)
  else:
    other_features.append(0)

  if not _contains_hyphen(word):
    other_features.append(0)
  else:
    other_features.append(1)

def gen_word_features(sent, i, wordMap, word_features):
  if i-4 < 0:
    word_features.append(1)
  else:
    word_features.append(get_index(sent[i-4], wordMap))

  if i-3 < 0:
    word_features.append(1)
  else:
    word_features.append(get_index(sent[i-3], wordMap))

  if i-2 < 0:
    word_features.append(1)
  else:
    word_features.append(get_index(sent[i-2], wordMap))

  if i-1 < 0:
    word_features.append(1)
  else:
    word_features.append(get_index(sent[i-1],wordMap))

  word_features.append(get_index(sent[i], wordMap))

  sent_len = len(sent)
  if i+1 >= sent_len:
    word_features.append(2)
  else:
    word_features.append(get_index(sent[i+1], wordMap))

  if i+2 >= sent_len:
    word_features.append(2)
  else:
    word_features.append(get_index(sent[i+2], wordMap))

  if i+3 >= sent_len:
    word_features.append(2)
  else:
    word_features.append(get_index(sent[i+3], wordMap))

def Eval(sess):
  """Builds and evaluates a network."""
  wordMapPath = "word-map"
  tagMapPath = "tag-map"
  pMapPath = "prefix-list"
  sMapPath = "suffix-list"
  pMapPath3 = "prefix-list3"
  sMapPath3 = "suffix-list3"
  pMap = readAffix(pMapPath)
  sMap = readAffix(sMapPath)
  pMap3 = readAffix(pMapPath3)
  sMap3 = readAffix(sMapPath3)
  wordMap = readMap(wordMapPath)
  tagMap = readMap(tagMapPath)

  wordMap.insert(0,"-start-")
  wordMap.insert(0,"-end-")
  wordMap.insert(0,"-unknown-")

  pMap.insert(0,"-start-")
  pMap.insert(0,"-unknown-")
  sMap.insert(0,"-start-")
  sMap.insert(0,"-unknown-")

  feature_sizes = [8,2,8,8]
  domain_sizes = [39398, 3, len(pMap)+2, len(sMap)+2]
  num_actions = 45
  embedding_dims = [64,8,16,16]

  t = time.time()
  hidden_layer_sizes = map(int, FLAGS.hidden_layer_sizes.split(','))
  logging.info('Building training network with parameters: feature_sizes: %s '
               'domain_sizes: %s', feature_sizes, domain_sizes)
  config = {}
  config['format'] = 'penn2malt'
  config['test'] = 'wsj_23[0-9][0-9].mrg.3.pa.gs.tab'
  config['data_path'] = '/cs/natlang-user/vivian/penn-wsj-deps'
  testDataPool = DataPool(data_format  = config['format'],
                           data_regex   = config['test'],
                           data_path    = config['data_path'])

  test_tokens = []
  while testDataPool.has_next_data():
    sent = testDataPool.get_next_data()
    sent_words = sent.get_word_list()
    sent_tags = sent.get_pos_list()
    assert len(sent_words) == len(sent_tags)
    for idx, word in enumerate(sent_words):
      features = []
      word_features = []
      other_features = []
      prefix_features = []
      suffix_features = []
      prefix_features3 = []
      suffix_features3 = []
      gen_word_features(sent_words, idx, wordMap, word_features)
      gen_other_features(other_features, word)
      gen_prefix_feature2(sent_words, idx, pMap, prefix_features)
      gen_suffix_feature2(sent_words, idx, sMap, suffix_features)
      gen_prefix_feature3(sent_words, idx, pMap3, prefix_features3)
      gen_suffix_feature3(sent_words, idx, sMap3, suffix_features3)
      features.append(word_features)
      features.append(other_features)
      features.append(prefix_features)
      features.append(suffix_features)
      features.append(prefix_features3)
      features.append(suffix_features3)

      label = tagMap.index(sent_tags[idx])
      test_tokens.append(TrainExample(word, features,label)) 

  tagger = GreedyTagger(num_actions, 
                        feature_sizes, 
                        domain_sizes,
                        embedding_dims, 
                        hidden_layer_sizes, 
                        gate_gradients=True)
  tagger.AddEvaluation(FLAGS.batch_size)
  tagger.AddSaver()
  sess.run(tagger.inits.values())
  tagger.saver.restore(sess, FLAGS.model_path)
  t = time.time()
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  num_documents = 0
  index = 0
  epochs = 0
  while True:
    index, epochs, feature_endpoints, gold_tags, words = loadBatch(FLAGS.batch_size, index, test_tokens, epochs)
    tf_eval_metrics = sess.run(tagger.evaluation['logits'], feed_dict={tagger.test_input:feature_endpoints})
    for i in range(FLAGS.batch_size):
      best_action = 0
      best_score = float("-inf")
      if words[i] == "(":
        best_action = tagMap.index("-LRB-")
      elif words[i] == ")":
        best_action = tagMap.index("-RRB-")
      else:
        for j in range(45):
          if tf_eval_metrics[i][j] > best_score:
            best_score = tf_eval_metrics[i][j]
            best_action = j
      if best_action == gold_tags[i]:
        num_correct += 1
      num_tokens += 1
    if num_epochs is None:
      num_epochs = epochs
    elif num_epochs < epochs:
      break
  eval_metric = 0 if num_tokens == 0 else (100.0 * num_correct / num_tokens)
  logging.info('Number of Tokens: %d, Seconds elapsed in evaluation: %.2f, '
               'eval metric: %.2f%%', num_tokens, time.time() - t, eval_metric)
  logging.info('num correct tokens: %d', num_correct)


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  with tf.Session() as sess:
    Eval(sess)


if __name__ == '__main__':
  tf.app.run()