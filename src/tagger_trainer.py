import os
import os.path
import time
import tensorflow as tf
import string
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from graph_builder import GreedyTagger
from train_example import TrainExample

import pickle
from data.data_pool import DataPool

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tf_master', '', 'TensorFlow execution engine to connect to.')
flags.DEFINE_string('output_path', '', 'Top level for output.')
flags.DEFINE_string('params', '0', 'Unique identifier of parameter grid point.')
flags.DEFINE_string('training_corpus', 'training-corpus', 'Name of the context input to read training data from.')
flags.DEFINE_string('tuning_corpus', 'tuning-corpus', 'Name of the context input to read tuning data from.')
flags.DEFINE_string('word_embeddings', None, 
                    'Recordio containing pretrained word embeddings, will be '
'loaded as the first embedding matrix.')
flags.DEFINE_string('hidden_layer_sizes', '200,200', 'Comma separated list of hidden layer sizes.')
flags.DEFINE_integer('batch_size', 32, 'Number of sentences to process in parallel.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train for.')
flags.DEFINE_integer('max_steps', 50, 'Max number of parser steps during a training step.')
flags.DEFINE_integer('report_every', 100, 'Report cost and training accuracy every this many steps.')
flags.DEFINE_integer('checkpoint_every', 5000, 'Measure tuning UAS and checkpoint every this many steps.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate parameter.')
flags.DEFINE_integer('decay_steps', 4000,
                     'Decay learning rate by 0.96 every this many steps.')
flags.DEFINE_float('momentum', 0.9,
                   'Momentum parameter for momentum optimizer.')
flags.DEFINE_string('seed', '0', 'Initialization seed for TF variables.')
flags.DEFINE_string('pretrained_params', None,
                    'Path to model from which to load params.')
flags.DEFINE_string('pretrained_params_names', None,
                    'List of names of tensors to load from pretrained model.')
flags.DEFINE_float('averaging_decay', 0.9999,
                   'Decay for exponential moving average when computing'
'averaged parameters, set to 1 to do vanilla averaging.')


def Eval(sess, tagger, test_data, num_steps, best_eval_metric):
  """Evaluates a network and checkpoints it to disk.

  Args:
    sess: tensorflow session to use
    parser: graph builder containing all ops references
    num_steps: number of training steps taken, for logging
    best_eval_metric: current best eval metric, to decide whether this model is
        the best so far

  Returns:
    new best eval metric
  """
  logging.info('Evaluating training network.')
  t = time.time()
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  index = 0
  epochs = 0
  while True:
    index, epochs, feature_endpoints, gold_tags = loadBatch(32, index, test_data, epochs)
    tf_eval_metrics = sess.run(tagger.evaluation['logits'], feed_dict={tagger.test_input:feature_endpoints})
    for i in range(32):
      best_action = 0
      best_score = float("-inf")
      for j in range(len(tf_eval_metrics)):
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
  return max(eval_metric, best_eval_metric)

def Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, wordMap, tagMap):
  """Builds and trains the network
  Args:
    sess: tensorflow session to use
    num_actions: number of possible golden actions
    feature_sizes: size of each feature vector.
    domain_sizes: number of possible feature ids in each feature vector.
    embedding_dims: embedding dimension to use for each feature group.
  """
  t = time.time()
  hidden_layer_sizes = map(int, FLAGS.hidden_layer_sizes.split(','))
  logging.info('Building training network with parameters: feature_sizes: %s '
  'domain_sizes: %s', str(feature_sizes), str(domain_sizes))
  config = {}
  config['format'] = 'penn2malt'
  config['train'] = 'wsj_0[2-9][0-9][0-9].mrg.3.pa.gs.tab|wsj_1[0-9][0-9][0-9].mrg.3.pa.gs.tab|wsj_2[0-1][0-9][0-9].mrg.3.pa.gs.tab'
  config['test'] = 'wsj_23[0-9][0-9].mrg.3.pa.gs.tab'
  config['data_path'] = '/cs/natlang-user/vivian/penn-wsj-deps'

  trainDataPool = DataPool(data_format  = config['format'],
                           data_regex   = config['train'],
                           data_path    = config['data_path'])
  testDataPool = DataPool(data_format  = config['format'],
                           data_regex   = config['test'],
                           data_path    = config['data_path'])

  tagger = GreedyTagger(num_actions, 
                        feature_sizes, 
                        domain_sizes,
                        embedding_dims, 
                        hidden_layer_sizes, 
                        seed=int(FLAGS.seed),
                        gate_gradients=True,
                        average_decay=FLAGS.averaging_decay)
  tagger.AddTraining(32)
  tagger.AddEvaluation(32)
  logging.info('Initializing...')
  num_epochs = 0
  cost_sum = 0.0
  num_steps = 0
  best_eval_metric = 0.0
  sess.run(tagger.inits.values())
  logging.info('Loading the training data...')
  '''
  load the training examples
  '''
  tokens = []
  while trainDataPool.has_next_data():
    sent = trainDataPool.get_next_data()
    sent_words = sent.get_word_list()
    sent_tags = sent.get_pos_list()
    assert len(sent_words) == len(sent_tags)
    for idx, word in enumerate(sent_words):
      features = []
      word_features = []
      other_features = []
      gen_word_features(sent_words, idx, wordMap, word_features)
      gen_other_features(other_features, word)
      features.append(word_features)
      features.append(other_features)
      label = tagMap.index(sent_tags[idx])
      tokens.append(TrainExample(features,label)) 

  logging.info('Loading the test data...')
  '''
  load the training examples
  '''
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
      gen_word_features(sent_words, idx, wordMap, word_features)
      gen_other_features(other_features, word)
      features.append(word_features)
      features.append(other_features)
      label = tagMap.index(sent_tags[idx])
      test_tokens.append(TrainExample(features,label)) 

  index = 0 
  logging.info('Trainning...')
  while num_epochs < 10:
    index, num_epochs, feature_endpoints, gold_tags = loadBatch(32, index, tokens, num_epochs)
    tf_cost, _ = sess.run([tagger.training['cost'],tagger.training['train_op']], feed_dict={tagger.input:feature_endpoints, tagger.labels:gold_tags})
    cost_sum += tf_cost
    num_steps += 1
    if num_steps % FLAGS.report_every == 0:
      logging.info('Epochs: %d, num steps: %d, '
                   'seconds elapsed: %.2f, avg cost: %.2f, ', num_epochs,
                    num_steps, time.time() - t, cost_sum / FLAGS.report_every)
      cost_sum = 0.0
    if num_steps % 5000 == 0:
      best_eval_metric = Eval(sess, tagger, test_tokens, num_steps, best_eval_metric)

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
  for i in range(batch_size):
    if index == token_size:
      index = 0
      epochs += 1
    token = tokens[index]
    word_features.extend(token.get_features()[0])
    other_features.extend(token.get_features()[1])
    tags.append(token.get_label())
    i += 1
    index += 1
  features.append(','.join(str(e) for e in word_features))
  features.append(','.join(str(e) for e in other_features))      

  return index, epochs, features, tags

def get_index(word, wordMap):
  if word in wordMap:
    return wordMap.index(word)
  else:
    return 0

def _contains_digits(s):
  return any(char.isdigit() for char in s)

def _contains_hyphen(s):
  return any(char == "-" for char in s)

def _contains_upper(s):
  return any(char.isupper() for char in s)

def _contains_punc(s):
  return any(char in string.punctuation for char in s)

def gen_other_features(other_features, word):
  if not _contains_digits(word):
    other_features.append(0)
  else:
    other_features.append(1)
  if not _contains_hyphen(word):
    other_features.append(2)
  else:
    other_features.append(3)
  if not _contains_upper(word):
    other_features.append(4)
  else:
    other_features.append(5)
  if not _contains_punc(word):
    other_features.append(6)
  else:
    other_features.append(7)

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
    
def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  wordMapPath = "word-map"
  tagMapPath = "tag-map"
  wordMap = readMap(wordMapPath)
  tagMap = readMap(tagMapPath)
  wordMap.insert(0,"-start-")
  wordMap.insert(0,"-end-")
  wordMap.insert(0,"-unknown-")
  feature_sizes = [8,4]
  domain_sizes = [39398, 8]
  num_actions = 45
  embedding_dims = [64,8]
  logging.info('Preparing Lexicon...')
  logging.info(len(wordMap))
  logging.info(len(tagMap))    
  with tf.Session(FLAGS.tf_master) as sess:
    Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, wordMap, tagMap)

if __name__ == '__main__':
  tf.app.run() 

