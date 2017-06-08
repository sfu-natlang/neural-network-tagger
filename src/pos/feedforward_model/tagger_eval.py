
"""A program to annotate and evaluate a tensorflow neural net tagger."""


import os
import os.path
import time
import tempfile
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from graph_builder import GreedyTagger
import codecs
import pickle
from data_format import ConllData
from apply_bpe import BPE

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

def OutputPath(path):
  return os.path.join(FLAGS.output_path, path)

def get_index(word, wordMap):
  if word in wordMap:
    return wordMap.index(word)
  else:
    return 0

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

def get_current_features(sent_batch, num_epochs, data, wordMap, tagMap, pMap, sMap):
  batch_size = len(sent_batch)
  new_sent_batch = [sent for sent in sent_batch if sent.has_next_state()]
  for sent in sent_batch:
    if not sent.has_next_state():
      sent.reset_state()

  while len(new_sent_batch) < batch_size:
    sent, num_epochs = advance_sent(num_epochs, data)
    new_sent_batch.append(sent)

  features = []
  tags = []
  words = []
  cap_features = []
  word_features = []
  other_features = []
  prefix_features = []
  suffix_features = []
  tag_features = []

  for sent in new_sent_batch:
    word_list = sent.get_word_list()
    tag_list = sent.get_tag_list()
    state = sent.get_next_state()
    word = word_list[state]
    tag = get_index(tag_list[state], tagMap)
    words.append(word)
    tags.append(tag)
    cap_features.extend(sent.cap_features[state])
    word_features.extend(sent.word_features[state])
    other_features.extend(sent.other_features[state])
    prefix_features.extend(sent.prefix_features[state])
    suffix_features.extend(sent.suffix_features[state])
    tag_features.extend(sent.gen_tag_features(state))
  features.append(','.join(str(e) for e in cap_features))
  features.append(','.join(str(e) for e in word_features))
  features.append(','.join(str(e) for e in other_features))      
  features.append(','.join(str(e) for e in prefix_features))
  features.append(','.join(str(e) for e in suffix_features))
  features.append(','.join(str(e) for e in tag_features))
  return new_sent_batch, num_epochs, features, tags, words

def advance_sent(num_epochs, data):
  if not data.has_next_sent():
    data.reset_index()
    num_epochs += 1
  sent = data.get_next_sent()
  sent.set_epoch(num_epochs)
  return sent, num_epochs

def loadBatch(batch_size, num_epochs, data):
  size = data.get_sent_num()
  sent_batch = []
  for i in range(batch_size):
    if not data.has_next_sent():
      data.reset_index()
      num_epochs += 1
    sent = data.get_next_sent()
    sent.set_epoch(num_epochs)
    sent_batch.append(sent)
    
  return num_epochs, sent_batch

def Eval(sess):
  """Builds and evaluates a network."""
  logging.set_verbosity(logging.INFO)
  #bpe = BPE(codecs.open("code-file", encoding='utf-8'), "@@")
  wordMapPath = "word-map"
  tagMapPath = "tag-map"
  pMapPath = "prefix-list"
  sMapPath = "suffix-list"

  pMap = readAffix(pMapPath)
  sMap = readAffix(sMapPath)
  wordMap = readMap(wordMapPath)
  tagMap = readMap(tagMapPath)

  wordMap.insert(0,"-start-")
  wordMap.insert(0,"-end-")
  wordMap.insert(0,"-unknown-")

  pMap.insert(0,"-start-")
  pMap.insert(0,"-unknown-")
  sMap.insert(0,"-start-")
  sMap.insert(0,"-unknown-")

  feature_sizes = [8,8,2,8,8,4] #num of features for each feature group: capitalization, words, other, prefix_2, suffix_2, previous_tags
  domain_sizes = [3, len(wordMap)+3, 3, len(pMap)+2, len(sMap)+2, len(tagMap)+1]
  num_actions = 45
  embedding_dims = [8,64,8,16,16,16]

  t = time.time()
  hidden_layer_sizes = map(int, FLAGS.hidden_layer_sizes.split(','))
  logging.info('Building training network with parameters: feature_sizes: %s '
               'domain_sizes: %s', feature_sizes, domain_sizes)
  
  test_data_path = '/cs/natlang-user/vivian/wsj-conll/test.conllu'
  logging.info("loading data and precomputing features...")
  test_data = ConllData(test_data_path, wordMap, tagMap, pMap, sMap)

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
  index = 0
  epochs = 0

  epochs, sent_batch = loadBatch(FLAGS.batch_size, epochs, test_data)
  while True:
    sent_batch, epochs, feature_endpoints, gold_tags, words = get_current_features(sent_batch, epochs, test_data, wordMap, tagMap, pMap, sMap)
    tf_eval_metrics = sess.run(tagger.evaluation['logits'], feed_dict={tagger.test_input:feature_endpoints})
    for i in range(FLAGS.batch_size):
      best_action = 0
      best_score = float("-inf")
      for j in range(45):
        if tf_eval_metrics[i][j] > best_score:
          best_score = tf_eval_metrics[i][j]
          best_action = j
      sent_batch[i].set_tag(tagMap[best_action])
    if num_epochs is None:
      num_epochs = epochs
    elif num_epochs < sent_batch[0].get_epoch():
      break

  test_data.reset_index()
  while test_data.has_next_sent():
    sent = test_data.get_next_sent()
    gold_tags = sent.get_tag_list()
    output_tags = sent.get_tag_output()
    assert len(gold_tags) == len(output_tags)
    for idx, tag in enumerate(gold_tags):
      num_tokens += 1
      if tag == output_tags[idx]:
        num_correct += 1
    sent.reset_state()

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
