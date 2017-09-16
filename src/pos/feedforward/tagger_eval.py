
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
import utils
from dataset import Dataset

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', 'models/pos_models', 'Path to model parameters.')
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

def set_current_tags(sent_batch, predictions):
  for idx, sent in enumerate(sent_batch):
    sent.set_tag(predictions[idx])

def get_current_features(sent_batch, num_epochs, data, name):
  batch_size = len(sent_batch)
  new_sent_batch = [sent for sent in sent_batch if sent.has_next_state()]
  for sent in sent_batch:
    if not sent.has_next_state():
      sent.reset_state()

  while len(new_sent_batch) < batch_size:
    sent, num_epochs = advance_sent(num_epochs, data, name)
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
    tag_list = sent.pos_ids
    state = sent.get_next_state()
    word = word_list[state]
    tag = tag_list[state]
    words.append(word)
    tags.append(tag)
    cap_features.extend(sent.cap_features[state])
    word_features.extend(sent.word_features[state])
    other_features.extend(sent.other_features[state])
    prefix_features.extend(sent.prefix_features[state])
    suffix_features.extend(sent.suffix_features[state])
    tag_features.extend(sent.get_history_features())
  features.append(','.join(str(e) for e in cap_features))
  features.append(','.join(str(e) for e in word_features))
  features.append(','.join(str(e) for e in other_features))      
  features.append(','.join(str(e) for e in prefix_features))
  features.append(','.join(str(e) for e in suffix_features))
  features.append(','.join(str(e) for e in tag_features))
  return new_sent_batch, num_epochs, features, tags, words

def advance_sent(num_epochs, data, name):
  if not data.has_next_sent(name):
    data.reset_index(name)
    num_epochs += 1
  sent = data.get_next_sent(name)
  sent.set_epoch(num_epochs)
  return sent, num_epochs

def loadBatch(batch_size, num_epochs, data, name):
  size = data.get_sent_num(name)
  sent_batch = []
  for i in range(batch_size):
    if not data.has_next_sent(name):
      data.reset_index(name)
      num_epochs += 1
    sent = data.get_next_sent(name)
    sent.set_epoch(num_epochs)
    sent_batch.append(sent)
    
  return num_epochs, sent_batch

def Eval(sess):
  """Builds and evaluates a network."""
  logging.set_verbosity(logging.INFO)
  #bpe = BPE(codecs.open("code-file", encoding='utf-8'), "@@")
  wordMapPath = "word_index"
  pMapPath = "prefix_index"
  sMapPath = "suffix_index"
  charMapPath = "char_index"
  posMapPath = "tag_index"


  pMap = utils.read_pickle_file(pMapPath)
  sMap = utils.read_pickle_file(sMapPath)
  wordMap = utils.read_pickle_file(wordMapPath)
  charMap = utils.read_pickle_file(charMapPath)
  posMap = utils.read_pickle_file(posMapPath)

  loading_time = time.time()
  test_data_path = '/cs/natlang-user/vivian/wsj-conll/test.conllu'
  logging.info("loading data and precomputing features...")
  dataset = Dataset(None, None, test_data_path)
  dataset.load_dataset(word_index=wordMap, tag_index=posMap, prefix_index=pMap, suffix_index=sMap, fgen=False)

  logging.info('training sentences: %d', dataset.get_sent_num('test'))
  logging.info("logging time: %.2f", time.time() - loading_time)

  feature_sizes = [8,8,2,8,8,4]#num of features for each feature group: capitalization, words, prefix_2, suffix_2, tags_history
  domain_sizes = [3,dataset.vocabulary_size,3,dataset.prefix_size,dataset.suffix_size,dataset.number_of_classes+1]
  num_actions = dataset.number_of_classes
  embedding_dims = [8,100,8,50,50,50]
  hidden_layer_sizes = map(int, FLAGS.hidden_layer_sizes.split(','))
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

  start = time.time()
  graph_building_time = 0.0
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  index = 0
  epochs = 0
  name = "test"
  epochs, sent_batch = loadBatch(FLAGS.batch_size, epochs, dataset, name)
  while True:
    sent_batch, epochs, feature_endpoints, gold_tags, words = get_current_features(sent_batch, epochs, dataset, name)
    #t = time.time()
    predictions, tf_eval_metrics = sess.run([tagger.evaluation['predictions'], tagger.evaluation['logits']], feed_dict={tagger.test_input:feature_endpoints})
    #graph_building_time += time.time() - t
    set_current_tags(sent_batch, predictions)
    
    if num_epochs is None:
      num_epochs = epochs
    if num_epochs < sent_batch[0].get_epoch():
      break
  end = time.time()

  dataset.reset_index(name)
  while dataset.has_next_sent(name):
    sent = dataset.get_next_sent(name)
    words = sent.get_word_list()
    gold_tags = sent.pos_ids
    output_tags = sent.get_tag_output()
    for idx, tag in enumerate(gold_tags):
      num_tokens += 1
      if tag == output_tags[idx]:
        num_correct += 1
    sent.reset_state()
  test_time = end - start
  eval_metric = 0 if num_tokens == 0 else (100.0 * num_correct / num_tokens)
  logging.info('Number of Tokens: %d, Seconds elapsed in evaluation: %.2f, '
               'eval metric: %.2f%%', num_tokens, test_time, eval_metric)

def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  with tf.Session() as sess:
    Eval(sess)


if __name__ == '__main__':
  tf.app.run()
