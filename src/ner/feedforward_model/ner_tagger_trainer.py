import os
import os.path
import time
import tensorflow as tf
import string
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from graph_builder import GreedyTagger
import pickle
from ner_dataset import Dataset

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tf_master', '', 'TensorFlow execution engine to connect to.')
flags.DEFINE_string('output_path', 'ner_models', 'Top level for output.')
flags.DEFINE_string('params', '0', 'Unique identifier of parameter grid point.')
flags.DEFINE_string('training_corpus', 'training-corpus', 'Name of the context input to read training data from.')
flags.DEFINE_string('tuning_corpus', 'tuning-corpus', 'Name of the context input to read tuning data from.')
flags.DEFINE_string('word_embeddings', None, 
                    'Recordio containing pretrained word embeddings, will be '
                    'loaded as the first embedding matrix.')
flags.DEFINE_string('hidden_layer_sizes', '128', 'Comma separated list of hidden layer sizes.')
flags.DEFINE_integer('batch_size', 32, 'Number of sentences to process in parallel.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train for.')
flags.DEFINE_integer('max_steps', 50, 'Max number of parser steps during a training step.')
flags.DEFINE_integer('report_every', 100, 'Report cost and training accuracy every this many steps.')
flags.DEFINE_integer('checkpoint_every', 50, 'Measure tuning UAS and checkpoint every this many steps.')
flags.DEFINE_float('learning_rate', 0.09, 'Initial learning rate parameter.')
flags.DEFINE_integer('decay_steps', 3600,
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

def OutputPath(path):
  return os.path.join(FLAGS.output_path, path)

def Eval(sess, tagger, data, num_steps, best_eval_metric, name):
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
  epochs = 0

  epochs, sent_batch = loadBatch(FLAGS.batch_size, epochs, data, name)
  while True:
    sent_batch, epochs, feature_endpoints, gold_tags, words = get_current_features(sent_batch, epochs, data, name)
    tf_eval_metrics = sess.run(tagger.evaluation['logits'], feed_dict={tagger.test_input:feature_endpoints})
    for i in range(FLAGS.batch_size):
      best_action = 0
      best_score = float("-inf")
      for j in range(data.number_of_classes):
        if tf_eval_metrics[i][j] > best_score:
          best_score = tf_eval_metrics[i][j]
          best_action = j
      sent_batch[i].set_tag(data.ner_map[best_action])
    if num_epochs is None:
      num_epochs = epochs
    elif num_epochs < sent_batch[0].get_epoch():
      break
  data.reset_index(name)
  while data.has_next_sent(name):
    sent = data.get_next_sent(name)
    gold_tags = sent.get_ner_list()
    output_tags = sent.get_tag_output()
    assert len(gold_tags) == len(output_tags)
    for idx, tag in enumerate(gold_tags):
      num_tokens += 1
      if tag == output_tags[idx]:
        num_correct += 1
    sent.reset_state()
  data.reset_index(name)
  eval_metric = 0 if num_tokens == 0 else (100.0 * num_correct / num_tokens)
  logging.info('Number of Tokens: %d, Seconds elapsed in evaluation: %.2f, '
               'eval metric: %.2f%%', num_tokens, time.time() - t, eval_metric)
  if eval_metric > best_eval_metric:
    logging.info("saving...")
    tagger.saver.save(sess, OutputPath('model'))
  return max(eval_metric, best_eval_metric)

def Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, dataset):
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

  tagger = GreedyTagger(num_actions, 
                        feature_sizes, 
                        domain_sizes,
                        embedding_dims, 
                        hidden_layer_sizes, 
                        seed=int(FLAGS.seed),
                        gate_gradients=True,
                        average_decay=FLAGS.averaging_decay)

  tagger.AddTraining(FLAGS.batch_size)
  tagger.AddEvaluation(FLAGS.batch_size)
  tagger.AddSaver()
  logging.info('Initializing...')
  num_epochs = 0
  cost_sum = 0.0
  num_steps = 0
  best_eval_metric = 0.0
  index = 0
  sess.run(tagger.inits.values()) 
  logging.info('Trainning...')

  num_epochs, sent_batch = loadBatch(FLAGS.batch_size, num_epochs, dataset, 'train')

  while True:
    sent_batch, num_epochs, feature_endpoints, gold_tags, _ = get_current_features(sent_batch, num_epochs, dataset, 'train')
    logits_score, tf_cost, _ = sess.run([tagger.training['logits'], tagger.training['cost'],tagger.training['train_op']], 
                                          feed_dict={tagger.input:feature_endpoints, tagger.labels:gold_tags})
    cost_sum += tf_cost
    num_steps += 1
    if num_steps % FLAGS.report_every == 0:
      logging.info('Epochs: %d, num steps: %d, '
                   'seconds elapsed: %.2f, avg cost: %.2f, ', num_epochs,
                    num_steps, time.time() - t, cost_sum / FLAGS.report_every)
      cost_sum = 0.0
    if num_steps % 5000 == 0:
      best_eval_metric = Eval(sess, tagger, dataset, num_steps, best_eval_metric, 'dev')

    if num_epochs >= FLAGS.num_epochs:
      break
    
  out_file = open("ner_out1","w")
  t = time.time()
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  epochs = 0
  name = 'test'
  epochs, sent_batch = loadBatch(FLAGS.batch_size, epochs, dataset, name)
  while True:
    sent_batch, epochs, feature_endpoints, gold_tags, words = get_current_features(sent_batch, epochs, dataset, name)
    tf_eval_metrics = sess.run(tagger.evaluation['logits'], feed_dict={tagger.test_input:feature_endpoints})
    for i in range(FLAGS.batch_size):
      best_action = 0
      best_score = float("-inf")
      for j in range(dataset.number_of_classes):
        if tf_eval_metrics[i][j] > best_score:
          best_score = tf_eval_metrics[i][j]
          best_action = j
      sent_batch[i].set_tag(dataset.ner_map[best_action])
    if num_epochs is None:
      num_epochs = epochs
    elif num_epochs < sent_batch[0].get_epoch():
      break
  dataset.reset_index(name)

  while dataset.has_next_sent('test'):
    sent = dataset.get_next_sent('test')
    words = sent.get_word_list()
    pos = sent.get_pos_list()
    gold_tags = sent.get_ner_list()
    output_tags = sent.get_tag_output()
    assert len(gold_tags) == len(output_tags)
    for idx, tag in enumerate(gold_tags):
      out_file.write("%s %s %s %s\n"%(words[idx], pos[idx], tag, output_tags[idx]))
    out_file.write("\n")
  out_file.close()
  logging.info('seconds elapsed: %.2f', time.time() - t)


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
    tag_list = sent.ner_ids
    state = sent.get_next_state()
    word = word_list[state]
    tag = tag_list[state]
    words.append(word)
    tags.append(tag)
    cap_features.extend(sent.cap_features[state])
    word_features.extend(sent.word_features[state])
    #other_features.extend(sent.other_features[state])
    prefix_features.extend(sent.prefix_features[state])
    suffix_features.extend(sent.suffix_features[state])
    tag_features.extend(sent.pos_features[state])
  features.append(','.join(str(e) for e in cap_features))
  features.append(','.join(str(e) for e in word_features))
  #features.append(','.join(str(e) for e in other_features))      
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
  print name
  print size
  sent_batch = []
  for i in range(batch_size):
    if not data.has_next_sent(name):
      data.reset_index(name)
      num_epochs += 1
    sent = data.get_next_sent(name)
    sent.set_epoch(num_epochs)
    sent_batch.append(sent)
    
  return num_epochs, sent_batch
    
def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))
    
  loading_time = time.time()
  train_data_path = '/cs/natlang-data/CoNLL/CoNLL-2003/eng.testa'
  dev_data_path = '/cs/natlang-data/CoNLL/CoNLL-2003/eng.testa'
  test_data_path = '/cs/natlang-data/CoNLL/CoNLL-2003/eng.testb'
  logging.info("loading data and precomputing features...")
  dataset = Dataset(train_data_path, dev_data_path, test_data_path)
  dataset.load_dataset()
  logging.info('training sentences: %d', dataset.get_sent_num('train'))
  logging.info("logging time: %.2f", time.time() - loading_time)

  feature_sizes = [8]*5 #num of features for each feature group: capitalization, words, prefix_2, suffix_2, pos_tags
  domain_sizes = [3, dataset.vocabulary_size, dataset.prefix_size, dataset.suffix_size, dataset.pos_classes]
  num_actions = dataset.number_of_classes
  embedding_dims = [8,64,16,16,16]
  #logging.info(train_data.vocabulary_size)
  #logging.info(train_data.prefix_size)
  #logging.info(train_data.suffix_size)
  with tf.Session(FLAGS.tf_master) as sess:
    Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, dataset)

if __name__ == '__main__':
  tf.app.run() 

