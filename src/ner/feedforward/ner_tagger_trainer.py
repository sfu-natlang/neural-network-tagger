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
import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tf_master', '', 'TensorFlow execution engine to connect to.')
flags.DEFINE_string('output_path', 'ner_models', 'Top level for output.')
flags.DEFINE_string('training_corpus', '/cs/natlang-user/vivian/eng.train', 'Name of the context input to read training data from.')
flags.DEFINE_string('tuning_corpus', '/cs/natlang-user/vivian/eng.testa', 'Name of the context input to read tuning data from.')
flags.DEFINE_string('hidden_layer_sizes', '128', 'Comma separated list of hidden layer sizes.')
flags.DEFINE_boolean('word_only', False, 'if use word only features')
flags.DEFINE_integer('batch_size', 32, 'Number of sentences to process in parallel.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train for.')
flags.DEFINE_integer('checkpoint_every', 1000, 'Report cost and training accuracy every this many steps.')
flags.DEFINE_float('learning_rate', 0.09, 'Initial learning rate parameter.')
flags.DEFINE_integer('decay_steps', 3600, 'Decay learning rate by 0.96 every this many steps.')
flags.DEFINE_float('momentum', 0.9, 'Momentum parameter for momentum optimizer.')
flags.DEFINE_string('seed', '0', 'Initialization seed for TF variables.')
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
  epochs = 0
  logging.info(data.get_sent_num(name))
  epochs, sent_batch = utils.loadBatch(FLAGS.batch_size, epochs, data, name)
  number_of_words = 0
  while True:
    sent_batch, epochs, feature_endpoints, gold_tags, words = utils.get_current_features(sent_batch, epochs, data, name)
    predictions, tf_eval_metrics = sess.run([tagger.evaluation['predictions'], tagger.evaluation['logits']], feed_dict={tagger.test_input:feature_endpoints})
    utils.set_current_tags(sent_batch, predictions)
    if num_epochs is None:
      num_epochs = epochs
    elif num_epochs < sent_batch[0].get_epoch():
      break
  t_end = time.time()
  data.reset_index(name)
  for sent in sent_batch:
    sent.reset_state()
  accs = []
  correct_preds, total_correct, total_preds = 0., 0., 0.
  
  while data.has_next_sent(name):
    sent = data.get_next_sent(name)
    words = sent.get_word_list()
    number_of_words += len(words)
    gold_labels = sent.ner_ids
    accs += [a==b for (a, b) in zip(gold_labels, sent.output_tags)]
    lab_chunks = set(utils.get_chunks(gold_labels, data.id2tag))
    lab_pred_chunks = set(utils.get_chunks(sent.output_tags, data.id2tag))
    correct_preds += len(lab_chunks & lab_pred_chunks)
    total_preds += len(lab_pred_chunks)
    total_correct += len(lab_chunks)
  test_time = t_end - t
  p = correct_preds / total_preds if correct_preds > 0 else 0
  r = correct_preds / total_correct if correct_preds > 0 else 0
  f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
  logging.info("f1 score:")
  logging.info(f1)
  logging.info(number_of_words)
  data.reset_index(name)
  return test_time

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
  total_time = 0.0
  count = 0
  sess.run(tagger.inits.values())
  #load_pretrained_token_embeddings(sess, tagger, dataset) 
  logging.info('Trainning...')
  num_epochs, sent_batch = utils.loadBatch(FLAGS.batch_size, num_epochs, dataset, 'train')
  while True:
    sent_batch, num_epochs, feature_endpoints, gold_tags, _ = utils.get_current_features(sent_batch, num_epochs, dataset, 'train')
    predictions, logits_score, tf_cost, _ = sess.run([tagger.training['predictions'], tagger.training['logits'], tagger.training['cost'],tagger.training['train_op']], 
                                          feed_dict={tagger.input:feature_endpoints, tagger.labels:gold_tags})
    utils.set_current_tags(sent_batch, gold_tags)
    cost_sum += tf_cost
    num_steps += 1
    if num_steps % FLAGS.checkpoint_every == 0:
      logging.info('Epochs: %d, num steps: %d, '
                   'seconds elapsed: %.2f, avg cost: %.2f, ', num_epochs,
                    num_steps, time.time() - t, cost_sum / FLAGS.checkpoint_every)
      cost_sum = 0.0
      total_time += Eval(sess, tagger, dataset, num_steps, best_eval_metric, 'dev')
      count += 1

    if num_epochs >= FLAGS.num_epochs:
      break

  tagger.saver.save(sess, OutputPath(FLAGS.output_path))

  logging.info('training done! seconds elapsed: %.2f', time.time() - t)
  logging.info('test time seconds elapsed: %.2f', total_time/count)
  
    
def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))
    
  loading_time = time.time()
  logging.info("loading data and precomputing features...")
  dataset = Dataset(FLAGS.training_corpus, FLAGS.tuning_corpus, data_output=FLAGS.output_path)
  dataset.load_dataset()
  logging.info('training sentences: %d', dataset.get_sent_num('train'))
  logging.info("logging time: %.2f", time.time() - loading_time)
  logging.info(dataset.number_of_classes)
  num_actions = dataset.number_of_classes
  if FLAGS.word_only:
    feature_sizes = [8]
    domain_sizes = [dataset.vocabulary_size]
    embedding_dims = [100]
  else:
    feature_sizes = [8,8,2,8,8,4]#num of features for each feature group: capitalization, words, prefix_2, suffix_2, tags_history
    domain_sizes = [3,dataset.vocabulary_size,3,dataset.prefix_size,dataset.suffix_size,dataset.number_of_classes+1]
    embedding_dims = [8,100,8,50,50,50]
  with tf.Session(FLAGS.tf_master) as sess:
    Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, dataset)

if __name__ == '__main__':
  tf.app.run() 

