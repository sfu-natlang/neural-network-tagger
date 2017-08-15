import os
import os.path
import time
import tensorflow as tf
import string
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from graph_builder import GreedyTagger
import pickle
from dataset import Dataset

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tf_master', '', 'TensorFlow execution engine to connect to.')
flags.DEFINE_string('output_path', 'models', 'Top level for output.')
flags.DEFINE_string('params', '0', 'Unique identifier of parameter grid point.')
flags.DEFINE_string('training_corpus', 'training-corpus', 'Name of the context input to read training data from.')
flags.DEFINE_string('tuning_corpus', 'tuning-corpus', 'Name of the context input to read tuning data from.')
flags.DEFINE_string('word_embeddings', None, 
                    'Recordio containing pretrained word embeddings, will be '
'loaded as the first embedding matrix.')
flags.DEFINE_string('hidden_layer_sizes', '128', 'Comma separated list of hidden layer sizes.')
flags.DEFINE_integer('batch_size', 32, 'Number of sentences to process in parallel.')
flags.DEFINE_integer('num_epochs', 5, 'Number of epochs to train for.')
flags.DEFINE_integer('max_steps', 50, 'Max number of parser steps during a training step.')
flags.DEFINE_integer('report_every', 100, 'Report cost and training accuracy every this many steps.')
flags.DEFINE_integer('checkpoint_every', 5000, 'Measure tuning UAS and checkpoint every this many steps.')
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
    predictions, tf_eval_metrics = sess.run([tagger.evaluation['predictions'], tagger.evaluation['logits']], feed_dict={tagger.test_input:feature_endpoints})
    set_current_tags(sent_batch, predictions)
    if num_epochs is None:
      num_epochs = epochs
    elif num_epochs < sent_batch[0].get_epoch():
      break
  t_end = time.time()
  data.reset_index(name)
  while data.has_next_sent(name):
    sent = data.get_next_sent(name)
    words = sent.get_word_list()
    gold_tags = sent.pos_ids
    output_tags = sent.get_tag_output()
    for idx, tag in enumerate(gold_tags):
      num_tokens += 1
      if tag == output_tags[idx]:
        num_correct += 1
    sent.reset_state()
  test_time = t_end - t
  eval_metric = 0 if num_tokens == 0 else (100.0 * num_correct / num_tokens)
  logging.info('Number of Tokens: %d, Seconds elapsed in evaluation: %.2f, '
               'eval metric: %.2f%%', num_tokens, test_time, eval_metric)
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

  num_epochs, sent_batch = loadBatch(FLAGS.batch_size, num_epochs, dataset, 'train')

  while True:
    sent_batch, num_epochs, feature_endpoints, gold_tags, _ = get_current_features(sent_batch, num_epochs, dataset, 'train')
    predictions, logits_score, tf_cost, _ = sess.run([tagger.training['predictions'], tagger.training['logits'], tagger.training['cost'],tagger.training['train_op']], 
                                          feed_dict={tagger.input:feature_endpoints, tagger.labels:gold_tags})
    assert len(gold_tags) == len(predictions)
    set_current_tags(sent_batch, gold_tags)
    cost_sum += tf_cost
    num_steps += 1
    if num_steps % FLAGS.checkpoint_every == 0:
      logging.info('Epochs: %d, num steps: %d, '
                   'seconds elapsed: %.2f, avg cost: %.2f, ', num_epochs,
                    num_steps, time.time() - t, cost_sum / FLAGS.report_every)
      cost_sum = 0.0
      total_time += Eval(sess, tagger, dataset, num_steps, best_eval_metric, 'test')
      count += 1

    if num_epochs >= FLAGS.num_epochs:
      break

  tagger.saver.save(sess, OutputPath('pos_models'))

  logging.info('training done! seconds elapsed: %.2f', time.time() - t)
  logging.info('test time seconds elapsed: %.2f', total_time/count)
  
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
    
def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))
    
  loading_time = time.time()
  train_data_path = '/cs/natlang-user/vivian/wsj-conll/train.conllu'
  dev_data_path = '/cs/natlang-user/vivian/wsj-conll/dev.conllu'
  test_data_path = '/cs/natlang-user/vivian/wsj-conll/test.conllu'
  logging.info("loading data and precomputing features...")
  dataset = Dataset(train_data_path, dev_data_path, test_data_path)
  dataset.load_dataset()
  logging.info('training sentences: %d', dataset.get_sent_num('train'))
  logging.info("logging time: %.2f", time.time() - loading_time)
  logging.info(dataset.number_of_classes)
  feature_sizes = [8,8,2,8,8,4]#num of features for each feature group: capitalization, words, prefix_2, suffix_2, tags_history
  domain_sizes = [3,dataset.vocabulary_size,3,dataset.prefix_size,dataset.suffix_size,dataset.number_of_classes+1]
  num_actions = dataset.number_of_classes
  embedding_dims = [8,100,8,50,50,50]

  with tf.Session(FLAGS.tf_master) as sess:
    Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, dataset)

if __name__ == '__main__':
  tf.app.run() 

