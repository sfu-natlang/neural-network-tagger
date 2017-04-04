import os
import os.path
import time
import tensorflow as tf
import string
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from graph_builder import GreedyTagger
import pickle
from data_format import ConllData

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
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train for.')
flags.DEFINE_integer('max_steps', 50, 'Max number of parser steps during a training step.')
flags.DEFINE_integer('report_every', 100, 'Report cost and training accuracy every this many steps.')
flags.DEFINE_integer('checkpoint_every', 5000, 'Measure tuning UAS and checkpoint every this many steps.')
flags.DEFINE_float('learning_rate', 0.08, 'Initial learning rate parameter.')
flags.DEFINE_integer('decay_steps', 3600,
                     'Decay learning rate by 0.96 every this many steps.')
flags.DEFINE_float('momentum', 0.9,
                   'Momentum parameter for momentum optimizer.')
flags.DEFINE_string('seed', '1', 'Initialization seed for TF variables.')
flags.DEFINE_string('pretrained_params', None,
                    'Path to model from which to load params.')
flags.DEFINE_string('pretrained_params_names', None,
                    'List of names of tensors to load from pretrained model.')
flags.DEFINE_float('averaging_decay', 0.9999,
                   'Decay for exponential moving average when computing'
'averaged parameters, set to 1 to do vanilla averaging.')

def OutputPath(path):
  return os.path.join(FLAGS.output_path, path)

def Eval(sess, tagger, test_data, num_steps, best_eval_metric, wordMap, tagMap, pMap, sMap):
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
  test_data.reset_index()
  epochs, sent_batch = loadBatch(FLAGS.batch_size, epochs, test_data)
  logging.info(epochs)
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
  if eval_metric > best_eval_metric:
    logging.info("saving")
    tagger.saver.save(sess, OutputPath('model'))
  return max(eval_metric, best_eval_metric)

def process_seg_sent(sent):
  output = []
  for word in sent:
    if word[-2:] == "@@":
      output.append(word[:-2])
    else:
      output.append(word)
  return output


def Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, wordMap, tagMap, pMap, sMap, train_data, dev_data):
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

  num_epochs, sent_batch = loadBatch(FLAGS.batch_size, num_epochs, train_data)

  while True:
    sent_batch, num_epochs, feature_endpoints, gold_tags, _ = get_current_features(sent_batch, num_epochs, train_data, wordMap, tagMap, pMap, sMap)
    tf_cost, _ = sess.run([tagger.training['cost'],tagger.training['train_op']], feed_dict={tagger.input:feature_endpoints, tagger.labels:gold_tags})
    cost_sum += tf_cost
    num_steps += 1
    '''
    if num_steps % FLAGS.report_every == 0:
      logging.info('Epochs: %d, num steps: %d, '
                   'seconds elapsed: %.2f, avg cost: %.2f, ', num_epochs,
                    num_steps, time.time() - t, cost_sum / FLAGS.report_every)
      cost_sum = 0.0
    '''
    if num_steps % 5000 == 0:
      best_eval_metric = Eval(sess, tagger, dev_data, num_steps, best_eval_metric, wordMap, tagMap, pMap, sMap)

    if num_epochs >= FLAGS.num_epochs:
      break

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
  word_features = []
  other_features = []
  prefix_features = []
  suffix_features = []

  for sent in new_sent_batch:
    word_list = sent.get_word_list()
    tag_list = sent.get_tag_list()
    state = sent.get_next_state()
    word = word_list[state]
    tag = get_index(tag_list[state], tagMap)
    words.append(word)
    tags.append(tag)
    gen_word_features(word_list, state, wordMap, word_features)
    gen_other_features(other_features, word)
    gen_prefix_feature2(word_list, state, pMap, prefix_features)
    gen_suffix_feature2(word_list, state, sMap, suffix_features)

  features.append(','.join(str(e) for e in word_features))
  features.append(','.join(str(e) for e in other_features))      
  features.append(','.join(str(e) for e in prefix_features))
  features.append(','.join(str(e) for e in suffix_features))
  #features.append(','.join(str(e) for e in prefix_features3))
  #features.append(','.join(str(e) for e in suffix_features3)) 
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
      logging.info("!!!")
    sent = data.get_next_sent()
    sent.set_epoch(num_epochs)
    sent_batch.append(sent)
    
  return num_epochs, sent_batch

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
    
def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))
  #bpe = BPE(codecs.open("code-file", encoding='utf-8'), "@@")
  wordMapPath = "word-map"
  tagMapPath = "tag-map"
  pMapPath2 = "prefix-list"
  sMapPath2 = "suffix-list"
  pMapPath3 = "prefix-list3"
  sMapPath3 = "suffix-list3"

  pMap2 = readAffix(pMapPath2)
  sMap2 = readAffix(sMapPath2)
  pMap3 = readAffix(pMapPath3)
  sMap3 = readAffix(sMapPath3)
  wordMap = readMap(wordMapPath)
  tagMap = readMap(tagMapPath)

  wordMap.insert(0,"-start-")
  wordMap.insert(0,"-end-")
  wordMap.insert(0,"-unknown-")

  pMap2.insert(0,"-start-")
  pMap2.insert(0,"-unknown-")
  sMap2.insert(0,"-start-")
  sMap2.insert(0,"-unknown-")

  pMap3.insert(0,"-start-")
  pMap3.insert(0,"-unknown-")
  sMap3.insert(0,"-start-")
  sMap3.insert(0,"-unknown-")


  feature_sizes = [8,2,8,8] #num of features for each feature group: words, other, prefix_2, suffix_2
  domain_sizes = [len(wordMap)+3, 3, len(pMap2)+2, len(sMap2)+2]
  num_actions = 45
  embedding_dims = [64,8,16,16]

  train_data_path = '/cs/natlang-user/vivian/wsj-conll/train.conllu'
  dev_data_path = '/cs/natlang-user/vivian/wsj-conll/dev.conllu'

  train_data = ConllData(train_data_path)
  dev_data = ConllData(dev_data_path)

  with tf.Session(FLAGS.tf_master) as sess:
    Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, wordMap, tagMap, pMap2, sMap2, train_data, dev_data)

if __name__ == '__main__':
  tf.app.run() 

