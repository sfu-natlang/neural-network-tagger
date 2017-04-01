import os
import os.path
import time
import tensorflow as tf
import string
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from graph_builder import GreedyTagger
from train_example import TrainExample
import codecs
import pickle
from data.data_pool import DataPool
from apply_bpe import BPE

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

'''
 combine the segmented sub-words into one word
 using vote method to find the tag
 '''
def find_tag(tag_list):
  ret = 0
  count = 1
  index = 1
  while index < len(tag_list):
    if tag_list[index] == tag_list[ret]:
      count += 1
    else:
      count -= 1

    if count == 0:
      ret = index
      count = 1
    index += 1
  return tag_list[ret]

def find_word(word_list):
  ret = []
  wlen = len(word_list)-1
  for index, word in enumerate(word_list):
    if index == wlen:
      ret.append(word)
    else:
      ret.append(word[:-2])
  return "".join(ret)


def combine_seg(word_input, tag_input):
  word_output = []
  tag_output = []
  start = 0
  while start < len(word_input):
    end = start
    while(word_input[end][-2:] == "@@"):
      end += 1
    word_output.append(find_word(word_input[start : end+1]))
    tag_output.append(find_tag(tag_input[start : end+1]))
    start = end + 1
  return word_output, tag_output

def OutputPath(path):
  return os.path.join(FLAGS.output_path, path)

def Eval(sess, tagger, test_data, num_steps, best_eval_metric, tagMap, seg_tokens, tags):
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
  output = []
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  index = 0
  epochs = 0
  while True:
    index, epochs, feature_endpoints, gold_tags, words = loadBatch(FLAGS.batch_size, index, test_data, epochs)
    tf_eval_metrics = sess.run(tagger.evaluation['logits'], feed_dict={tagger.test_input:feature_endpoints})
    for i in range(FLAGS.batch_size):
      best_action = 0
      best_score = float("-inf")
      for j in range(45):
        if tf_eval_metrics[i][j] > best_score:
          best_score = tf_eval_metrics[i][j]
          best_action = j
      output.append(best_action)
    if num_epochs is None:
      num_epochs = epochs
    elif num_epochs < epochs:
      break
  logging.info(len(tags))
  logging.info(len(output))
  words_out, tags_out = combine_seg(seg_tokens, output)
  for idx, tag in enumerate(tags):
    if tag == tagMap[tags_out[idx]]:
      num_correct += 1
  eval_metric = 0 if len(tags) == 0 else (100.0 * num_correct / len(tags))
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


def Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, wordMap, tagMap, pMap, sMap, bpe):
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
  #config['train'] = 'wsj_0[2-9][0-9][0-9].mrg.3.pa.gs.tab|wsj_1[0-9][0-9][0-9].mrg.3.pa.gs.tab|wsj_2[0-1][0-9][0-9].mrg.3.pa.gs.tab'
  config['train'] = 'wsj_02[0-9][0-9].mrg.3.pa.gs.tab'
  config['test'] = 'wsj_22[0-9][0-9].mrg.3.pa.gs.tab'
  config['data_path'] = '/cs/natlang-user/vivian/penn-wsj-deps'

  trainDataPool = DataPool(data_format  = config['format'],
                           data_regex   = config['train'],
                           data_path    = config['data_path'])
  testDataPool = DataPool(data_format  = config['format'],
                           data_regex   = config['test'],
                           data_path    = config['data_path'])

  logging.info('Loading the training data...')
  '''
  load the training examples
  '''
  tokens = []
  while trainDataPool.has_next_data():
    sent = trainDataPool.get_next_data()
    sent_words = sent.get_word_list()
    sent_tags = sent.get_pos_list()
    new_sent_words, sent_tags = bpe.segment(sent_words, sent_tags)
    sent_words = process_seg_sent(new_sent_words)
    for idx, word in enumerate(sent_words):
      features = []
      word_features = []
      other_features = []
      prefix_features = []
      suffix_features = []
      gen_word_features(sent_words, idx, wordMap, word_features)
      gen_other_features(other_features, word)
      #gen_prefix_feature2(sent_words, idx, pMap, prefix_features)
      #gen_suffix_feature2(sent_words, idx, sMap, suffix_features)
      features.append(word_features)
      features.append(other_features)
      #features.append(prefix_features)
      #features.append(suffix_features)
      label = tagMap.index(sent_tags[idx])
      tokens.append(TrainExample(word, features, label)) 


  logging.info('Loading the test data...')
  test_tokens = []
  origin_tokens = []
  origin_tags = []
  while testDataPool.has_next_data():
    sent = testDataPool.get_next_data()
    sent_words = sent.get_word_list()
    sent_tags = sent.get_pos_list()
    origin_tags.extend(sent_tags)
    new_sent_words, sent_tags = bpe.segment(sent_words, sent_tags)
    origin_tokens.extend(new_sent_words)
    sent_words = process_seg_sent(new_sent_words)
    for idx, word in enumerate(sent_words):
      features = []
      word_features = []
      other_features = []
      prefix_features = []
      suffix_features = []
      gen_word_features(sent_words, idx, wordMap, word_features)
      gen_other_features(other_features, word)
      #gen_prefix_feature2(sent_words, idx, pMap, prefix_features)
      #gen_suffix_feature2(sent_words, idx, sMap, suffix_features)
      features.append(word_features)
      features.append(other_features)
      #features.append(prefix_features)
      #features.append(suffix_features)

      label = tagMap.index(sent_tags[idx])
      test_tokens.append(TrainExample(word, features,label)) 


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
  sess.run(tagger.inits.values())

  index = 0 
  logging.info('Trainning...')
  while num_epochs < 10:
    index, num_epochs, feature_endpoints, gold_tags, _ = loadBatch(FLAGS.batch_size, index, tokens, num_epochs)
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
      best_eval_metric = Eval(sess, tagger, test_tokens, num_steps, best_eval_metric, tagMap, origin_tokens, origin_tags)

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
    #prefix_features.extend(token.get_features()[2])
    #suffix_features.extend(token.get_features()[3])
    tags.append(token.get_label())
    index += 1
  features.append(','.join(str(e) for e in word_features))
  features.append(','.join(str(e) for e in other_features))      
  #features.append(','.join(str(e) for e in prefix_features))
  #features.append(','.join(str(e) for e in suffix_features))
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
    
def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))
  bpe = BPE(codecs.open("code-file", encoding='utf-8'), "@@")
  wordMapPath = "word-map"
  tagMapPath = "tag-map"
  pMapPath = "prefix-list"
  sMapPath = "suffix-list"
  pMapPath3 = "prefix-list3"
  sMapPath3 = "suffix-list3"
  pMap = readAffix(pMapPath)
  sMap = readAffix(sMapPath)
  wordMap = readMap(wordMapPath)
  tagMap = readMap(tagMapPath)

  wordMap, _ = bpe.segment(wordMap)
  wordMap = list(set(process_seg_sent(wordMap)))
  wordMap.insert(0,"-start-")
  wordMap.insert(0,"-end-")
  wordMap.insert(0,"-unknown-")

  pMap.insert(0,"-start-")
  pMap.insert(0,"-unknown-")
  sMap.insert(0,"-start-")
  sMap.insert(0,"-unknown-")

  feature_sizes = [8,2]
  domain_sizes = [len(wordMap)+3, 3]
  num_actions = 45
  embedding_dims = [64,8]
  logging.info('Preparing Lexicon...')
  logging.info(len(wordMap))
  logging.info(len(tagMap))    
  with tf.Session(FLAGS.tf_master) as sess:
    Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims, wordMap, tagMap, pMap, sMap, bpe)

if __name__ == '__main__':
  tf.app.run() 

