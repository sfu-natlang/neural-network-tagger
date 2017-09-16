
"""A program to annotate and evaluate a tensorflow neural net tagger."""
import os
import os.path
import time
import tempfile
import tensorflow as tf
import codecs
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from graph_builder import GreedyTagger
import codecs
import pickle
from ner_dataset import Dataset
import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', 'ner_models/model', 'Path to model parameters.')
flags.DEFINE_string('output_path', 'ner_results', 'Path to model parameters.')
flags.DEFINE_string('hidden_layer_sizes', '128',
                    'Comma separated list of hidden layer sizes.')
flags.DEFINE_string('test_corpus', '/cs/natlang-data/CoNLL/CoNLL-2003/eng.testb',
                    'Comma separated list of hidden layer sizes.')
flags.DEFINE_boolean('word_only', False, 'if use word only features')
flags.DEFINE_integer('batch_size', 32,
                     'Number of sentences to process in parallel.')

def Eval(sess):
  """Builds and evaluates a network."""
  logging.set_verbosity(logging.INFO)
  #bpe = BPE(codecs.open("code-file", encoding='utf-8'), "@@")
  wordMapPath = "ner_models/ner_word2id"
  nerMapPath = "ner_models/ner_tag2id"
  pMapPath = "ner_models/ner_prefix2id"
  sMapPath = "ner_models/ner_suffix2id"

  prefix2id = utils.read_pickle_file(pMapPath)
  suffix2id = utils.read_pickle_file(sMapPath)
  word2id = utils.read_pickle_file(wordMapPath)
  tag2id = utils.read_pickle_file(nerMapPath)

  loading_time = time.time()
  logging.info("loading data and precomputing features...")
  dataset = Dataset(None, None, FLAGS.test_corpus, format_list=['FORM', 'a', 'b', 'NER'])
  dataset.load_dataset(word2id=word2id, tag2id=tag2id, prefix2id=prefix2id, suffix2id=suffix2id, fgen=False)

  logging.info('training sentences: %d', dataset.get_sent_num('test'))
  logging.info("logging time: %.2f", time.time() - loading_time)
  '''
  if FLAGS.word_only:
    feature_sizes = [8]
    domain_sizes = [dataset.vocabulary_size]
    embedding_dims = [100]
  else:
    feature_sizes = [8,8,2,8,8,4]#num of features for each feature group: capitalization, words, prefix_2, suffix_2, tags_history
    domain_sizes = [3,dataset.vocabulary_size,3,dataset.prefix_size,dataset.suffix_size,dataset.number_of_classes+1]
    embedding_dims = [8,100,8,50,50,50]

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
  logging.info('Evaluating training network.')
  t = time.time()
  num_epochs = None
  epochs = 0
  logging.info(data.get_sent_num(name))
  epochs, sent_batch = utils.loadBatch(FLAGS.batch_size, epochs, dataset, 'test')
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
  '''

def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  with tf.Session() as sess:
    Eval(sess)

if __name__ == '__main__':
  tf.app.run()
