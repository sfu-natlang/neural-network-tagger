import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import utils
import time, os
from data_format import ConllData
from collections import defaultdict, namedtuple
from rnn_crf_model import EntityLSTM
from tensorflow.python.platform import gfile


DataSet = namedtuple("DataSet", "number_of_classes, vocabulary_size, alphabet_size")

def OutputPath(path):
  return os.path.join("rnn_models", path)

def readMap(path):
  ret = []
  with open(path, 'rb') as f:
    for idx, line in enumerate(f):
      if idx == 0:
        continue
      ret.append(line.split()[0])
  return ret

def main():
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))
  parameters = {}
  parameters['use_character_lstm'] = False
  parameters['character_embedding_dimension'] = 25
  parameters['token_embedding_dimension'] = 100
  parameters['freeze_token_embeddings'] = False
  parameters['character_lstm_hidden_state_dimension'] = 25
  parameters['token_lstm_hidden_state_dimension'] = 100
  parameters['use_crf'] = False
  parameters['optimizer'] = 'sgd'
  parameters['learning_rate'] = 0.005
  parameters['gradient_clipping_value'] = False
  parameters['dropout_rate'] = 0.5
  parameters['maximum_number_of_epochs'] = 10
  wordMapPath = "word-map"
  tagMapPath = "tag-map"

  wordMap = readMap(wordMapPath)
  tagMap = readMap(tagMapPath)

  wordMap.append("-unknown-")
  wordMap.insert(0,"-padding-")

  loading_time = time.time()
  train_data_path = '/cs/natlang-user/vivian/wsj-conll/train.conllu'
  dev_data_path = '/cs/natlang-user/vivian/wsj-conll/test.conllu'
  logging.info("loading data and precomputing features...")
  train_data = ConllData(train_data_path, wordMap, tagMap)
  logging.info('training sentences: %d', train_data.get_sent_num())
  dev_data = ConllData(dev_data_path, wordMap, tagMap)
  logging.info("logging time: %.2f", time.time() - loading_time)
  train_data.gen_char_map()
  dev_data.gen_char_map()
  dataset = DataSet(len(tagMap), len(wordMap), train_data.char_map_len)

  sess = tf.Session()
  with sess.as_default():
    model = EntityLSTM(dataset, parameters)
    sess.run(tf.global_variables_initializer())
    epoch_num = 0
    start = time.time()
    while True:
      step = 0
      epoch_num += 1
      while train_data.has_next_sent():
        sent = train_data.get_next_sent()
        step += 1
        feed_dict = {
          model.input_token_indices: sent.wordid_list,
          model.input_label_indices_vector: sent.label_vector,
          model.input_token_character_indices: utils.pad_lists(sent.char_list),
          model.input_token_lengths: sent.word_length,
          model.input_label_indices_flat: sent.tagid_list,
          model.dropout_keep_prob: 1-parameters['dropout_rate']
        }
        _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_parameters],
                    feed_dict)
        if step % 10 == 0:
          logging.info('Training %.2f%% done', (100.0*step/train_data.get_sent_num()))

      train_data.reset_index()
      if epoch_num >= parameters['maximum_number_of_epochs']: 
        break
    logging.info("finished training, time is %.2f", time.time()-start)
    model_saver = tf.train.Saver(max_to_keep=parameters['maximum_number_of_epochs'])
    model_saver.save(sess, OutputPath('model_{0:05d}.ckpt'.format(epoch_num)))
    total_token_num = 0
    correct_token_num = 0
    start = time.time()
    while dev_data.has_next_sent():
      sent = dev_data.get_next_sent()
      feed_dict = {
        model.input_token_indices: sent.wordid_list,
        model.input_label_indices_vector: sent.label_vector,
        model.input_token_character_indices: utils.pad_lists(sent.char_list),
        model.input_token_lengths: sent.word_length,
        model.input_label_indices_flat: sent.tagid_list,
        model.dropout_keep_prob: 1
      }
      unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
      if parameters['use_crf']:
          predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
          predictions = predictions[1:-1]
      else:
          predictions = predictions.tolist()
      gold_labels = sent.tagid_list
      assert(len(predictions) == len(gold_labels))
      total_token_num += len(predictions)
      for idx, p in enumerate(predictions):
        if p == gold_labels[idx]:
          correct_token_num += 1
    dev_data.reset_index()
    logging.info('token number is %d, accuracy is %.2f%%, time is %.2f', total_token_num, (100.0*correct_token_num/total_token_num), time.time()-start)

if __name__ == '__main__':
  main()