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
  parameters['use_character_lstm'] = True
  parameters['character_embedding_dimension'] = 25
  parameters['token_embedding_dimension'] = 100
  parameters['freeze_token_embeddings'] = False
  parameters['character_lstm_hidden_state_dimension'] = 25
  parameters['token_lstm_hidden_state_dimension'] = 100
  parameters['use_crf'] = True
  parameters['optimizer'] = 'adam'
  parameters['learning_rate'] = 0.005
  parameters['gradient_clipping_value'] = 5.0
  parameters['dropout_rate'] = 0.5
  parameters['maximum_number_of_epochs'] = 10
  parameters['pretrained_model_checkpoint_filepath'] = OutputPath('model_{0:05d}.ckpt'.format(1))
  wordMapPath = "word-map"
  tagMapPath = "tag-map"

  wordMap = readMap(wordMapPath)
  tagMap = readMap(tagMapPath)

  wordMap.append("-unknown-")
  wordMap.insert(0,"-padding-")

  loading_time = time.time()

  test_data_path = '/cs/natlang-user/vivian/wsj-conll/test.conllu'
  test_data = ConllData(test_data_path, wordMap, tagMap)
  logging.info("logging time: %.2f", time.time() - loading_time)
  test_data.gen_char_map()

  dataset = DataSet(len(tagMap), len(wordMap), test_data.char_map_len)

  sess = tf.Session()
  with sess.as_default():
    model = EntityLSTM(dataset, parameters)
    sess.run(tf.global_variables_initializer())
    model_saver = tf.train.Saver(max_to_keep=parameters['maximum_number_of_epochs'])
    model_saver.restore(sess, parameters['pretrained_model_checkpoint_filepath'])
    
    total_token_num = 0
    correct_token_num = 0
    start = time.time()
    transition_params_trained = sess.run(model.transition_parameters)
    while test_data.has_next_sent():
      sent = test_data.get_next_sent()
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
    test_data.reset_index()
    logging.info('token number is %d, accuracy is %.2f%%, time is %.2f', total_token_num, (100.0*correct_token_num/total_token_num), time.time()-start)

if __name__ == '__main__':
  main()