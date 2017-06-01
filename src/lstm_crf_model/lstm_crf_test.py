import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import utils
import time, os
from dataset import Dataset
from collections import defaultdict, namedtuple
from lstm_char_model import EntityLSTM
import pickle
from tensorflow.python.platform import gfile


DataSet = namedtuple("DataSet", "number_of_classes, vocabulary_size, alphabet_size")

def OutputPath(path):
  return os.path.join("lstm_char_output", path)

def readMap(path):
  return pickle.load(open(path, 'rb'))

def main():
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))
  parameters = {}
  parameters['use_character_lstm'] = True
  parameters['character_embedding_dimension'] = 25
  parameters['token_embedding_dimension'] = 100
  parameters['token_pretrained_embedding_filepath'] = ''
  parameters['pretrained_model_checkpoint_filepath'] = OutputPath('char_model_{0:05d}.ckpt'.format(2))
  parameters['character_lstm_hidden_state_dimension'] = 25
  parameters['token_lstm_hidden_state_dimension'] = 100
  parameters['use_crf'] = True
  parameters['optimizer'] = 'adam'
  parameters['learning_rate'] = 0.005
  parameters['gradient_clipping_value'] = 5.0
  parameters['dropout_rate'] = 0.2
  parameters['maximum_number_of_epochs'] = 10

  loading_time = time.time()
  test_data_path = '/cs/natlang-user/vivian/wsj-conll/test.conllu'
  wordMapPath = 'word_map'
  tagMapPath = 'tag_map'
  charMapPath = 'char_map'
  word_map = readMap(wordMapPath)
  tag_map = readMap(tagMapPath)
  char_map = readMap(charMapPath)

  test_data = Dataset(test_data_path)
  test_data.load_dataset(word_map, tag_map, char_map)

  sess = tf.Session()
  with sess.as_default():
    model = EntityLSTM(test_data, parameters)
    sess.run(tf.global_variables_initializer())

    model_saver = tf.train.Saver(max_to_keep=parameters['maximum_number_of_epochs'])
    model_saver.restore(sess, parameters['pretrained_model_checkpoint_filepath'])
    
    total_token_num = 0
    correct_token_num = 0
    start = time.time()
    transition_params_trained = sess.run(model.transition_parameters)
    start = time.time()
    while test_data.has_next_sent():
      sent = test_data.get_next_sent()
      feed_dict = {
        model.input_token_indices: sent.word_ids,
        model.input_label_indices: sent.tag_ids,
        model.input_token_character_indices: utils.pad_lists(sent.char_lists),
        model.input_token_lengths: sent.word_lengths,
        model.dropout_keep_prob: 1-parameters['dropout_rate']
      }
      unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
      if parameters['use_crf']:
        predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
        predictions = predictions[1:-1]
      else:
        predictions = predictions.tolist()
      gold_labels = sent.tag_ids
      total_token_num += len(predictions)
      for idx, p in enumerate(predictions):
        if p == gold_labels[idx]:
          correct_token_num += 1  
    logging.info('token number is %d, accuracy is %.2f%%, time is %.2f', total_token_num, (100.0*correct_token_num/total_token_num), time.time()-start)

if __name__ == '__main__':
  main()
