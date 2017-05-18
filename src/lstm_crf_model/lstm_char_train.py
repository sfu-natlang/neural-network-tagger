import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import utils
import time, os
from dataset import Dataset
from collections import defaultdict, namedtuple
from lstm_char_model import EntityLSTM
from tensorflow.python.platform import gfile


def OutputPath(path):
  return os.path.join("rnn_models", path)


def Evaluate(sess, model, dev_data, transition_params_trained, parameters):
  total_token_num = 0
  correct_token_num = 0
  start = time.time()
  while dev_data.has_next_sent():
    sent = dev_data.get_next_sent()
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
    assert(len(predictions) == len(gold_labels))
    total_token_num += len(predictions)
    for idx, p in enumerate(predictions):
      if p == gold_labels[idx]:
        correct_token_num += 1
  dev_data.reset_index()
  logging.info('token number is %d, accuracy is %.2f%%, time is %.2f', total_token_num, (100.0*correct_token_num/total_token_num), time.time()-start)
  return correct_token_num

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
  parameters['maximum_number_of_epochs'] = 100

  loading_time = time.time()
  train_data_path = '/cs/natlang-user/vivian/wsj-conll/train.conllu'
  dev_data_path = '/cs/natlang-user/vivian/wsj-conll/test.conllu'
  logging.info("loading data and precomputing features...")
  train_data = Dataset(train_data_path)
  train_data.load_dataset()
  test_data = Dataset(dev_data_path)
  test_data.load_dataset(train_data.word_map, train_data.tag_map, train_data.char_map)
  logging.info(test_data.get_next_sent().tag_ids)
  sess = tf.Session()
  with sess.as_default():
    model = EntityLSTM(train_data, parameters)
    sess.run(tf.global_variables_initializer())
    epoch_num = 0
    start = time.time()
    best = 0
    while True:
      step = 0
      epoch_num += 1
      while train_data.has_next_sent():
        sent = train_data.get_next_sent()
        step += 1
        feed_dict = {
          model.input_token_indices: sent.word_ids,
          model.input_label_indices: sent.tag_ids,
          model.input_token_character_indices: utils.pad_lists(sent.char_lists),
          model.input_token_lengths: sent.word_lengths,
          model.dropout_keep_prob: 1-parameters['dropout_rate']
        }
        _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_parameters],
                    feed_dict)
        if step % 1000 == 0:
          current = Evaluate(sess, model, test_data, transition_params_trained, parameters)
          logging.info('EPOCH %d, Training %.2f%% done', epoch_num, (100.0*step/train_data.get_sent_num()))
          logging.info('best accuracy is %.2f%%', (100.0*best/56684.0))
          if current > best:
            model_saver = tf.train.Saver(max_to_keep=parameters['maximum_number_of_epochs'])
            model_saver.save(sess, OutputPath('char_model_{0:05d}.ckpt'.format(epoch_num)))
            best = current
      train_data.reset_index()
      if epoch_num >= parameters['maximum_number_of_epochs']: 
        break
    logging.info("finished training, time is %.2f", time.time()-start)

    total_token_num = 0
    correct_token_num = 0
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
      logging.info("Train...")
      logging.info(sent.tag_ids)
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
