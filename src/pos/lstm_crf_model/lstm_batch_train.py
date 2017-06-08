import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import utils
import time, os
from dataset import Dataset
from collections import defaultdict, namedtuple
from rnn_crf_model import LSTM_CRF_Model as EntityLSTM
from tensorflow.python.platform import gfile


def OutputPath(path):
  return os.path.join("rnn_models", path)

def advance_sent(num_epochs, data):
  if not data.has_next_sent():
    data.reset_index()
    num_epochs += 1
  sent = data.get_next_sent()
  sent.set_epoch(num_epochs)
  return sent, num_epochs

def Evaluate(sess, model, dev_data, transition_params_trained, parameters):
  total_token_num = 0
  correct_token_num = 0
  start = time.time()
  epoch_num = 0
  while True:
    step = 0
    sent_list = []
    sentences = []
    tags = []
    sentence_lengths = []
    word_lengths = []
    while len(sentences) < parameters['batch_size']:
      sent, epoch_num = advance_sent(epoch_num, dev_data)
      sent_list.append(sent)
      sentences.append(sent.word_ids)
      tags.append(sent.tag_ids)
      sentence_lengths.append(sent.get_sent_len())
    feed_dict = {
      model.input_token_indices: utils.pad_lists(sentences),
      model.input_sent_lengths: sentence_lengths,
      model.input_label_indices:utils.pad_lists(tags),
      model.dropout_keep_prob: 1-parameters['dropout_rate']
    }
    unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
    for index in range(parameters["batch_size"]):
      if parameters['use_crf']:
        outputs, _ = tf.contrib.crf.viterbi_decode(unary_scores[index], transition_params_trained)
        sent_list[index].set_sent_tags(outputs[1:-1])
      else:
        outputs = predictions[index]
        sent_list[index].set_sent_tags(outputs)
    if epoch_num >= 1:
      break
  num_tokens = 0
  num_correct = 0
  dev_data.reset_index()
  while dev_data.has_next_sent():
    sent = dev_data.get_next_sent()
    gold_tags = sent.tag_ids
    output_tags = sent.get_tag_output()
    assert len(gold_tags) == len(output_tags)
    for idx, tag in enumerate(gold_tags):
      num_tokens += 1
      if gold_tags[idx] == output_tags[idx]:
        num_correct += 1
  dev_data.reset_index()
  logging.info(num_correct)
  logging.info(num_tokens)
  logging.info('token number is %d, accuracy is %.2f%%', num_tokens, (100.0*num_correct/num_tokens))
  return 100.0 * num_correct / num_tokens

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
  parameters['use_crf'] = True
  parameters['optimizer'] = 'adam'
  parameters['learning_rate'] = 0.002
  parameters['gradient_clipping_value'] = 5.0
  parameters['dropout_rate'] = 0.4
  parameters['maximum_number_of_epochs'] = 10
  parameters['batch_size'] = 32

  loading_time = time.time()
  train_data_path = '/cs/natlang-user/vivian/wsj-conll/train.conllu'
  dev_data_path = '/cs/natlang-user/vivian/wsj-conll/test.conllu'
  logging.info("loading data and precomputing features...")
  train_data = Dataset(train_data_path)
  train_data.load_dataset()
  test_data = Dataset(dev_data_path)
  test_data.load_dataset(train_data.word_map, train_data.tag_map, train_data.char_map)

  sess = tf.Session()
  epoch_num = 0
  with sess.as_default():
    model = EntityLSTM(train_data, parameters)
    sess.run(tf.global_variables_initializer())
    start = time.time()
    best = 0.0
    while True:
      step = 0
      sentences = []
      tags = []
      sentence_lengths = []
      word_lengths = []
      while len(sentences) < parameters['batch_size']:
        sent, epoch_num = advance_sent(epoch_num, train_data)
        sentences.append(sent.word_ids)
        tags.append(sent.tag_ids)
        sentence_lengths.append(sent.get_sent_len())
      feed_dict = {
        model.input_token_indices: utils.pad_lists(sentences),
        model.input_sent_lengths: sentence_lengths,
        model.input_label_indices:utils.pad_lists(tags),
        model.dropout_keep_prob: 1-parameters['dropout_rate']
      }
      _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_parameters],
                    feed_dict)
      step += 1
      '''
      if step % 10 == 0:
        current = Evaluate(sess, model, test_data, transition_params_trained, parameters)
        if current > best:
          model_saver = tf.train.Saver(max_to_keep=parameters['maximum_number_of_epochs'])
          model_saver.save(sess, OutputPath('char_model_{0:05d}.ckpt'.format(epoch_num)))
          best = current
        logging.info('EPOCH %d, Training %.2f%% done', epoch_num, (100.0*step/train_data.get_sent_num()))
        logging.info('best accuracy is %.2f%%', best)
      '''
      if epoch_num >= parameters['maximum_number_of_epochs']: 
        break
    best = Evaluate(sess, model, test_data, transition_params_trained, parameters)
    logging.info("finished training, time is %.2f", time.time()-start)

if __name__ == '__main__':
  main()
