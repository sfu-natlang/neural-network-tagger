import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import utils
import time, os
from ner_dataset import Dataset
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

def Evaluate(sess, model, dev_data, transition_params_trained, parameters, ner_map):
  total_token_num = 0
  correct_token_num = 0
  start = time.time()
  epoch_num = 0
  while True:
    step = 0
    sent_list = []
    sentences = []
    pos_tags = []
    ner_tags = []
    sentence_lengths = []
    word_lengths = []
    while len(sentences) < parameters['batch_size']:
      sent, epoch_num = advance_sent(epoch_num, dev_data)
      sent_list.append(sent)
      sentences.append(sent.word_ids)
      ner_tags.append(sent.ner_ids)
      sentence_lengths.append(sent.get_sent_len())
    feed_dict = {
      model.input_token_indices: utils.pad_lists(sentences),
      model.input_sent_lengths: sentence_lengths,
      model.input_label_indices:utils.pad_lists(ner_tags),
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

  dev_data.reset_index()
  out_file = open("ner_out2","w") 
  while dev_data.has_next_sent():
    sent = dev_data.get_next_sent()
    words = sent.get_word_list()
    pos = sent.get_pos_list()
    gold_tags = sent.ner_ids
    output_tags = sent.get_tag_output()
    assert len(gold_tags) == len(output_tags)
    for idx, tag in enumerate(gold_tags):
      out_file.write("%s %s %s %s\n"%(words[idx], pos[idx], ner_map[tag], ner_map[output_tags[idx]]))
    out_file.write("\n")
  out_file.close()
  dev_data.reset_index()

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
  parameters['learning_rate'] = 0.005
  parameters['gradient_clipping_value'] = 5.0
  parameters['dropout_rate'] = 0.2
  parameters['maximum_number_of_epochs'] = 10
  parameters['batch_size'] = 32

  loading_time = time.time()
  train_data_path = '/cs/natlang-data/CoNLL/CoNLL-2003/eng.train'
  dev_data_path = '/cs/natlang-data/CoNLL/CoNLL-2003/eng.testa'
  logging.info("loading data and precomputing features...")
  train_data = Dataset(train_data_path)
  train_data.load_dataset()
  test_data = Dataset(dev_data_path)
  test_data.load_dataset(train_data.word_map, train_data.tag_map, train_data.char_map, train_data.ner_map)

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
      pos_tags = []
      ner_tags = []
      sentence_lengths = []
      word_lengths = []
      while len(sentences) < parameters['batch_size']:
        sent, epoch_num = advance_sent(epoch_num, train_data)
        sentences.append(sent.word_ids)
        ner_tags.append(sent.ner_ids)
        pos_tags.append(sent.pos_ids)
        sentence_lengths.append(sent.get_sent_len())
      feed_dict = {
        model.input_token_indices: utils.pad_lists(sentences),
        model.input_sent_lengths: sentence_lengths,
        model.input_label_indices: utils.pad_lists(ner_tags),
        model.dropout_keep_prob: 1-parameters['dropout_rate']
      }
      _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_parameters],
                    feed_dict)
      step += 1
      if epoch_num >= parameters['maximum_number_of_epochs']: 
        break
    Evaluate(sess, model, test_data, transition_params_trained, parameters, train_data.ner_map)
    logging.info("finished training, time is %.2f", time.time()-start)

if __name__ == '__main__':
  main()
