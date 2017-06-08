import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import utils
import time, os, codecs
from ner_dataset import Dataset
from collections import defaultdict, namedtuple
from lstm_char_model import EntityLSTM
from tensorflow.python.platform import gfile
import pickle

def load_token_vector(parameters):
  file_input = codecs.open(parameters['token_pretrained_embedding_filepath'], 'r', 'UTF-8')
  count = -1
  token_to_vector = {}
  for cur_line in file_input:
    count += 1
    cur_line = cur_line.strip()
    cur_line = cur_line.split(' ')
    if len(cur_line)==0:continue
    token = cur_line[0]
    vector =cur_line[1:]
    token_to_vector[token] = vector
  file_input.close()
  return token_to_vector

def OutputPath(path):
  return os.path.join("lstm_ner_output1", path)

def load_pretrained_token_embeddings(sess, model, dataset, parameters):
  if parameters['token_pretrained_embedding_filepath'] == '':
    return
  start = time.time()
  token_to_vector = load_token_vector(parameters)
  logging.info("token to vector length: %d, vocab size: %d", len(token_to_vector.keys()), len(dataset.word_map))
  initial_weights = sess.run(model.token_embedding_weights.read_value())
  number_of_loaded_word_vectors = 0
  number_of_token_original_case_found = 0
  number_of_token_lowercase_found = 0
  for token in dataset.word_map:
    if token in token_to_vector.keys():
      initial_weights[dataset.word_map.index(token)] = token_to_vector[token]
      number_of_token_original_case_found += 1
    elif token.lower() in token_to_vector.keys():
      initial_weights[dataset.word_map.index(token)] = token_to_vector[token.lower()]
      number_of_token_lowercase_found += 1
    else:
      continue
    number_of_loaded_word_vectors += 1
    if number_of_loaded_word_vectors % 100 == 0:
      logging.info("number of loaded word vectors: %d", number_of_loaded_word_vectors)
  logging.info("finished loading pretrained token embeddings, time used %d", time.time() - start)
  sess.run(model.token_embedding_weights.assign(initial_weights))

def Evaluate(sess, model, data, transition_params_trained, parameters):
  total_token_num = 0
  correct_token_num = 0
  start = time.time()
  while data.has_next_sent('dev'):
    sent = data.get_next_sent('dev')
    feed_dict = {
      model.input_token_indices: sent.word_ids,
      model.input_label_indices: sent.ner_ids,
      model.input_pos_indices: sent.pos_ids,
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
    gold_labels = sent.ner_ids
    total_token_num += len(predictions)
    for idx, p in enumerate(predictions):
      if p == gold_labels[idx]:
        correct_token_num += 1
  data.reset_index('dev')
  return 100.0*correct_token_num/total_token_num

def main():
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))
  log_output = codecs.open("ner_training_log1", 'w')
  parameters = {}
  parameters['use_character_lstm'] = True
  parameters['character_embedding_dimension'] = 25
  parameters['token_embedding_dimension'] = 100
  parameters['token_pretrained_embedding_filepath'] = ''
  parameters['character_lstm_hidden_state_dimension'] = 25
  parameters['token_lstm_hidden_state_dimension'] = 100
  parameters['use_crf'] = True
  parameters['optimizer'] = 'sgd'
  parameters['learning_rate'] = 0.01
  parameters['gradient_clipping_value'] = 5.0
  parameters['dropout_rate'] = 0.2
  parameters['maximum_number_of_epochs'] = 10
  parameters['use_tag_embedding'] = True
  parameters['pos_embedding_dimension'] = 16

  loading_time = time.time()
  train_data_path = '/cs/natlang-data/CoNLL/CoNLL-2003/eng.train'
  dev_data_path = '/cs/natlang-data/CoNLL/CoNLL-2003/eng.testa'
  test_data_path = '/cs/natlang-data/CoNLL/CoNLL-2003/eng.testb'
  logging.info("loading data and precomputing features...")
  
  dataset = Dataset(train_data_path, dev_data_path, test_data_path)
  dataset.load_dataset()

  sess = tf.Session()
  with sess.as_default():
    model = EntityLSTM(dataset, parameters)
    sess.run(tf.global_variables_initializer())
    #load glove token embeddings
    load_pretrained_token_embeddings(sess, model, dataset, parameters)
    epoch_num = 0
    start = time.time()
    best = 0.0
    while True:
      step = 0
      epoch_num += 1
      cost_sum = 0
      while dataset.has_next_sent('train'):
        sent = dataset.get_next_sent('train')
        step += 1
        feed_dict = {
          model.input_token_indices: sent.word_ids,
          model.input_label_indices: sent.ner_ids,
          model.input_pos_indices: sent.pos_ids,
          model.input_token_character_indices: utils.pad_lists(sent.char_lists),
          model.input_token_lengths: sent.word_lengths,
          model.dropout_keep_prob: 1-parameters['dropout_rate']
        }
        _, _, loss, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.transition_parameters],
                    feed_dict)
        cost_sum += loss
        if step % 1000 == 0:
          current = Evaluate(sess, model, dataset, transition_params_trained, parameters)
          log_output.write('EPOCH %d, loss is %.2f'%(epoch_num, cost_sum/1000))
          if current > best:
            logging.info("saving the model...")
            model_saver = tf.train.Saver(max_to_keep=parameters['maximum_number_of_epochs'])
            model_saver.save(sess, OutputPath('char_model_{0:05d}.ckpt'.format(epoch_num)))
            best = current
          cost_sum = 0
      dataset.reset_index('train')
      if epoch_num >= parameters['maximum_number_of_epochs']: 
        break
    log_output.close()
    logging.info("finished training, time is %.2f", time.time()-start)

    total_token_num = 0
    correct_token_num = 0
    start = time.time()
    out_file = open("ner_out","w") 
    while dataset.has_next_sent('test'):
      sent = dataset.get_next_sent('test')
      feed_dict = {
        model.input_token_indices: sent.word_ids,
        model.input_label_indices: sent.ner_ids,
        model.input_pos_indices: sent.pos_ids,
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
      total_token_num += len(predictions)
      gold_labels = sent.ner_ids
      words = sent.get_word_list()
      pos = sent.get_pos_list()
      for idx, p in enumerate(predictions):
        tag = gold_labels[idx]
        if p == tag:
          correct_token_num += 1
        out_file.write("%s %s %s %s\n"%(words[idx], pos[idx], dataset.ner_map[tag], dataset.ner_map[p]))
      out_file.write("\n")
    out_file.close()

    logging.info('token number is %d, accuracy is %.2f%%, time is %.2f', total_token_num, (100.0*correct_token_num/total_token_num), time.time()-start)
if __name__ == '__main__':
  main()
