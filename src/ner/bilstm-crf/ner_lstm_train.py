import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import utils
import time, os, codecs
from ner_dataset import Dataset
from collections import defaultdict, namedtuple
from ner_lstm_model import EntityLSTM
from tensorflow.python.platform import gfile
import pickle

def OutputPath(path):
  return os.path.join("lstm_ner_models", path)

def Evaluate(sess, model, dataset, transition_params_trained, parameters, epoch_num):

  start = time.time()
  accs = []
  correct_preds, total_correct, total_preds = 0., 0., 0.
  word_count = 0
  while dataset.has_next_sent('test'):
    sent = dataset.get_next_sent('test')
    feed_dict = {
      model.input_token_indices: sent.word_ids,
      model.input_token_character_indices: utils.pad_lists(sent.char_lists),
      model.input_token_lengths: sent.word_lengths,
      model.dropout_keep_prob: 1
    }
    unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
    if parameters['use_crf']:
      predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
      predictions = predictions[1:-1]
    gold_labels = sent.ner_ids
    words = sent.word_ids
    word_count += len(words)
    accs += [a==b for (a, b) in zip(gold_labels, predictions)]
    lab_chunks = set(utils.get_chunks(gold_labels, dataset.ner_map))
    lab_pred_chunks = set(utils.get_chunks(predictions, dataset.ner_map))
    #logging.info(sent.ner_ids)
    #logging.info(predictions)
    correct_preds += len(lab_chunks & lab_pred_chunks)
    total_preds += len(lab_pred_chunks)
    total_correct += len(lab_chunks)
    
  p = correct_preds / total_preds if correct_preds > 0 else 0
  r = correct_preds / total_correct if correct_preds > 0 else 0
  f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

  test_time = time.time() - start
  dataset.reset_index('test')
  logging.info("epoch: %d, f1 score: %.2f", epoch_num, f1*100.0)

  return test_time




def main():
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))
  parameters = {}
  parameters['use_crf'] = True
  parameters['use_character_lstm'] = True
  parameters['character_embedding_dimension'] = 25
  parameters['token_embedding_dimension'] = 100
  parameters['token_pretrained_embedding_filepath'] = '/cs/natlang-user/vivian/NeuroNER/data/word_vectors/glove.6B.100d.txt'
  #parameters['token_pretrained_embedding_filepath'] = ''
  parameters['character_lstm_hidden_state_dimension'] = 50
  parameters['token_lstm_hidden_state_dimension'] = 100
  parameters['optimizer'] = 'sgd'
  parameters['learning_rate'] = 0.005
  parameters['gradient_clipping_value'] = 0
  parameters['dropout_rate'] = 0.5
  parameters['maximum_number_of_epochs'] = 50
  parameters['freeze_token_embeddings'] = False
  
  loading_time = time.time()
  train_data_path = '/cs/natlang-user/vivian/engonto.train'
  dev_data_path = '/cs/natlang-user/vivian/engonto.testa'
  test_data_path = '/cs/natlang-user/vivian/engonto.testb'
  logging.info("loading data and precomputing features...")
  
  dataset = Dataset(train_data_path, dev_data_path, test_data_path, use_char=True)
  dataset.load_dataset()

  logging.info(dataset.ner_map)
  logging.info(dataset.ner_index)
  logging.info(time.time()-loading_time)
  total_time = 0.0

  sess = tf.Session()
  with sess.as_default():
    model = EntityLSTM(dataset, parameters)
    sess.run(tf.global_variables_initializer())
    #load glove token embeddings
    model.load_pretrained_token_embeddings(sess, dataset, parameters)
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
          model.input_token_character_indices: utils.pad_lists(sent.char_lists),
          model.input_token_lengths: sent.word_lengths,
          model.dropout_keep_prob: 1-parameters['dropout_rate']
        }
        if parameters['use_crf']:
          _, loss, transition_params_trained = sess.run(
                    [model.train_op, model.loss, model.transition_parameters],
                    feed_dict)
        else:
          _, loss = sess.run(
                    [model.train_op, model.loss],
                    feed_dict)
          transition_params_trained = None

        '''
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
        '''
      current = Evaluate(sess, model, dataset, transition_params_trained, parameters, epoch_num)
      dataset.reset_index('train')
      if epoch_num >= parameters['maximum_number_of_epochs']: 
        break
      
      model_saver = tf.train.Saver(max_to_keep=parameters['maximum_number_of_epochs'])
      model_saver.save(sess, OutputPath('char_model'))
      
      total_time += Evaluate(sess, model, dataset, transition_params_trained, parameters, epoch_num)
    logging.info("done")

if __name__ == '__main__':
  main()
