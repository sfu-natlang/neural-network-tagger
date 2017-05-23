import tensorflow as tf
import numpy as np
import codecs
import re
import time
from tensorflow.python.ops import control_flow_ops as cf
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging

def bidirectional_LSTM(input, hidden_state_dimension, initializer, sequence_length=None, output_sequence=True):

  with tf.variable_scope("bidirectional_LSTM"):
    if sequence_length != None:
      batch_size = tf.shape(sequence_length)[0]
    else:
      batch_size = 1
      sequence_length = tf.shape(input)[1]

    lstm_cell = {}
    initial_state = {}
    for direction in ["forward", "backward"]:
      with tf.variable_scope(direction):
        # LSTM cell
        lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(hidden_state_dimension, forget_bias=1.0, initializer=initializer, state_is_tuple=True)
        initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
        initial_output_state = tf.get_variable("initial_output_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
        c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
        h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
        initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

    outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                  lstm_cell["backward"],
                                  input,
                                  dtype=tf.float32,
                                  sequence_length=sequence_length,
                                  initial_state_fw=initial_state["forward"],
                                  initial_state_bw=initial_state["backward"])
    if output_sequence == True:
      outputs_forward, outputs_backward = outputs
      output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
    else:
      final_states_forward, final_states_backward = final_states
      output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=1, name='output')
  return output


class LSTM_CRF_Model(object):

  def __init__(self, dataset, parameters):

    self.verbose = False
    self.input_token_indices = tf.placeholder(tf.int32, shape=(parameters["batch_size"], None), name="batch_sentences")
    self.input_sent_lengths = tf.placeholder(tf.int32, shape=(None,), name="sentence_length")
    self.input_label_indices = tf.placeholder(tf.int32, shape=(parameters["batch_size"], None), name="batch_tags")
    self.input_token_character_indices = tf.placeholder(tf.int32, [None, None], name="input_token_indices")
    self.input_token_lengths = tf.placeholder(tf.int32, [None], name="input_token_lengths")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    #self.cap_features = tf.placeholder(tf.int32, shape=(parameters["batch_size"], None), name="cap_features")
    #self.other_features = tf.placeholder(tf.int32, shape=(parameters["batch_size"], None), name="other_features")
    #self.prefix_features = tf.placeholder(tf.int32, shape=(parameters["batch_size"], None), name="prefix_features")
    #self.suffix_features = tf.placeholder(tf.int32, shape=(parameters["batch_size"], None), name="suffix_features")
    # Initializer
    initializer = tf.contrib.layers.xavier_initializer()

    if parameters['use_character_lstm']:
      with tf.variable_scope("character_embedding"):
        self.character_embedding_weights = tf.get_variable(
          "character_embedding_weights",
          shape=[dataset.alphabet_size, parameters['character_embedding_dimension']],
          initializer=initializer)
        embedded_characters = tf.nn.embedding_lookup(self.character_embedding_weights, self.input_token_character_indices, name='embedded_characters')

      # Character LSTM layer
      with tf.variable_scope('character_lstm') as vs:
        character_lstm_output = bidirectional_LSTM(embedded_characters, parameters['character_lstm_hidden_state_dimension'], initializer,
                               sequence_length=self.input_token_lengths, output_sequence=False)
        token_lstm_input = tf.reshape(character_lstm_output, [parameters["batch_size"], -1, 2*parameters['character_lstm_hidden_state_dimension']])
    
    # Token embedding layer
    with tf.variable_scope("token_embedding"):
      self.token_embedding_weights = tf.get_variable(
        "token_embedding_weights",
        shape=[dataset.vocabulary_size, parameters['token_embedding_dimension']],
        initializer=initializer,
        dtype=tf.float32)
      embedded_tokens = tf.nn.embedding_lookup(self.token_embedding_weights, self.input_token_indices)

    # Concatenate character LSTM outputs and token embeddings
    if parameters['use_character_lstm']:
      with tf.variable_scope("concatenate_token_and_character_vectors"):
        character_lstm_output_reshape = tf.reshape(character_lstm_output, [parameters["batch_size"], -1, 2*parameters['character_lstm_hidden_state_dimension']])
        token_lstm_input = tf.concat([character_lstm_output_reshape, embedded_tokens], axis=2, name='token_lstm_input')
    else:
      token_lstm_input = embedded_tokens

    # Add dropout
    with tf.variable_scope("dropout"):
      token_lstm_input_drop = tf.nn.dropout(token_lstm_input, self.dropout_keep_prob, name='token_lstm_input_drop')
    
    # Token LSTM layer
    with tf.variable_scope('token_lstm') as vs:
      cell_forward = tf.contrib.rnn.LSTMCell(num_units=parameters['token_lstm_hidden_state_dimension'], state_is_tuple=True)
      cell_backward = tf.contrib.rnn.LSTMCell(num_units=parameters['token_lstm_hidden_state_dimension'], state_is_tuple=True)
      token_lstm_output = bidirectional_LSTM(token_lstm_input, parameters['token_lstm_hidden_state_dimension'], initializer, sequence_length=self.input_sent_lengths, output_sequence=True)
      token_lstm_output = tf.reshape(token_lstm_output, [-1, 2*parameters['token_lstm_hidden_state_dimension']])

    # Needed only if Bidirectional LSTM is used for token level
    with tf.variable_scope("feedforward_after_lstm") as vs:
      W = tf.get_variable(
        "W",
        shape=[2 * parameters['token_lstm_hidden_state_dimension'], parameters['token_lstm_hidden_state_dimension']],
        initializer=initializer)
      b = tf.Variable(tf.constant(0.0, shape=[parameters['token_lstm_hidden_state_dimension']]), name="bias")
      outputs = tf.nn.xw_plus_b(token_lstm_output, W, b, name="output_before_tanh")
      outputs = tf.nn.tanh(outputs, name="output_after_tanh")
      outputs = tf.reshape(outputs, [-1, parameters['token_lstm_hidden_state_dimension']])

    with tf.variable_scope("feedforward_before_crf") as vs:
      '''
      self.prefix_embedding_weights = tf.get_variable(
        "prefix_embedding_weights",
        shape=[dataset.prefix_size, parameters['prefix_embedding']],
        initializer=initializer,
        dtype=tf.float32)
      prefix_embedding = tf.nn.embedding_lookup(self.prefix_embedding_weights, self.prefix_features)
      prefix_embedding = tf.reshape(prefix_embedding, [-1, 16])
      self.suffix_embedding_weights = tf.get_variable(
        "suffix_embedding_weights",
        shape=[dataset.suffix_size, parameters['suffix_embedding']],
        initializer=initializer,
        dtype=tf.float32)
      suffix_embedding = tf.nn.embedding_lookup(self.suffix_embedding_weights, self.suffix_features)
      suffix_embedding = tf.reshape(suffix_embedding, [-1, 16])
      outputs = tf.concat([outputs, prefix_embedding, suffix_embedding], axis=1, name='token_affix_output')
      '''
      W = tf.get_variable(
        "W",
        shape=[parameters['token_lstm_hidden_state_dimension'], dataset.number_of_classes],
        initializer=initializer)
      b = tf.Variable(tf.constant(0.0, shape=[dataset.number_of_classes]), name="bias")
      scores = tf.nn.xw_plus_b(outputs, W, b, name="scores")
      self.unary_scores = tf.reshape(scores, [parameters['batch_size'], -1, dataset.number_of_classes])
      self.predictions = tf.argmax(self.unary_scores, 2, name="predictions")

    # CRF layer
    if parameters['use_crf']:
      with tf.variable_scope("crf") as vs:
        # Add start and end tokens
        batch_size = parameters['batch_size']
        small_score = -1000.0
        large_score = 0.0
        sequence_length = tf.shape(self.unary_scores)[1]
        unary_scores_with_start_and_end = tf.concat([self.unary_scores, tf.tile( tf.constant(small_score, shape=[1,1,2]) , [batch_size, sequence_length, 1])], 2)

        start_unary_scores = [[[small_score] * dataset.number_of_classes + [large_score, small_score]]]*batch_size
        end_unary_scores = [[[small_score] * dataset.number_of_classes + [small_score, large_score]]]*batch_size

        self.unary_scores = tf.concat([start_unary_scores, unary_scores_with_start_and_end, end_unary_scores], 1)

        start_index = dataset.number_of_classes
        end_index = dataset.number_of_classes + 1
        input_label_indices_flat_with_start_and_end = tf.concat([ tf.constant(start_index, shape=[batch_size, 1]), self.input_label_indices, tf.constant(end_index, shape=[batch_size, 1]) ], 1)

        self.transition_parameters=tf.get_variable(
          "transitions",
          shape=[dataset.number_of_classes+2, dataset.number_of_classes+2],
          initializer=initializer)

        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
          self.unary_scores, input_label_indices_flat_with_start_and_end, self.input_sent_lengths, transition_params=self.transition_parameters)

        self.loss =  tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')
        self.accuracy = tf.constant(1)
    else:
      with tf.variable_scope("crf") as vs:
        self.transition_parameters = tf.get_variable(
          "transitions",
          shape=[dataset.number_of_classes+2, dataset.number_of_classes+2],
          initializer=initializer)
      # Calculate mean cross-entropy loss
      with tf.variable_scope("loss"):
        fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.unary_scores, labels=self.input_label_indices, name='softmax')
        mask = tf.cast(tf.sign(self.input_label_indices), dtype=tf.float32)
        loss_per_example_per_step = tf.multiply(fake_loss, mask)
        loss_per_example_sum = tf.reduce_sum(loss_per_example_per_step, reduction_indices=[1])
        loss_per_example_average = tf.div(x=loss_per_example_sum, y=tf.cast(self.input_sent_lengths, tf.float32))
        self.loss =  tf.reduce_mean(loss_per_example_average, name='cross_entropy_mean_loss')
      
    #training procedure
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    if parameters['optimizer'] == 'adam':
      self.optimizer = tf.train.AdamOptimizer(parameters['learning_rate'])
    elif parameters['optimizer'] == 'sgd':
      self.optimizer = tf.train.GradientDescentOptimizer(parameters['learning_rate'])
    elif parameters['optimizer'] == 'adadelta':
      self.optimizer = tf.train.AdadeltaOptimizer(parameters['learning_rate'])
    else:
      lr = self._AddLearningRate(0.1, 4000)
      self.optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    if parameters['optimizer'] != None:
      grads_and_vars = self.optimizer.compute_gradients(self.loss)
      if parameters['use_crf']:
        grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
    else:
      self.train_op = self.optimizer.minimize(self.loss)

  def GetStep(self):
    def OnesInitializer(shape, dtype=tf.float32, partition_info=None):
      return tf.ones(shape, dtype)
    step = tf.get_variable("step", [], tf.int32, OnesInitializer)
    return step 

  def _IncrementCounter(self, counter):
    return state_ops.assign_add(counter, 1, use_locking=True)
  def _AddLearningRate(self, initial_learning_rate, decay_steps):
    step = self.GetStep()
    return cf.with_dependencies(
    [self._IncrementCounter(step)],
    tf.train.exponential_decay(initial_learning_rate,
                   step,
                   decay_steps,
                   0.96,
                   staircase=True))


