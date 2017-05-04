import tensorflow as tf
import numpy as np
import codecs
import re
import time

def bidirectional_LSTM(input, hidden_state_dimension, initializer, sequence_length=None, output_sequence=True):

		with tf.variable_scope("bidirectional_LSTM"):
				if sequence_length == None:
						batch_size = 1
						sequence_length = tf.shape(input)[1]
						sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')
				else:
						batch_size = tf.shape(sequence_length)[0]

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
		"""
		An LSTM architecture for named entity recognition.
		Uses a character embedding layer followed by an LSTM to generate vector representation from characters for each token.
		Then the character vector is concatenated with token embedding vector, which is input to another LSTM  followed by a CRF layer.
		"""
		def __init__(self, dataset, parameters):

				self.verbose = False

				# Placeholders for input, output and dropout
				self.input_token_indices = tf.placeholder(tf.int32, [None], name="input_token_indices")
				self.input_label_indices_vector = tf.placeholder(tf.float32, [None, dataset.number_of_classes], name="input_label_indices_vector")
				self.input_label_indices_flat = tf.placeholder(tf.int32, [None], name="input_label_indices_flat")
				self.input_token_character_indices = tf.placeholder(tf.int32, [None, None], name="input_token_indices")
				self.input_token_lengths = tf.placeholder(tf.int32, [None], name="input_token_lengths")
				self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

				# Initializer
				initializer = tf.contrib.layers.xavier_initializer()

				
				if parameters['use_character_lstm']:
						# Character embedding layer
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


				# Token embedding layer
				with tf.variable_scope("token_embedding"):
						self.token_embedding_weights = tf.get_variable(
								"token_embedding_weights",
								shape=[dataset.vocabulary_size, parameters['token_embedding_dimension']],
								initializer=initializer,
								trainable=not parameters['freeze_token_embeddings'])
						embedded_tokens = tf.nn.embedding_lookup(self.token_embedding_weights, self.input_token_indices)

				# Concatenate character LSTM outputs and token embeddings
				if parameters['use_character_lstm']:
						with tf.variable_scope("concatenate_token_and_character_vectors"):
								token_lstm_input = tf.concat([character_lstm_output, embedded_tokens], axis=1, name='token_lstm_input')
				else:
						token_lstm_input = embedded_tokens

				# Add dropout
				with tf.variable_scope("dropout"):
						token_lstm_input_drop = tf.nn.dropout(token_lstm_input, self.dropout_keep_prob, name='token_lstm_input_drop')
						if self.verbose: print("token_lstm_input_drop: {0}".format(token_lstm_input_drop))
						token_lstm_input_drop_expanded = tf.expand_dims(token_lstm_input_drop, axis=0, name='token_lstm_input_drop_expanded')
						if self.verbose: print("token_lstm_input_drop_expanded: {0}".format(token_lstm_input_drop_expanded))

				# Token LSTM layer
				with tf.variable_scope('token_lstm') as vs:
						token_lstm_output = bidirectional_LSTM(token_lstm_input_drop_expanded, parameters['token_lstm_hidden_state_dimension'], initializer, output_sequence=True)
						token_lstm_output_squeezed = tf.squeeze(token_lstm_output, axis=0)

				# Needed only if Bidirectional LSTM is used for token level
				with tf.variable_scope("feedforward_after_lstm") as vs:
						W = tf.get_variable(
								"W",
								shape=[2 * parameters['token_lstm_hidden_state_dimension'], parameters['token_lstm_hidden_state_dimension']],
								initializer=initializer)
						b = tf.Variable(tf.constant(0.0, shape=[parameters['token_lstm_hidden_state_dimension']]), name="bias")
						outputs = tf.nn.xw_plus_b(token_lstm_output_squeezed, W, b, name="output_before_tanh")
						outputs = tf.nn.tanh(outputs, name="output_after_tanh")

				with tf.variable_scope("feedforward_before_crf") as vs:
						W = tf.get_variable(
								"W",
								shape=[parameters['token_lstm_hidden_state_dimension'], dataset.number_of_classes],
								initializer=initializer)
						b = tf.Variable(tf.constant(0.0, shape=[dataset.number_of_classes]), name="bias")
						scores = tf.nn.xw_plus_b(outputs, W, b, name="scores")
						self.unary_scores = scores
						self.predictions = tf.argmax(self.unary_scores, 1, name="predictions")

				# CRF layer
				if parameters['use_crf']:
						with tf.variable_scope("crf") as vs:
								# Add start and end tokens
								small_score = -1000.0
								large_score = 0.0
								sequence_length = tf.shape(self.unary_scores)[0]
								unary_scores_with_start_and_end = tf.concat([self.unary_scores, tf.tile( tf.constant(small_score, shape=[1, 2]) , [sequence_length, 1])], 1)
								start_unary_scores = [[small_score] * dataset.number_of_classes + [large_score, small_score]]
								end_unary_scores = [[small_score] * dataset.number_of_classes + [small_score, large_score]]
								self.unary_scores = tf.concat([start_unary_scores, unary_scores_with_start_and_end, end_unary_scores], 0)
								start_index = dataset.number_of_classes
								end_index = dataset.number_of_classes + 1
								input_label_indices_flat_with_start_and_end = tf.concat([ tf.constant(start_index, shape=[1]), self.input_label_indices_flat, tf.constant(end_index, shape=[1]) ], 0)

								# Apply CRF layer
								sequence_length = tf.shape(self.unary_scores)[0]
								sequence_lengths = tf.expand_dims(sequence_length, axis=0, name='sequence_lengths')
								unary_scores_expanded = tf.expand_dims(self.unary_scores, axis=0, name='unary_scores_expanded')
								input_label_indices_flat_batch = tf.expand_dims(input_label_indices_flat_with_start_and_end, axis=0, name='input_label_indices_flat_batch')

								self.transition_parameters=tf.get_variable(
										"transitions",
										shape=[dataset.number_of_classes+2, dataset.number_of_classes+2],
										initializer=initializer)

								log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
										unary_scores_expanded, input_label_indices_flat_batch, sequence_lengths, transition_params=self.transition_parameters)

								self.loss =  tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')
				else:
						# Calculate mean cross-entropy loss
						with tf.variable_scope("loss"):
								losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.unary_scores, labels=self.input_label_indices_vector, name='softmax')
								self.loss =  tf.reduce_mean(losses, name='cross_entropy_mean_loss')
				#training procedure
				self.global_step = tf.Variable(0, name="global_step", trainable=False)
				if parameters['optimizer'] == 'adam':
						self.optimizer = tf.train.AdamOptimizer(parameters['learning_rate'])
				elif parameters['optimizer'] == 'sgd':
						self.optimizer = tf.train.GradientDescentOptimizer(parameters['learning_rate'])
				elif parameters['optimizer'] == 'adadelta':
						self.optimizer = tf.train.AdadeltaOptimizer(parameters['learning_rate'])
				else:
						raise ValueError('The lr_method parameter must be either adadelta, adam or sgd.')

				grads_and_vars = self.optimizer.compute_gradients(self.loss)
				if parameters['gradient_clipping_value']:
						grads_and_vars = [(tf.clip_by_value(grad, -parameters['gradient_clipping_value'], parameters['gradient_clipping_value']), var) 
															for grad, var in grads_and_vars]
				self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

