
"""Builds parser models."""

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops as cf
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging

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

def BatchedSparseToDense(sparse_indices, output_size):
  """Batch compatible sparse to dense conversion.

  This is useful for one-hot coded target labels.

  Args:
    sparse_indices: [batch_size] tensor containing one index per batch
    output_size: needed in order to generate the correct dense output

  Returns:
    A [batch_size, output_size] dense tensor.
  """
  eye = tf.diag(tf.fill([output_size], tf.constant(1, tf.float32)))
  return tf.nn.embedding_lookup(eye, sparse_indices)


def EmbeddingLookupFeatures(params, sparse_features, allow_weights):
  """Computes embeddings for each entry of sparse features sparse_features.

  Args:
    params: list of 2D tensors containing vector embeddings
    sparse_features: 1D tensor of strings. Each entry is a string encoding of
      dist_belief.SparseFeatures, and represents a variable length list of
      feature ids, and optionally, corresponding weights values.
    allow_weights: boolean to control whether the weights returned from the
      SparseFeatures are used to multiply the embeddings.

  Returns:
    A tensor representing the combined embeddings for the sparse features.
    For each entry s in sparse_features, the function looks up the embeddings
    for each id and sums them into a single tensor weighing them by the
    weight of each id. It returns a tensor with each entry of sparse_features
    replaced by this combined embedding.
  """
  if not isinstance(params, list):
    params = [params]
  # Lookup embeddings.
  st = tf.string_split(sparse_features, delimiter=',')
  sparse_features = tf.string_to_number(st.values, out_type=tf.int32)
  embeddings = tf.nn.embedding_lookup(params, sparse_features)
  return embeddings


class GreedyTagger(object):
  """Builds a Chen & Manning style greedy neural net tagger

  Builds a graph with an optional reader op connected at one end and
  operations needed to train the network on the other. Supports multiple
  network instantiations sharing the same parameters and network topology.

  The following named nodes are added to the training and eval networks:
    epochs: a tensor containing the current epoch number
    cost: a tensor containing the current training step cost
    gold_actions: a tensor containing actions from gold decoding
    feature_endpoints: a list of sparse feature vectors
    logits: output of the final layer before computing softmax
  The training network also contains:
    train_op: an op that executes a single training step
  """

  def __init__(self,
               num_actions,
               num_features,
               num_feature_ids,
               embedding_sizes,
               hidden_layer_sizes,
               seed=None,
               gate_gradients=False,
               use_locking=False,
               embedding_init=1.0,
               relu_init=1e-4,
               bias_init=0.2,
               softmax_init=1e-4,
               averaging_decay=0.9999,
               use_averaging=True,
               check_parameters=True,
               check_every=1,
               allow_feature_weights=False,
               only_train='',
               arg_prefix=None,
               **unused_kwargs):
    """Initialize the graph builder with parameters defining the network.

    Args:
      num_actions: int size of the set of parser actions
      num_features: int list of dimensions of the feature vectors
      num_feature_ids: int list of same length as num_features corresponding to
        the sizes of the input feature spaces
      embedding_sizes: int list of same length as num_features of the desired
        embedding layer sizes
      hidden_layer_sizes: int list of desired relu layer sizes; may be empty
      seed: optional random initializer seed to enable reproducibility
      gate_gradients: if True, gradient updates are computed synchronously,
        ensuring consistency and reproducibility
      use_locking: if True, use locking to avoid read-write contention when
        updating Variables
      embedding_init: sets the std dev of normal initializer of embeddings to
        embedding_init / embedding_size ** .5
      relu_init: sets the std dev of normal initializer of relu weights
        to relu_init
      bias_init: sets constant initializer of relu bias to bias_init
      softmax_init: sets the std dev of normal initializer of softmax init
        to softmax_init
      averaging_decay: decay for exponential moving average when computing
        averaged parameters, set to 1 to do vanilla averaging
      use_averaging: whether to use moving averages of parameters during evals
      check_parameters: whether to check for NaN/Inf parameters during
        training
      check_every: checks numerics every check_every steps.
      allow_feature_weights: whether feature weights are allowed.
      only_train: the comma separated set of parameter names to train. If empty,
        all model parameters will be trained.
      arg_prefix: prefix for context parameters.
    """
    self._num_actions = num_actions
    self._num_features = num_features
    self._num_feature_ids = num_feature_ids
    self._embedding_sizes = embedding_sizes
    self._hidden_layer_sizes = hidden_layer_sizes
    self._seed = seed
    self._gate_gradients = gate_gradients
    self._use_locking = use_locking
    self._use_averaging = use_averaging
    self._check_parameters = check_parameters
    self._check_every = check_every
    self._allow_feature_weights = allow_feature_weights
    self._only_train = set(only_train.split(',')) if only_train else None
    self._feature_size = len(embedding_sizes)
    self._embedding_init = embedding_init
    self._relu_init = relu_init
    self._softmax_init = softmax_init
    self._arg_prefix = arg_prefix
    # Parameters of the network with respect to which training is done.
    self.params = {}
    # Other variables, with respect to which no training is done, but which we
    # nonetheless need to save in order to capture the state of the graph.
    self.variables = {}
    # Operations to initialize any nodes that require initialization.
    self.inits = {}
    # Training- and eval-related nodes.
    self.training = {}
    self.evaluation = {}
    self.saver = None
    # Nodes to compute moving averages of parameters, called every train step.
    self._averaging = {}
    self._averaging_decay = averaging_decay

    # After the following 'with' statement, we'll be able to re-enter the
    # 'params' scope by re-using the self._param_scope member variable. See for
    # instance _AddParam.
    self.input = tf.placeholder(dtype=tf.string)
    self.labels = tf.placeholder(dtype=tf.int32)
    self.dropout = tf.placeholder(tf.float32)
    self.input_type_indices = tf.placeholder(tf.int32, [None], name="input_type_indices")
    self.input_mention_length = tf.placeholder(tf.int32, [None], name="input_mention_length")
    self.input_mention_indices = tf.placeholder(tf.int32, [None, None], name="input_mention_indices")
    with tf.name_scope('params') as self._param_scope:
      self._relu_bias_init = tf.constant_initializer(bias_init)
    self.training.update(self._BuildNetwork(self.input,
                                      return_average=False))

  @property
  def embedding_size(self):
    size = 0
    for i in range(self._feature_size):
      size += self._num_features[i] * self._embedding_sizes[i]
    return size

  def _AddParam(self,
                shape,
                dtype,
                name,
                initializer=None,
                return_average=False):
    """Add a model parameter w.r.t. we expect to compute gradients.

    _AddParam creates both regular parameters (usually for training) and
    averaged nodes (usually for inference). It returns one or the other based
    on the 'return_average' arg.

    Args:
      shape: int list, tensor shape of the parameter to create
      dtype: tf.DataType, data type of the parameter
      name: string, name of the parameter in the TF graph
      initializer: optional initializer for the paramter
      return_average: if False, return parameter otherwise return moving average

    Returns:
      parameter or averaged parameter
    """
    if name not in self.params:
      step = tf.cast(self.GetStep(), tf.float32)
      # Put all parameters and their initializing ops in their own scope
      # irrespective of the current scope (training or eval).
      with tf.name_scope(self._param_scope):
        self.params[name] = tf.get_variable(name, shape, dtype, initializer)
        param = self.params[name]
        if initializer is not None:
          self.inits[name] = state_ops.init_variable(param, initializer)
        if self._averaging_decay == 1:
          logging.info('Using vanilla averaging of parameters.')
          ema = tf.train.ExponentialMovingAverage(decay=(step / (step + 1.0)),
                                                  num_updates=None)
        else:
          ema = tf.train.ExponentialMovingAverage(decay=self._averaging_decay,
                                                  num_updates=step)
        self._averaging[name + '_avg_update'] = ema.apply([param])
        self.variables[name + '_avg_var'] = ema.average(param)
        self.inits[name + '_avg_init'] = state_ops.init_variable(
            ema.average(param), tf.constant_initializer(0.0))
    return (self.variables[name + '_avg_var'] if return_average else
            self.params[name])

  def GetStep(self):
    def OnesInitializer(shape, dtype=tf.float32, partition_info=None):
      return tf.ones(shape, dtype)
    return self._AddVariable([], tf.int32, 'step', OnesInitializer)

  def _AddVariable(self, shape, dtype, name, initializer=None):
    if name in self.variables:
      return self.variables[name]
    self.variables[name] = tf.get_variable(name, shape, dtype, initializer)
    if initializer is not None:
      self.inits[name] = state_ops.init_variable(self.variables[name],
                                                 initializer)
    return self.variables[name]

  def _ReluWeightInitializer(self):
    with tf.name_scope(self._param_scope):
      return tf.random_normal_initializer(stddev=self._relu_init,
                                          seed=self._seed)

  def _EmbeddingMatrixInitializer(self, index, embedding_size):
    return tf.random_normal_initializer(
          stddev=self._embedding_init / embedding_size**.5,
          seed=self._seed)

  def _AddEmbedding(self,
                    features,
                    num_features,
                    num_ids,
                    embedding_size,
                    index,
                    return_average=False):
    """Adds an embedding matrix and passes the `features` vector through it."""
    embedding_matrix = self._AddParam(
        [num_ids, embedding_size],
        tf.float32,
        'embedding_matrix_%d' % index,
        self._EmbeddingMatrixInitializer(index, embedding_size),
        return_average=return_average)
    embedding = EmbeddingLookupFeatures(embedding_matrix,
                                        tf.reshape(features,
                                                   [-1],
                                                   name='feature_%d' % index),
                                        self._allow_feature_weights)
    return tf.reshape(embedding, [-1, num_features * embedding_size])

  def _BuildNetwork(self, feature_endpoints, return_average=False):
    """Builds a feed-forward part of the net given features as input.

    The network topology is already defined in the constructor, so multiple
    calls to BuildForward build multiple networks whose parameters are all
    shared. It is the source of the input features and the use of the output
    that distinguishes each network.

    Args:
      feature_endpoints: tensors with input features to the network
      return_average: whether to use moving averages as model parameters

    Returns:
      logits: output of the final layer before computing softmax
    """
    # Create embedding layer.
    embeddings = []
    for i in range(self._feature_size):
      embeddings.append(self._AddEmbedding(feature_endpoints[i],
                                           self._num_features[i],
                                           self._num_feature_ids[i],
                                           self._embedding_sizes[i],
                                           i,
                                           return_average=return_average))
    last_layer = tf.concat(embeddings, 1)
    last_layer =  tf.nn.dropout(last_layer, self.dropout)
    last_layer_size = self.embedding_size

    # Create ReLU layers.
    for i, hidden_layer_size in enumerate(self._hidden_layer_sizes):
      weights = self._AddParam(
          [last_layer_size, hidden_layer_size],
          tf.float32,
          'weights_%d' % i,
          self._ReluWeightInitializer(),
          return_average=return_average)
      bias = self._AddParam([hidden_layer_size],
                            tf.float32,
                            'bias_%d' % i,
                            self._relu_bias_init,
                            return_average=return_average)
      last_layer = tf.nn.relu_layer(last_layer,
                                    weights,
                                    bias,
                                    name='layer_%d' % i)
      last_layer_size = hidden_layer_size

    # Create softmax layer.
    softmax_weight = self._AddParam(
        [last_layer_size, self._num_actions],
        tf.float32,
        'softmax_weight',
        tf.random_normal_initializer(stddev=self._softmax_init,
                                     seed=self._seed),
        return_average=return_average)
    softmax_bias = self._AddParam(
        [self._num_actions],
        tf.float32,
        'softmax_bias',
        tf.constant_initializer(0.0),
        return_average=return_average)
    logits = tf.nn.xw_plus_b(last_layer,
                             softmax_weight,
                             softmax_bias,
                             name='logits')
    predictions = tf.argmax(logits, 1, name="predictions")

    # Create CRF layer.
    small_score = -1000.0
    large_score = 0.0
    sequence_length = tf.shape(logits)[0]
    unary_scores_with_start_and_end = tf.concat([logits, tf.tile( tf.constant(small_score, shape=[1, 2]) , [sequence_length, 1])], 1)
    start_unary_scores = [[small_score] * self._num_actions + [large_score, small_score]]
    end_unary_scores = [[small_score] * self._num_actions + [small_score, large_score]]
    unary_scores = tf.concat([start_unary_scores, unary_scores_with_start_and_end, end_unary_scores], 0)
    start_index = self._num_actions
    end_index = self._num_actions + 1
    input_label_indices_flat_with_start_and_end = tf.concat([ tf.constant(start_index, shape=[1]), self.labels, tf.constant(end_index, shape=[1]) ], 0)
    sequence_lengths = tf.expand_dims(sequence_length, axis=0, name='sequence_lengths')
    unary_scores_expanded = tf.expand_dims(unary_scores, axis=0, name='unary_scores_expanded')
    input_label_indices_flat_batch = tf.expand_dims(input_label_indices_flat_with_start_and_end, axis=0, name='input_label_indices_flat_batch')
    
    transition_parameters = self._AddParam(
        [self._num_actions+2, self._num_actions+2],
        tf.float32,
        'trainable_params',
        tf.contrib.layers.xavier_initializer(),
        return_average=return_average
        )
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    unary_scores_expanded, input_label_indices_flat_batch, sequence_lengths, transition_params=transition_parameters)
    boundry_loss =  tf.reduce_mean(-log_likelihood, name='cost')

    embedded_mentions = tf.nn.embedding_lookup(last_layer, self.input_mention_indices, name='embedded_mentions')
    mention_lstm_output = bidirectional_LSTM(embedded_mentions, 128, tf.contrib.layers.xavier_initializer(),
                            sequence_length=self.input_mention_length, output_sequence=False)
    W = tf.get_variable(
      "W",
      shape=[256, 5],
      initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.0, shape=[4]), name="bias")
    self.type_scores = tf.nn.xw_plus_b(mention_lstm_output, W, b, name="scores")
    self.type_predictions = tf.argmax(self.type_scores, 1, name="predictions")
    type_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_type_indices, logits=self.type_scores))    
    loss_sum = tf.add(type_loss, boundry_loss)
    # Add the optimizer
    trainable_params = self.params.values()
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train_op_sum = optimizer.minimize(loss_sum, var_list=trainable_params)

    return {'predictions': predictions, 
            'unary_scores': unary_scores,
            'cost_sum': loss_sum, 
            'cost_boundry': boundry_loss, 
            'train_op_boundry': train_op_boundry, 
            'train_op_sum': train_op_sum, 
            'transition_parameters': transition_parameters}

