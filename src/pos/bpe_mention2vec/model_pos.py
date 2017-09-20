import numpy as np
import os
import tensorflow as tf
from data_utils import minibatches, pad_sequences, get_chunks, find_mentions, find_labels, merge_labels
from general_utils import Progbar, print_sentence
import time
import codecs


class POSmodel(object):
    def __init__(self, config, embeddings, ntags, nchars=None, niob=None, ntype=None):
        """
        Args:
            config: class with hyper parameters
            embeddings: np array with embeddings
            nchars: (int) size of chars vocabulary
        """
        self.config     = config
        self.embeddings = embeddings
        self.nchars     = nchars
        self.ntags      = ntags
        self.logger     = config.logger # now instantiated in config
        self.niob       = niob
        self.ntype      = ntype


    def add_placeholders(self):
        """
        Adds placeholders to self
        """

        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")

        self.iob_type = tf.placeholder(tf.int32, shape=[None, None])
        self.mention_type = tf.placeholder(tf.int32, shape=[None, None])
        self.mention = tf.placeholder(dtype=tf.int32, shape=[None, None, None])
        self.mention_length = tf.placeholder(tf.int32, shape=[None, None])
        self.mention_size = tf.placeholder(tf.int32, shape=[None])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], 
                        name="lr")


    def get_feed_dict(self, words, word_features, lr=None, dropout=None, iob=None, mention_type=None, mentions=None):
        """
        Given some data, pad it and build a feed dictionary
        Args:
            words: list of sentences. A sentence is a list of ids of a list of words. 
                A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        # perform padding of the given data
        word_ids, sequence_lengths = pad_sequences(word_features, [0,0,0,0,0,0,0,0])

        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.chars:
            char_ids, _ = zip(*words)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
        
        mention_size = []
    
        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout
        if iob is not None:
            feed[self.iob_type], _ = pad_sequences(iob, 0)
        mention_num = 0
        if mention_type is not None:
            feed[self.mention_type], mention_size = pad_sequences(mention_type, 0)
            feed[self.mention_size] = mention_size
        if mentions is not None:
            feed[self.mention], mention_length = pad_sequences(mentions, pad_tok=0, nlevels=2)
            feed[self.mention_length] = mention_length
            

        return feed, sequence_lengths, mention_size


    def add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32, 
                                trainable=self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, 
                name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.chars:
                # get embeddings matrix
                _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32, 
                    shape=[self.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, 
                    name="char_embeddings")
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[-1])
                # bi lstm on chars
                # need 2 instances of cells since tf 1.1
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, 
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, 
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    state_is_tuple=True)

                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                    cell_bw, char_embeddings, sequence_length=word_lengths, 
                    dtype=tf.float32)

                output = tf.concat([output_fw, output_bw], axis=-1)
                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[-1, 2*self.config.char_hidden_size])

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)
        s = tf.shape(self.word_embeddings)
        self.word_embeddings = tf.reshape(self.word_embeddings, shape=[-1, 800])
        if self.config.chars:
            self.word_embeddings = tf.concat([self.word_embeddings, output], axis=-1)

    def add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("ff"):
            if self.config.chars:
                input_size = 8*self.config.dim+2*self.config.dim_char
            else:
                input_size = 8*self.config.dim
            weights = tf.get_variable("W", shape=[input_size, 2*self.config.hidden_size], 
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable("b", shape=[2*self.config.hidden_size], dtype=tf.float32, 
                initializer=tf.zeros_initializer())
            last_layer = tf.nn.relu_layer(self.word_embeddings,
                                    weights,
                                    bias)
            self.hidden_embeddings = tf.reshape(last_layer, shape=[self.config.batch_size,-1,2*self.config.hidden_size] )       

        with tf.variable_scope("type"):
            to_concat = []
            for i in range(self.config.batch_size):
                embeddings = tf.nn.embedding_lookup(self.hidden_embeddings, i)
                lookup = tf.nn.embedding_lookup(self.mention, i)
                mention_embeddings = tf.nn.embedding_lookup(embeddings, lookup)
                to_concat.append(tf.expand_dims(mention_embeddings, axis=0))
            mention_embeddings = tf.concat(to_concat, 0)
            s = tf.shape(mention_embeddings)
            
            mention_embeddings = tf.reshape(mention_embeddings, shape=[-1, s[-2], 2*self.config.hidden_size])
            lengths = tf.reshape(self.mention_length, shape=[-1])
            cell_fw1 = tf.contrib.rnn.LSTMCell(2*self.config.hidden_size, initializer=tf.contrib.layers.xavier_initializer())
            cell_bw1 = tf.contrib.rnn.LSTMCell(2*self.config.hidden_size, initializer=tf.contrib.layers.xavier_initializer())
            _, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, mention_embeddings, sequence_length=lengths, dtype=tf.float32)
            final_states_forward, final_states_backward = final_states
            type_output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=1, name='output')
            W1 = tf.get_variable("W1", shape=[4*self.config.hidden_size, self.ntype], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", shape=[self.ntype], dtype=tf.float32, initializer=tf.zeros_initializer())
            type_scores = tf.matmul(type_output, W1) + b1
            #self.r = tf.shape(type_scores)
            self.type_logits = tf.reshape(type_scores, [s[0], s[1], self.ntype])
            
    def add_pred_op(self):
        """
        Adds labels_pred to self
        """
        self.type_pred = tf.cast(tf.argmax(self.type_logits, axis=-1), tf.int32)


    def add_loss_op(self):
        """
        Adds loss to self
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.type_logits, labels=self.mention_type)
        mask = tf.sequence_mask(self.mention_size)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)
        
        #tf.summary.scalar("loss", self.loss)


    def add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            # sgd method
            if self.config.lr_method == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.lr_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif self.config.lr_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif self.config.lr_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise NotImplementedError("Unknown train op {}".format(
                                          self.config.lr_method))

            # gradient clipping if config.clip is positive
            if self.config.clip > 0:
                gradients, variables   = zip(*optimizer.compute_gradients(self.loss))
                gradients, global_norm = tf.clip_by_global_norm(gradients, self.config.clip)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def add_init_op(self):
        self.init = tf.global_variables_initializer()


    def add_summary(self, sess): 
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)


    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()

    def predict_type_batch(self, sess, words, word_features, mentions):
        fd, _, mention_sizes = self.get_feed_dict(words, word_features, mentions=mentions, dropout=1.0)
        type_pred = sess.run(self.type_pred, feed_dict=fd)

        return type_pred, mention_sizes


    def run_epoch(self, sess, train, dev, tags, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        Args:
            sess: tensorflow session
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
            epoch: (int) number of the epoch
        """
        nbatches = (len(train) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=nbatches)
        total_loss = 0.0
        count = 0
        for i, (words, labels, iob, mention_type, mentions, word_features) in enumerate(minibatches(train, self.config.batch_size)):
            fd, _, _ = self.get_feed_dict(words, word_features, self.config.lr, self.config.dropout, iob, mention_type, mentions)
            _, train_loss= sess.run([self.train_op, self.loss], feed_dict=fd)
            total_loss += train_loss
            count += 1
        print total_loss/count

        acc, f1 = self.run_evaluate(sess, dev, tags)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100*acc, 100*f1))
        return acc, f1


    def run_evaluate(self, sess, test, tags):
        """
        Evaluates performance on test set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
            tags: {tag: index} dictionary
        Returns:
            accuracy
            f1 score
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        output_file = codecs.open("output", 'w', 'UTF-8')
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        for words, labels, iob_gold, mention_type_gold, mentions_gold, word_features in minibatches(test, self.config.batch_size):
            mentions = []
            mention_sizes = []
            count = 0
            for i in range(self.config.batch_size):
                mention = find_mentions(iob_gold[i])
                mentions.append(mention)
                mention_sizes.append(len(mention))
            
            mentions_pred, _ = self.predict_type_batch(sess, words, word_features, mentions)
            
            for ws, iob_pred, mention, mention_pred, mention_size in zip(words, iob_gold, mentions, mentions_pred, mention_sizes):
                mention_pred = mention_pred[:mention_size]
                lab_pred = merge_labels(iob_pred, mention_pred, idx_to_tag)
                output_string = ""
                for l in lab_pred:
                    output_string += l + '\n'
                output_file.write(output_string+'\n')
        
        return 0.0, 0.0


    def train(self, train, dev, tags):
        """
        Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
        """
        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0
        with tf.Session() as sess:
            sess.run(self.init)
            if self.config.reload:
                self.logger.info("Reloading the latest trained model...")
                saver.restore(sess, self.config.model_output)
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

                acc, f1 = self.run_epoch(sess, train, dev, tags, epoch)

                # decay learning rate
                self.config.lr *= self.config.lr_decay

                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                                        nepoch_no_imprv))
                        break


    def evaluate(self, test, tags):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess, self.config.model_output)
            t = time.time()
            acc, f1 = self.run_evaluate(sess, test, tags)
            print time.time()- t
            self.logger.info("- test acc {:04.2f} - f1 {:04.2f}".format(100*acc, 100*f1))



