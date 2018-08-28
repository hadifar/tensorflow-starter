# -*- coding: utf-8 -*-
#
# Copyright 2018 Amir Hadifar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import tensorflow as tf

from lesson9 import utils

# from tensorflow.python import debug as tf_debug

tf.reset_default_graph()

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('INPUT_FILE', '../lesson9/text.txt', 'input file ')
tf.flags.DEFINE_string('SAVED_FILE', '../lesson9/', 'saved text file directory ')
tf.flags.DEFINE_string('CONVERTER_PATH', '../lesson9/converter.pkl', 'converter path')
tf.flags.DEFINE_integer('BATCH_SIZE', 32, 'default batch size')
tf.flags.DEFINE_integer('SEQ_LEN', 20, 'sequence length ')
tf.flags.DEFINE_integer('CHAR_SIZE', 40, 'unique char')
tf.flags.DEFINE_integer('EMBED_SIZE', 50, 'embedding size')
tf.flags.DEFINE_integer('RNN_SIZE', 50, 'recurrent hidden size')
tf.flags.DEFINE_integer('LAYER_SIZE', 2, 'number of stacked layer in RNN')
tf.flags.DEFINE_float('LEARNING_RATE', 0.001, 'learning rate')


class CharRNN(object):

    def __init__(self, dataset, num_classes, emb_size, rnn_size, layer_size, batch_size, seq_len, lr):
        self.dataset = dataset
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.rnn_size = rnn_size
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lr = lr
        self.global_step = tf.get_variable(name='global_step', initializer=0, trainable=False)

    def _create_input(self):
        with tf.name_scope('data'):
            self.iterator = dataset.make_initializable_iterator()
            self.inp, self.target = self.iterator.get_next()

    def _create_model(self):
        self._create_embedding()
        self._create_rnn()

    def _create_embedding(self):
        with tf.name_scope('embedding'):
            embed_matrix = tf.get_variable(name='embedding',
                                           initializer=tf.truncated_normal([self.num_classes, self.emb_size]))
            self.embed = tf.nn.embedding_lookup(embed_matrix, self.inp)

    def _create_rnn(self):
        with tf.name_scope('RNN'):
            cell = [tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)] * self.layer_size
            # tf.nn.rnn_cell.DropoutWrapper(rnn_cell,)
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            print(self.inp.shape)
            self.init_state = rnn_cell.zero_state(tf.shape(self.inp)[0], tf.float32)

            self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(rnn_cell, self.embed,
                                                                   initial_state=self.init_state,
                                                                   dtype=tf.float32)
            seq_output = tf.concat(self.rnn_outputs, axis=1)
            x = tf.reshape(seq_output, shape=[-1, self.rnn_size])

            w = tf.Variable(initial_value=tf.truncated_normal(shape=[self.rnn_size, self.num_classes]))
            b = tf.Variable(tf.zeros(self.num_classes))
            self.logits = tf.nn.xw_plus_b(x, w, b)
            self.prediction = tf.nn.softmax(logits=self.logits, name='predictions')

    def _create_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.target, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, shape=tf.shape(self.logits))
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_reshaped, logits=self.logits))

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def _create_summery(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('lstm_output', self.rnn_outputs)
            self.summary_op = tf.summary.merge_all()

    def train(self, training_step):

        with tf.Session() as sess:
            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6007")
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer)

            total_loss = 0
            writer = tf.summary.FileWriter('graphs/language_model/lr_' + str(self.lr), sess.graph)
            for i in range(training_step):

                try:
                    _, batch_loss, step, summary = sess.run(
                        [self.optimizer, self.loss, self.global_step, self.summary_op])
                    writer.add_summary(summary, global_step=step)
                    total_loss += batch_loss
                    if (i + 1) % 200 == 0:
                        print('Average loss at step {}: {:5.1f}'.format(i + 1, total_loss / 200))
                        total_loss = 0.0

                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)

            writer.close()

    def inference(self):
        converter = utils.TextReader(filename=FLAGS.CONVERTER_PATH)
        start = converter.text_to_arr('')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            samples = [c for c in start]
            new_state = sess.run(self.init_state)
            preds = np.ones((converter.vocab_size,))
            c = utils.pick_top_n(preds, converter.vocab_size)
            samples.append(c)

            for i in range(1000):
                x = np.zeros((1, 1))
                x[0, 0] = c
                feed_dict = {
                    self.inp: x,
                    self.init_state: new_state
                }
                preds, new_state = sess.run([self.prediction, self.final_state], feed_dict=feed_dict)
                c = utils.pick_top_n(preds, converter.vocab_size)
                samples.append(c)

            samples = np.array(samples)
            print(converter.arr_to_text(samples))

    def build_graph(self):
        self._create_input()
        self._create_model()
        self._create_loss()
        self._create_optimizer()
        self._create_summery()


if __name__ == '__main__':
    def gen():
        return utils.batch_generator(input_file=FLAGS.INPUT_FILE,
                                     saved_path=FLAGS.SAVED_FILE,
                                     batch_siz=FLAGS.BATCH_SIZE,
                                     seq_len=FLAGS.SEQ_LEN,
                                     unique_char=FLAGS.CHAR_SIZE)


    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32),
                                             (tf.TensorShape([None, None]),
                                              tf.TensorShape([None, None])))
    dataset = dataset.prefetch(2)

    charnn = CharRNN(dataset, FLAGS.CHAR_SIZE,
                     FLAGS.EMBED_SIZE, FLAGS.RNN_SIZE,
                     FLAGS.LAYER_SIZE, FLAGS.BATCH_SIZE,
                     FLAGS.SEQ_LEN, FLAGS.LEARNING_RATE)

    charnn.build_graph()
    charnn.train(1)
    charnn.inference()

    # train_data = tf.data.Dataset.from_tensor_slices(train)
    # train_data = train_data.shuffle(10000)  # if you want to shuffle your data
    # train_data = train_data.batch(batch_size)
    #
    # # create testing Dataset and batch it
    # test_data = tf.data.Dataset.from_tensor_slices(test)
    # test_data = test_data.shuffle(10000)
    # test_data = test_data.batch(batch_size)
    #
    # # create one iterator and initialize it with different datasets
    # iterator = tf.data.Iterator.from_structure(train_data.output_types,
    #                                            train_data.output_shapes)
    # img, label = iterator.get_next()
    #
    # train_init = iterator.make_initializer(train_data)  # initializer for train_data
    # test_init = iterator.make_initializer(test_data)  # initializer for train_data
