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

tf.reset_default_graph()

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('INPUT_FILE', '../lesson9/data/test.txt', 'input file ')
tf.flags.DEFINE_string('SAVED_FILE', '../lesson9/', 'saved text file directory ')
tf.flags.DEFINE_string('CONVERTER_PATH', '../lesson9/converter.pkl', 'converter path')
tf.flags.DEFINE_integer('BATCH_SIZE', 128, 'default batch size')
tf.flags.DEFINE_integer('SEQ_LEN', 40, 'sequence length ')
tf.flags.DEFINE_integer('NUM_CLASSES', 40, 'unique char (different classes)')  # 48
tf.flags.DEFINE_integer('EMBED_SIZE', 128, 'embedding size')
tf.flags.DEFINE_integer('RNN_SIZE', 128, 'recurrent hidden size')
tf.flags.DEFINE_integer('LAYER_SIZE', 1, 'number of stacked layer in RNN')
tf.flags.DEFINE_float('LEARNING_RATE', 0.002, 'learning rate')


class CharRNN(object):

    def __init__(self, num_classes, emb_size, rnn_size, layer_size, batch_size, seq_len, lr):
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.rnn_size = rnn_size
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lr = lr
        self.global_step = tf.get_variable(name='global_step', initializer=0, trainable=False)

    @staticmethod
    def _gen():
        return utils.batch_generator(input_file=FLAGS.INPUT_FILE,
                                     saved_path=FLAGS.SAVED_FILE,
                                     batch_siz=FLAGS.BATCH_SIZE,
                                     seq_len=FLAGS.SEQ_LEN,
                                     num_classes=FLAGS.NUM_CLASSES)

    def _create_input(self):
        with tf.name_scope('data'):
            x_t, y_t = self._gen()
            train_data = tf.data.Dataset.from_tensor_slices((x_t, y_t))
            train_data = train_data.batch(128)
            train_data = train_data.prefetch(2)

            x = np.random.randint(1, self.num_classes, [1, self.seq_len], dtype=np.int32)
            y = np.random.randint(1, self.num_classes, [1, 1], dtype=np.int32)
            test_data = tf.data.Dataset.from_tensor_slices((x, y))
            test_data = test_data.batch(1)

            self.iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                            train_data.output_shapes)

            self.inp, self.target = self.iterator.get_next()
            self.train_init = self.iterator.make_initializer(train_data)
            self.test_init = self.iterator.make_initializer(test_data)

    def _create_model(self):
        with tf.name_scope('RNN'):
            x = tf.one_hot(self.inp, depth=self.num_classes)

            cell = [tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)] * self.layer_size

            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell)

            self.init_state = rnn_cell.zero_state(tf.shape(self.inp)[0], tf.float32)

            self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(rnn_cell, x,
                                                                   initial_state=self.init_state,
                                                                   dtype=tf.float32)

            output = self.final_state[-1].h
            # x = self.rnn_outputs[:, tf.shape(self.rnn_outputs)[1], :]
            w = tf.Variable(initial_value=tf.truncated_normal(shape=[self.rnn_size, self.num_classes]))
            b = tf.Variable(tf.zeros(self.num_classes))
            self.logits = tf.nn.xw_plus_b(output, w, b)
            self.prediction = tf.nn.softmax(logits=self.logits, name='predictions')

    def _create_loss(self):
        with tf.name_scope('loss'):
            target_new = tf.reshape(self.target, [tf.shape(self.target)[0]])
            y_one_hot = tf.one_hot(target_new, self.num_classes)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=self.logits))

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
            sess.run(tf.global_variables_initializer())
            sess.run(self.train_init)

            total_loss = 0
            writer = tf.summary.FileWriter('graphs/language_model/lr_' + str(self.lr), sess.graph)
            for i in range(training_step):

                try:
                    _, batch_loss, step, summary = sess.run(
                        [self.optimizer, self.loss, self.global_step, self.summary_op])
                    writer.add_summary(summary, global_step=step)
                    total_loss += batch_loss
                    if (i + 1) % 200 == 0:
                        print('Average loss at step {}: {:5.10f}'.format(i + 1, total_loss / 200))
                        total_loss = 0.0
                        self.inference()

                except tf.errors.OutOfRangeError:
                    sess.run(self.train_init)

            writer.close()

    def inference(self):

        converter = utils.TextReader(filename=FLAGS.CONVERTER_PATH)
        rand_sentence = 'به نام خداوند جان و خرد'
        y = np.zeros([1, 1])
        input_eval = converter.text_to_arr(rand_sentence)
        input_eval = np.pad(input_eval, (self.seq_len - len(input_eval), 0), mode='constant')
        input_eval = np.expand_dims(input_eval, 0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.test_init)

            samples = []
            new_state = sess.run(self.init_state)
            for i in range(400):
                feed_dict = {
                    self.inp: input_eval,
                    self.target: y,
                    self.init_state: new_state
                }
                preds, new_state = sess.run([self.prediction, self.final_state], feed_dict=feed_dict)
                c = utils.sample(preds[0])
                samples.append(c)
                input_eval = np.roll(input_eval, shift=-1)
                input_eval[0][-1] = c

            samples = np.array(samples)
            print(rand_sentence)
            print(converter.arr_to_text(samples))

    def build_graph(self):
        self._create_input()
        self._create_model()
        self._create_loss()
        self._create_optimizer()
        self._create_summery()


def main(_):
    charnn = CharRNN(FLAGS.NUM_CLASSES,
                     FLAGS.EMBED_SIZE, FLAGS.RNN_SIZE,
                     FLAGS.LAYER_SIZE, FLAGS.BATCH_SIZE,
                     FLAGS.SEQ_LEN, FLAGS.LEARNING_RATE)

    charnn.build_graph()
    charnn.train(10000)


if __name__ == '__main__':
    tf.app.run()
