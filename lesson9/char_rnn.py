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
import codecs
import os
import time

import numpy as np
import tensorflow as tf

from lesson9.utils import TextReader, pick_top_n
from lesson9.utils import batch_generator2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint_path', '../lesson9/default2/model', 'checkpoint path')
tf.flags.DEFINE_string('converter_path', '../lesson9/default2/converter.pkl', 'converter path')
tf.flags.DEFINE_string('name', 'default2', 'the name of the model')
tf.flags.DEFINE_integer('num_seqs', 32, 'number of seqs in batch')
tf.flags.DEFINE_integer('num_seq', 20, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden layer')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.009, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.75,
                      'dropout rate during training process')
tf.flags.DEFINE_string('input_file', '../lesson9/data/test.txt', 'utf-8 encoded input file')
tf.flags.DEFINE_integer('max_steps', 10000, 'max steps of training')
tf.flags.DEFINE_integer('save_model_every', 1000,
                        'save the model every 1000 steps')
tf.flags.DEFINE_integer('log_every', 50, 'log the summaries every 10 steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'the maximum of char number')


class CharRNN(object):
    def __init__(self,
                 num_classes,
                 num_seqs=64,
                 num_seq=50,
                 lstm_size=128,
                 num_layers=2,
                 learning_rate=0.001,
                 grad_clip=5,
                 train_keep_prob=0.5):

        self.num_classes = num_classes
        self.batch_size = num_seqs
        self.num_seq = num_seq
        self.rnn_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def build_graph(self):
        self._build_inputs()
        self._build_rnn()
        self._build_loss()
        self._build_optimizer()

    def _build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(None, None), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(None, None), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def _build_rnn(self):
        with tf.name_scope('RNN'):
            self.rnn_inputs = tf.one_hot(self.inputs, self.num_classes)
            cell = [tf.nn.rnn_cell.GRUCell(num_units=self.rnn_size) for _ in range(self.num_layers)]
            cell = [tf.nn.rnn_cell.DropoutWrapper(cell=c, output_keep_prob=self.keep_prob) for c in cell]
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            self.initial_state = rnn_cell.zero_state(batch_size=tf.shape(self.inputs)[0], dtype=tf.float32)

            self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=self.rnn_inputs, initial_state=self.initial_state)

            seq_output = tf.concat(self.rnn_outputs, axis=1)
            x = tf.reshape(seq_output, shape=[-1, self.rnn_size])

            softmax_w = tf.Variable(initial_value=tf.truncated_normal([self.rnn_size, self.num_classes]))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.nn.xw_plus_b(x, softmax_w, softmax_b)
            self.prediction = tf.nn.softmax(logits=self.logits, name='predictions')

    def _build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, shape=tf.shape(self.logits))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def _build_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

    def train(self, model_path, batch_gen):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            if os.path.exists(model_path):
                saver.restore(sess, tf.train.latest_checkpoint(model_path))
                print('model restored!')
            else:
                os.makedirs(model_path)

            new_state = sess.run(self.initial_state,
                                 feed_dict={self.inputs: np.zeros([self.batch_size, 128], dtype=np.int32)})

            for x, y in batch_gen:
                start = time.time()
                feed_dict = {
                    self.inputs: x,
                    self.targets: y,
                    self.keep_prob: 0.75,
                    self.initial_state: new_state
                }
                _, step, new_state, loss = sess.run(
                    [self.optimizer, self.global_step, self.final_state, self.loss],
                    feed_dict)

                end = time.time()
                current_step = tf.train.global_step(sess, self.global_step)
                if step % 50 == 0:
                    print('step: {}/{}... '.format(step, 3500),
                          'loss: {:.4f}... '.format(loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if current_step % 500 == 0:
                    saver.save(sess,
                               os.path.join(model_path, 'model.ckpt'),
                               global_step=current_step)
                if current_step >= 3500:
                    break

    def inference(self):

        converter = TextReader(filename=FLAGS.converter_path)
        if os.path.isdir(FLAGS.checkpoint_path):
            FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

        start = converter.text_to_arr('')

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, FLAGS.checkpoint_path)
            samples = [c for c in start]
            new_state = sess.run(self.initial_state, feed_dict={self.inputs: np.zeros([1, 1], dtype=np.int32)})
            preds = np.ones((converter.vocab_size,))

            for c in start:
                x = np.zeros((1, 1))
                x[0, 0] = c
                feed_dict = {
                    self.inputs: x,
                    self.keep_prob: 1,
                    self.initial_state: new_state
                }
                preds, new_state = sess.run(
                    [self.prediction, self.final_state],
                    feed_dict=feed_dict)

            c = pick_top_n(preds, converter.vocab_size)

            samples.append(c)

            for i in range(400):
                x = np.zeros((1, 1))
                x[0, 0] = c
                feed_dict = {
                    self.inputs: x,
                    self.keep_prob: 1,
                    self.initial_state: new_state
                }
                preds, new_state = sess.run(
                    [self.prediction, self.final_state],
                    feed_dict=feed_dict)
                c = pick_top_n(preds, converter.vocab_size)
                samples.append(c)

            samples = np.array(samples)
            print(converter.arr_to_text(samples))


def main(_):
    if not os.path.exists(FLAGS.name):
        os.makedirs(FLAGS.name)

    model_path = os.path.join(FLAGS.name, 'model')

    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()

    print('read file')
    Reader = TextReader(text, FLAGS.max_vocab)
    Reader.save_to_file(os.path.join(FLAGS.name, 'converter.pkl'))

    arr = Reader.text_to_arr(text)
    batch_gen = batch_generator2(arr, FLAGS.num_seqs, FLAGS.num_seq)
    print('build model')

    char_rnn = CharRNN(
        num_classes=Reader.vocab_size,
        num_seqs=FLAGS.num_seqs,
        num_seq=FLAGS.num_seq,
        lstm_size=FLAGS.lstm_size,
        num_layers=FLAGS.num_layers,
        learning_rate=FLAGS.learning_rate,
        train_keep_prob=FLAGS.train_keep_prob)

    char_rnn.build_graph()
    char_rnn.train(model_path, batch_gen)
    char_rnn.inference()


if __name__ == '__main__':
    tf.app.run()
