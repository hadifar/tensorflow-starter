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
import os
import time

import numpy as np
import tensorflow as tf

from lesson7.utils import TextUtils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint_path', '../lesson7/default/', 'checkpoint path')
tf.flags.DEFINE_string('model_name', 'default', 'model name')
tf.flags.DEFINE_integer('n_layers', default=1, help='number of stacked layers')
tf.flags.DEFINE_integer('enc_hid_size', default=128, help='encoder rnn hidden size')
tf.flags.DEFINE_integer('dec_hid_size', default=128, help='decoder rnn hidden size')
tf.flags.DEFINE_integer('nb_examples', default=256, help='number of training example to train model')


class S2SModel(object):

    def __init__(self, src_vocab_size, trg_vocab_size,
                 eng_index_to_word, spa_index_to_word,
                 max_inp_seq, max_trg_seq,
                 enc_hidden_size, dec_hidden_size,
                 n_layers):

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.eng_index_to_word = eng_index_to_word
        self.spa_index_to_word = spa_index_to_word
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.max_inp_seq = max_inp_seq
        self.max_trg_seq = max_trg_seq
        self.nb_layers = n_layers
        self.global_step = tf.get_variable(name='global_step', trainable=False, initializer=0)

    def _create_input(self):
        with tf.name_scope('data'):
            self.enc_inps = tf.placeholder(dtype=tf.int32, shape=(None, None))  # batch_size x enc_seq_len
            self.dec_inps = tf.placeholder(dtype=tf.int32, shape=(None, None))  # batch_size x dec_seq_len
            self.target = tf.placeholder(dtype=tf.int32, shape=(None, None))  # batch_size x target_seq_len

    def _create_model(self):
        with tf.variable_scope('encoder'):
            enc_inp = tf.one_hot(self.enc_inps, self.src_vocab_size)
            enc_rnn_layers = [tf.nn.rnn_cell.GRUCell(self.enc_hidden_size) for _ in range(self.nb_layers)]
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_rnn_layers)
            _, enc_state = tf.nn.dynamic_rnn(cell=enc_cell,
                                             inputs=enc_inp,
                                             dtype=tf.float32)
        with tf.variable_scope('decoder'):
            dec_inp = tf.one_hot(self.dec_inps, self.trg_vocab_size)
            dec_rnn_layers = [tf.nn.rnn_cell.GRUCell(self.dec_hidden_size) for _ in range(self.nb_layers)]
            dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_rnn_layers)
            enc_out, enc_state = tf.nn.dynamic_rnn(cell=dec_cell,
                                                   inputs=dec_inp,
                                                   initial_state=enc_state,
                                                   dtype=tf.float32)

        self.logits = tf.layers.dense(enc_out, self.trg_vocab_size)
        self.pred = tf.nn.softmax(self.logits)

    def _create_loss(self):
        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summary'):
            tf.summary.scalar(name='loss', tensor=self.loss)
            tf.summary.histogram(name='logits', values=self.logits)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_input()
        self._create_model()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, model_path, enc, dec, trg):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if os.path.exists(model_path):
                saver.restore(sess, tf.train.latest_checkpoint(model_path))
                print('model restored!')
            else:
                os.makedirs(model_path)

            writer = tf.summary.FileWriter(model_path, sess.graph)
            while True:
                start = time.time()
                feed_dic = {
                    self.enc_inps: enc,
                    self.dec_inps: dec,
                    self.target: trg,
                }

                batch_loss, _, summary, step = sess.run([self.loss,
                                                         self.optimizer,
                                                         self.summary_op,
                                                         self.global_step],
                                                        feed_dict=feed_dic)

                end = time.time()
                writer.add_summary(summary, step)
                if step % 100 == 0:
                    print('step: {}/{}... '.format(step, 2000),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if step % 250 == 0:
                    saver.save(sess, os.path.join(model_path, 'model.ckpt'), global_step=step)

                if step > 2000:
                    break
        writer.close()

    def inference(self, input_sequences):
        if os.path.isdir(FLAGS.checkpoint_path):
            FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

        enc = np.expand_dims(input_sequences, 0)
        dec = np.zeros([1, self.max_trg_seq])
        dec[0][0] = 1  # <s> token
        i = 1
        decoded_sentence = '<s> '
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, FLAGS.checkpoint_path)
            while True:
                feed_dict = {
                    self.enc_inps: enc,
                    self.dec_inps: dec
                }
                prediction = sess.run([self.pred], feed_dict=feed_dict)[0]
                sampled_token_index = np.argmax(prediction[0, i - 1, :])
                if sampled_token_index == 0:
                    sample_word = ''
                else:
                    sample_word = self.spa_index_to_word[sampled_token_index]
                decoded_sentence += sample_word + ' '

                if sample_word == '</s>' or len(decoded_sentence.split()) > self.max_trg_seq:
                    break

                dec[0][i] = sampled_token_index
                i = i + 1
        print(decoded_sentence)


if __name__ == '__main__':
    text_utils = TextUtils()
    enc_sequence_inps, dec_sequence_inps, dec_sequence_outputs = text_utils.load_data(nb_examples=FLAGS.nb_examples)

    s2s = S2SModel(text_utils.eng_vocab_size,
                   text_utils.spa_vocab_size,
                   text_utils.eng_index_to_word,
                   text_utils.spa_index_to_word,
                   text_utils.max_inp_seq,
                   text_utils.max_trg_seq,
                   enc_hidden_size=FLAGS.enc_hid_size,
                   dec_hidden_size=FLAGS.dec_hid_size,
                   n_layers=FLAGS.n_layers)

    s2s.build_graph()

    s2s.train(FLAGS.model_name,
              enc_sequence_inps,
              dec_sequence_inps,
              dec_sequence_outputs)

    text_utils.print_sentence(enc_sequence_inps[0])
    s2s.inference(enc_sequence_inps[0])
