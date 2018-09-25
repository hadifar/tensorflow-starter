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

import nltk
import numpy as np
import tensorflow as tf


class S2SModel(object):
    def __init__(self, src_vocab_size, trg_vocab_size, enc_hidden_size, dec_hidden_size, n_layers):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.nb_layers = n_layers
        self.global_step = tf.get_variable(name='global_step', trainable=False, initializer=0)

    def _create_input(self):
        with tf.name_scope('data'):
            self.enc_inps = tf.placeholder(dtype=tf.int32, shape=(None, None))  # batch_size x enc_seq_len
            self.dec_inps = tf.placeholder(dtype=tf.int32, shape=(None, None))  # batch_size x dec_seq_len
            self.target = tf.placeholder(dtype=tf.int32, shape=(None, None, None))  # batch_size x target_seq_len

    def _create_model(self):
        with tf.name_scope('s2s'):
            enc_inp = tf.one_hot(self.enc_inps, self.src_vocab_size)
            dec_inp = tf.one_hot(self.dec_inps, self.trg_vocab_size)

            enc_rnn_layers = [tf.nn.rnn_cell.GRUCell(self.enc_hidden_size) for _ in range(self.nb_layers)]
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_rnn_layers)
            _, enc_state = tf.nn.dynamic_rnn(cell=enc_cell, inputs=enc_inp, dtype=tf.float32, scope='encoder')

            dec_rnn_layers = [tf.nn.rnn_cell.GRUCell(self.dec_hidden_size) for _ in range(self.nb_layers)]
            dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_rnn_layers)
            enc_out, self.enc_state = tf.nn.dynamic_rnn(cell=dec_cell, inputs=dec_inp, initial_state=enc_state,
                                                        dtype=tf.float32, scope='decoder')

            self.logits = tf.layers.dense(enc_out, self.trg_vocab_size)
            self.pred = tf.nn.softmax(self.logits)

    def _create_loss(self):
        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target, logits=self.logits)
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

    def train(self, enc, dec, trg):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if os.path.exists('graphs/'):
                saver.restore(sess, tf.train.latest_checkpoint('graphs/'))
                print('model restored!')
            else:
                os.makedirs('graphs/')

            writer = tf.summary.FileWriter('graphs/', sess.graph)
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
                    saver.save(sess,
                               os.path.join('graph/', 'model.ckpt'), global_step=step)

                if step > 2000:
                    break
        writer.close()

    def inference(self, input_sequences):

        for enc in input_sequences:
            print('--'*50)
            trans = ''
            for t in enc:
                if t !=0:
                    trans += eng_index_to_word[t]
            print(trans)

            enc = np.expand_dims(enc, 0)
            dec = np.zeros([1, max_trg_seq])
            dec[0][0] = 1  # <s> token
            i = 1
            decoded_sentence = '<s> '
            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(sess, '../lesson7/graphs/model.ckpt-1000')
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
                        sample_word = spa_index_to_word[sampled_token_index]
                    decoded_sentence += sample_word + ' '

                    if sample_word == '</s>' or len(decoded_sentence.split()) > max_trg_seq:
                        break

                    dec[0][i] = sampled_token_index
                    i = i + 1
            print(decoded_sentence)


if __name__ == '__main__':

    nb_examples = 256
    nb_epochs = 100

    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip',
        extract=True)

    data_path = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

    input_texts = []
    target_texts = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines[:nb_examples]:
        inp_txt, trg_txt = line.split('\t')
        inp_txt = nltk.word_tokenize(inp_txt)
        trg_txt = nltk.word_tokenize(trg_txt)

        input_texts.append('<s>' + ' ' + ' '.join(inp_txt) + ' ' + '</s>')
        target_texts.append('<s>' + ' ' + ' '.join(trg_txt) + ' ' + '</s>')

    print('number of training examples for language1: ', len(input_texts))
    print('number of training examples for language2: ', len(target_texts))


    def get_tokenizer(text):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(text)
        return tokenizer


    # define two tokenizer for both languages with helper function (get_tokenizer())
    eng_tokenizer = get_tokenizer(input_texts)
    spa_tokenizer = get_tokenizer(target_texts)

    # convert each sentence to sequence of integers
    enc_sequence_inps = eng_tokenizer.texts_to_sequences(input_texts)
    dec_sequence_inps = spa_tokenizer.texts_to_sequences(target_texts)

    # find maximum length of source and target sentences
    max_inp_seq = max([len(txt) for txt in enc_sequence_inps])
    max_trg_seq = max([len(txt) for txt in dec_sequence_inps])

    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    spa_vocab_size = len(spa_tokenizer.word_index) + 1

    print('max sequence length in language 1', max_inp_seq)
    print('max sequence length in language 2', max_trg_seq)

    # add zero padding to our sentences
    # padding is necessary in case batch processing
    enc_sequence_inps = tf.keras.preprocessing.sequence.pad_sequences(enc_sequence_inps, max_inp_seq, padding='pre')
    dec_sequence_inps = tf.keras.preprocessing.sequence.pad_sequences(dec_sequence_inps, max_trg_seq, padding='post')
    # our target (ground truth) is one token ahead of decoder input
    dec_sequence_outputs = np.zeros_like(dec_sequence_inps)
    dec_sequence_outputs[:, :max_trg_seq - 1] = dec_sequence_inps[:, 1:]
    dec_sequence_outputs = tf.keras.utils.to_categorical(dec_sequence_outputs, spa_vocab_size)

    # create two dictionary for convert id to word (its used in inference time)
    spa_index_to_word = dict([(value, key) for (key, value) in spa_tokenizer.word_index.items()])
    eng_index_to_word = dict([(value, key) for (key, value) in eng_tokenizer.word_index.items()])

    s2s = S2SModel(eng_vocab_size, spa_vocab_size, 128, 128, 1)
    s2s.build_graph()
    s2s.train(enc_sequence_inps, dec_sequence_inps, dec_sequence_outputs)
    s2s.inference(enc_sequence_inps[0:30])
