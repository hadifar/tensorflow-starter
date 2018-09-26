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

import nltk
import numpy as np
import tensorflow as tf


class TextUtils(object):
    def __init__(self):
        self.nb_examples = 0
        self.eng_tokenizer = None
        self.spa_tokenizer = None
        self.spa_index_to_word = None
        self.eng_index_to_word = None

    def load_data(self, nb_examples=256):
        self.nb_examples = nb_examples

        input_texts, target_texts = self._download_read_data()

        self.eng_tokenizer = self._get_tokenizer(input_texts)
        self.spa_tokenizer = self._get_tokenizer(target_texts)
        # create two dictionary for convert id to word (its used in inference time)
        self.spa_index_to_word = dict([(value, key) for (key, value) in self.spa_tokenizer.word_index.items()])
        self.eng_index_to_word = dict([(value, key) for (key, value) in self.eng_tokenizer.word_index.items()])

        return self._tokenizer_sentences(input_texts, target_texts)

    def _download_read_data(self):
        path_to_zip = tf.keras.utils.get_file(
            'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip',
            extract=True)

        data_path = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

        input_texts = []
        target_texts = []

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        for line in lines[:self.nb_examples]:
            inp_txt, trg_txt = line.split('\t')
            inp_txt = nltk.word_tokenize(inp_txt)
            trg_txt = nltk.word_tokenize(trg_txt)

            input_texts.append('<s>' + ' ' + ' '.join(inp_txt) + ' ' + '</s>')
            target_texts.append('<s>' + ' ' + ' '.join(trg_txt) + ' ' + '</s>')

        print('number of training examples for language1: ', len(input_texts))
        print('number of training examples for language2: ', len(target_texts))
        return input_texts, target_texts

    def _get_tokenizer(self, text):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(text)
        return tokenizer

    def _tokenizer_sentences(self, input_texts, target_texts):
        # convert each sentence to sequence of integers
        enc_sequence_inps = self.eng_tokenizer.texts_to_sequences(input_texts)
        dec_sequence_inps = self.spa_tokenizer.texts_to_sequences(target_texts)

        # find maximum length of source and target sentences
        self.max_inp_seq = max([len(txt) for txt in enc_sequence_inps])
        self.max_trg_seq = max([len(txt) for txt in dec_sequence_inps])

        self.eng_vocab_size = len(self.eng_tokenizer.word_index) + 1
        self.spa_vocab_size = len(self.spa_tokenizer.word_index) + 1

        print('max sequence length in language 1', self.max_inp_seq)
        print('max sequence length in language 2', self.max_trg_seq)

        # add zero padding to our sentences
        # padding is necessary in case batch processing
        enc_sequence_inps = tf.keras.preprocessing.sequence.pad_sequences(enc_sequence_inps,
                                                                          self.max_inp_seq,
                                                                          padding='post')
        dec_sequence_inps = tf.keras.preprocessing.sequence.pad_sequences(dec_sequence_inps,
                                                                          self.max_trg_seq,
                                                                          padding='post')
        # our target (ground truth) is one token ahead of decoder input
        dec_sequence_outputs = np.zeros_like(dec_sequence_inps)
        dec_sequence_outputs[:, :self.max_trg_seq - 1] = dec_sequence_inps[:, 1:]

        return enc_sequence_inps, dec_sequence_inps, dec_sequence_outputs

    def print_sentence(self, sequences):
        trans = ''
        for t in sequences:
            if t != 0:
                trans += self.eng_index_to_word[t]
        print(trans)
