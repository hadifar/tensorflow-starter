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

from __future__ import print_function

import os

import nltk
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

################################################################################
################################################################################
################################################################################
# nb_examples = 118964
nb_examples = 100

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


eng_tokenizer = get_tokenizer(input_texts)
spa_tokenizer = get_tokenizer(target_texts)

input_sequences = eng_tokenizer.texts_to_sequences(input_texts)
target_sequences = spa_tokenizer.texts_to_sequences(target_texts)

max_inp_seq = max([len(txt) for txt in input_sequences])
max_trg_seq = max([len(txt) for txt in target_sequences])

eng_vocab_size = len(eng_tokenizer.word_index)
spa_vocab_size = len(spa_tokenizer.word_index)

print('max sequence length in language 1', max_inp_seq)
print('max sequence length in language 2', max_trg_seq)

input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, max_inp_seq)
target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, max_trg_seq)
output_sequences = np.zeros_like(target_sequences)
output_sequences[:, :max_trg_seq - 1] = target_sequences[:, 1:]

################################################################################
################################################################################
################################################################################



