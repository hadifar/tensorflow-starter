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
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Lambda, Dense, CuDNNGRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.reset_default_graph()

###############################################################################
###############################################################################
###############################################################################
nb_examples = 118964

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

eng_vocab_size = len(eng_tokenizer.word_index) + 1
spa_vocab_size = len(spa_tokenizer.word_index) + 1

print('max sequence length in language 1', max_inp_seq)
print('max sequence length in language 2', max_trg_seq)

input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, max_inp_seq)
target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, max_trg_seq)
output_sequences = np.zeros_like(target_sequences)
output_sequences[:, :max_trg_seq - 1] = target_sequences[:, 1:]
output_sequences = tf.keras.utils.to_categorical(output_sequences, spa_vocab_size)

spa_index_to_word = dict([(value, key) for (key, value) in spa_tokenizer.word_index.items()])
eng_index_to_word = dict([(value, key) for (key, value) in eng_tokenizer.word_index.items()])


################################################################################
################################################################################
################################################################################


def one_hot(x, num_classes):
    return K.one_hot(x, num_classes=num_classes)


enc_size = 128
dec_size = 128

# encoder part
enc_inp = Input(shape=(max_inp_seq,), dtype=tf.int32)
enc_one_hot = Lambda(one_hot, arguments={'num_classes': eng_vocab_size})(enc_inp)
if tf.test.is_gpu_available():
    enc_output, enc_state = CuDNNGRU(units=enc_size, return_state=True)(enc_one_hot)
else:
    enc_output, enc_state = GRU(units=enc_size, return_state=True)(enc_one_hot)

# decoder part
dec_inp = Input(shape=(max_trg_seq,), dtype=tf.int32)
dec_one_hot = Lambda(one_hot, arguments={'num_classes': spa_vocab_size})(dec_inp)
if tf.test.is_gpu_available():
    dec_output = CuDNNGRU(units=dec_size, return_sequences=True)(dec_one_hot, initial_state=enc_state)
else:
    dec_output = GRU(units=dec_size, return_sequences=True)(dec_one_hot, initial_state=enc_state)
pred = Dense(spa_vocab_size, activation='softmax')(dec_output)

# compile and fit
model = Model(inputs=[enc_inp, dec_inp], outputs=pred)
model.compile(optimizer=Adam(0.005), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([input_sequences, target_sequences], output_sequences, batch_size=128, epochs=100)

# save model
model.save('s2s.hd5')

################################################################################
################################################################################
################################################################################
# retrieve model
new_model = tf.keras.models.load_model('s2s.hd5')


def translate(inp_sequence):
    generated = ''
    new_state = np.zeros(max_trg_seq)
    i = 1
    while True:
        p = new_model.predict([np.expand_dims(inp_sequence, 0), np.expand_dims(new_state, 0)])
        p_id = p[0][i].argmax()
        if p_id != 0:
            generated = generated + ' ' + spa_index_to_word[p_id]
            new_state[i - 1] = p_id
            i = i + 1
        else:
            break
    return generated


for sequence in range(32):
    print(" ".join([eng_index_to_word[t] for t in input_sequences[sequence] if t != 0]))
    print(" ".join([spa_index_to_word[t] for t in target_sequences[sequence] if t != 0]))
    print(translate(input_sequences[sequence]))
    print(50 * '-')
