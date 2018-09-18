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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

tf.reset_default_graph()

###############################################################################
###############################################################################
###############################################################################

nb_examples = 10000
nb_examples = 64

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
enc_sequence_inps = tf.keras.preprocessing.sequence.pad_sequences(enc_sequence_inps, max_inp_seq)
dec_sequence_inps = tf.keras.preprocessing.sequence.pad_sequences(dec_sequence_inps, max_trg_seq)
# our target (ground truth) is one token ahead of decoder input
dec_sequence_outputs = np.zeros_like(dec_sequence_inps)
dec_sequence_outputs[:, :max_trg_seq - 1] = dec_sequence_inps[:, 1:]
dec_sequence_outputs = tf.keras.utils.to_categorical(dec_sequence_outputs, spa_vocab_size)

# create two dictionary for convert id to word (its used in inference time)
spa_index_to_word = dict([(value, key) for (key, value) in spa_tokenizer.word_index.items()])
eng_index_to_word = dict([(value, key) for (key, value) in eng_tokenizer.word_index.items()])


################################################################################
################################################################################
################################################################################


def one_hot(x, num_classes):
    return K.one_hot(x, num_classes=num_classes)


enc_size = 256
dec_size = 256

# encoder part
enc_inp = Input(shape=(max_inp_seq,), dtype=tf.int32)
# our inputs are sparse but we need one-hot encoding
enc_one_hot = Lambda(one_hot, arguments={'num_classes': eng_vocab_size})(enc_inp)
# use CuDNNGRU if available, is 3x faster
if tf.test.is_gpu_available():
    enc_output, enc_state = CuDNNGRU(units=enc_size, return_state=True)(enc_one_hot)
else:
    enc_output, enc_state = GRU(units=enc_size, return_state=True)(enc_one_hot)

# decoder part
dec_inp = Input(shape=(max_trg_seq,), dtype=tf.int32)
# our outputs are sparse but we need one-hot encoding
dec_one_hot = Lambda(one_hot, arguments={'num_classes': spa_vocab_size})(dec_inp)
# use CuDNNGRU if available, is 3x faster
if tf.test.is_gpu_available():
    dec_output = CuDNNGRU(units=dec_size, return_sequences=True)(dec_one_hot, initial_state=enc_state)
else:
    dec_output = GRU(units=dec_size, return_sequences=True)(dec_one_hot, initial_state=enc_state)

pred = Dense(spa_vocab_size, activation='softmax')(dec_output)

# compile and fit
model = Model(inputs=[enc_inp, dec_inp], outputs=pred)
model.compile(optimizer=Adam(0.005), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([enc_sequence_inps, dec_sequence_inps], dec_sequence_outputs, batch_size=128, epochs=100)

# save model
model.save('s2s.hd5')

################################################################################
################################################################################
################################################################################
# retrieve model
new_model = load_model('s2s.hd5')


def search(our_model, src_input, top_k=1, sequence_max_len=25):
    # (log(1), initialize_of_zeros)
    candidate_sequences = [(0, [0] * sequence_max_len)]

    # l : point on target sentence to predict
    for l in range(sequence_max_len):
        tmp_candidates = []
        for prob, sent_predict in candidate_sequences:
            predicted = our_model.predict([np.expand_dims(src_input, 0), np.expand_dims(sent_predict, 0)])[0]
            # top k!
            possible_k = predicted[l].argsort()[-top_k:][::-1]

            # add to all possible candidates for k-beams
            tmp_candidates += [
                (
                    sum(np.log(predicted[i][sent_predict[i + 1]]) for i in range(l)) + np.log(predicted[l][next_wid]),
                    list(sent_predict[:l]) + [next_wid] + [0] * (sequence_max_len - l - 1)
                )
                for next_wid in possible_k
            ]

        # pick top k
        candidate_sequences = sorted(tmp_candidates)[-top_k:]

    return candidate_sequences


# inference
def translate(inp_sequence):
    k_beam = search(model, inp_sequence, top_k=2, sequence_max_len=max_trg_seq)

    for prob, pred_sequences in k_beam:
        print('prob ', prob, 'translation:', " ".join([spa_index_to_word[t] for t in pred_sequences if t != 0]))


# pick 32 sentences from corpus then translate them into spanish
for seq_i in range(32):
    print(" ".join([eng_index_to_word[t] for t in enc_sequence_inps[seq_i] if t != 0]))
    print(" ".join([spa_index_to_word[t] for t in dec_sequence_inps[seq_i] if t != 0]))
    translate(enc_sequence_inps[seq_i])
    print(50 * '-')
