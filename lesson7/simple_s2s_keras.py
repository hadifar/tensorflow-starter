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

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU, CuDNNGRU
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from lesson7.utils import TextUtils

tf.reset_default_graph()

###############################################################################
###############################################################################
###############################################################################

text_utils = TextUtils()

enc_sequence_inps, dec_sequence_inps, dec_sequence_outputs = text_utils.load_data(nb_examples=256)
dec_sequence_outputs = tf.keras.utils.to_categorical(dec_sequence_outputs)  # currently keras has issue with sparse loss

max_inp_seq = text_utils.max_inp_seq
max_trg_seq = text_utils.max_trg_seq

eng_vocab_size = text_utils.eng_vocab_size
spa_vocab_size = text_utils.spa_vocab_size

spa_index_to_word = text_utils.spa_index_to_word
eng_index_to_word = text_utils.eng_index_to_word


################################################################################
################################################################################
################################################################################


def one_hot(x, num_classes):
    return K.one_hot(x, num_classes=num_classes)


enc_size = 256
dec_size = 256

# encoder part
enc_inp = Input(shape=(None,), dtype=tf.int32)
# our inputs are sparse but we need one-hot encoding
enc_one_hot = Lambda(function=one_hot,
                     arguments={'num_classes': eng_vocab_size},
                     output_shape=(max_inp_seq, eng_vocab_size))(enc_inp)
# use CuDNNGRU if available, is 3x faster
if tf.test.is_gpu_available():
    enc_gru = CuDNNGRU(units=enc_size, return_state=True)
    enc_output, enc_state = enc_gru(enc_one_hot)
else:
    enc_gru = GRU(units=enc_size, return_state=True)
    enc_output, enc_state = enc_gru(enc_one_hot)

# decoder part
dec_inp = Input(shape=(None,), dtype=tf.int32)
# our outputs are sparse but we need one-hot encoding
dec_one_hot = Lambda(function=one_hot,
                     arguments={'num_classes': spa_vocab_size},
                     output_shape=(max_trg_seq, spa_vocab_size))(dec_inp)
# use CuDNNGRU if available, is 3x faster
if tf.test.is_gpu_available():
    dec_gru = CuDNNGRU(units=dec_size, return_sequences=True, return_state=True)
    dec_output, _ = dec_gru(dec_one_hot, initial_state=enc_state)
else:
    dec_gru = GRU(units=dec_size, return_sequences=True, return_state=True)
    dec_output, _ = dec_gru(dec_one_hot, initial_state=enc_state)

dec_dense = Dense(spa_vocab_size, activation='softmax')
pred = dec_dense(dec_output)

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
model = load_model('s2s.hd5')

encoder_inputs = model.input[0]  # input_1
_, encoder_states = model.layers[4].output  # gru_1
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_one_hot = model.layers[3](decoder_inputs)
decoder_states_inputs = [Input(shape=(256,), name='input_3')]
decoder_gru = model.layers[5]
decoder_outputs, decoder_states = decoder_gru(decoder_one_hot, initial_state=decoder_states_inputs)
decoder_dense = model.layers[6]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + [decoder_states])


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0][0] = 0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, new_state = decoder_model.predict([target_seq] + [states_value])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            sample_word = ''
        else:
            sample_word = spa_index_to_word[sampled_token_index]
        decoded_sentence += sample_word + ' '

        # Exit condition: either hit max length
        # or find stop character.
        if sample_word == '</s>' or len(decoded_sentence.split()) > max_trg_seq:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0][0] = sampled_token_index

        # Update states
        states_value = new_state

    return decoded_sentence


# translate 32 sentences from data
for seq_index in range(32):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = enc_sequence_inps[seq_index: seq_index + 1]

    text_utils.print_sentence(input_seq)

    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Decoded sentence:', decoded_sentence)
