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
import random

import numpy as np
import tensorflow.keras as keras


# load data (raw text file)
################################################################################
################################################################################

path = '../lesson9/data/test.txt'
with open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

# create two dictionaries: (index to id) & (id to index)
################################################################################
################################################################################

chars = sorted(list(set(text)))
len_of_char = len(chars)
print('total chars:', len(chars))
char2idx = dict((c, i) for i, c in enumerate(chars))
idx2char = dict((i, c) for i, c in enumerate(chars))

# convert each character to an unique id
################################################################################
################################################################################

max_length = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - max_length, step):
    sentences.append(text[i: i + max_length])
    next_chars.append(text[i + max_length])
print('nb sequences:', len(sentences))

# create proper vector for training model
################################################################################
################################################################################
print('Vectorization...')
x = np.zeros((len(sentences), max_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences)))
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char2idx[char]] = 1
    y[i] = char2idx[next_chars[i]]


# two helper functions to print generated text at the end of each epoch
################################################################################
################################################################################
def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - max_length - 1)
    for diversity in [1.0]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + max_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')

        for i in range(400):
            x_pred = np.zeros((1, max_length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char2idx[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            pred_id = sample(preds, diversity)
            next_char = idx2char[pred_id]

            generated += next_char
            sentence = sentence[1:] + next_char

        print(generated)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Define simple GRU model for generating text
# In case of using CPU replace the CuDNNGRU with GRU
################################################################################
################################################################################

model = keras.Sequential()
# model.add(keras.layers.CuDNNGRU(256, input_shape=(max_length, len_of_char)))
model.add(keras.layers.GRU(256, input_shape=(max_length, len_of_char)))
model.add(keras.layers.Dense(len_of_char, activation='softmax'))

# Create graph and train the model for 60 epoch
################################################################################
################################################################################

model.compile(optimizer=keras.optimizers.RMSprop(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
print_callback = keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x=x, y=y, epochs=60, batch_size=128, callbacks=[print_callback])
