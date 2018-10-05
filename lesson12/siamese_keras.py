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

import tensorflow as tf
from tensorflow import keras

from lesson12 import general_utils

_BATCH_SIZE = 128
_MAX_VOCAB = 100000

embedding, word_index, train, test, dev = general_utils.load_data()
(q1_dev, q2_dev, label_dev) = dev
(q1_test, q2_test, label_test) = test
(q1_train, q2_train, label_train) = train

nb_vocab = min(len(word_index), _MAX_VOCAB) + 1

inps1 = keras.layers.Input(shape=(50,), dtype=tf.int32)
inps2 = keras.layers.Input(shape=(50,), dtype=tf.int32)

embed = keras.layers.Embedding(input_dim=nb_vocab, output_dim=300, weights=[embedding], trainable=False)
embed1 = embed(inps1)
embed2 = embed(inps2)

if tf.test.is_gpu_available():
    gru = keras.layers.CuDNNGRU(256)
else:
    gru = keras.layers.GRU(256)
gru1 = gru(embed1)
gru2 = gru(embed2)

dense = keras.layers.Dense(256, activation='relu')
dense1 = dense(gru1)
dense2 = dense(gru2)

dropout = keras.layers.Dropout(0.5)
dropout1 = dropout(dense1)
dropout2 = dropout(dense2)

concat = keras.layers.concatenate([dropout1, dropout2])

preds = keras.layers.Dense(256, activation='relu')(concat)
preds = keras.layers.Dense(1, activation='sigmoid')(preds)

model = keras.models.Model(inputs=[inps1, inps2], outputs=preds)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(
    x=[q1_train, q2_train], y=label_train,
    epochs=50,
    validation_data=([q1_test, q2_test], label_test),
    verbose=1)
