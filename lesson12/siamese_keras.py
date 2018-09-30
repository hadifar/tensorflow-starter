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
import numpy as np

from lesson12 import general_utils

_BATCH_SIZE = 128


def tfdata_generator(question_pairs, labels, is_training, batch_size=128):
    '''Construct a data generator using tf.Dataset'''

    dataset = tf.data.Dataset.from_tensor_slices((question_pairs, labels))
    if is_training:
        dataset = dataset.shuffle(1000)  # depends on sample size

    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


embedding, word_index, train, test, dev = general_utils.load_data()
(q1_dev, q2_dev, label_dev) = dev
(q1_test, q2_test, label_test) = test
(q1_train, q2_train, label_train) = train

nb_vocab = min(len(word_index), 60000) + 1

train_dataset = tfdata_generator(np.array([q1_train, q2_train]), label_train, is_training=True, batch_size=_BATCH_SIZE)
dev_dataset = tfdata_generator((q1_dev, q2_dev), label_dev, is_training=False, batch_size=_BATCH_SIZE)
test_dataset = tfdata_generator(np.array([q1_test, q2_test]), label_test, is_training=False, batch_size=_BATCH_SIZE)

inps1 = keras.layers.Input(shape=(50,))
inps2 = keras.layers.Input(shape=(50,))

embed = keras.layers.Embedding(input_dim=nb_vocab, output_dim=300, weights=[embedding], trainable=False)
embed1 = embed(inps1)
embed2 = embed(inps2)

gru = keras.layers.CuDNNGRU(256)
gru1 = gru(embed1)
gru2 = gru(embed2)

dense = keras.layers.Dense(256, activation='relu')
dense1 = dense(gru1)
dense2 = dense(gru2)

dropout = keras.layers.Dropout(0.5)
dropout1 = dropout(dense1)
dropout2 = dropout(dense2)

concat = keras.layers.concatenate([dropout1, dropout2])

preds = keras.layers.Dense(256, 'sigmoid')(concat)

model = keras.models.Model(inputs=[inps1, inps2], outputs=preds)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(
    train_dataset.make_one_shot_iterator(),
    steps_per_epoch=len(q1_train) // _BATCH_SIZE,
    epochs=50,
    # validation_data=test_dataset.make_one_shot_iterator(),
    # validation_steps=len(q1_test) // _BATCH_SIZE,
    verbose=1)
