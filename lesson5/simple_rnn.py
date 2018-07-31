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

print('load data')
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

print('preprocessing...')
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=256)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=256)

x_val = x_train[:10000]
y_val = y_train[:10000]

x_train = x_train[10000:]
y_train = y_train[10000:]

print('build model')
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.SimpleRNN(50))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('train model')
model.fit(x_train,
          y_train,
          epochs=15,
          batch_size=512,
          validation_data=(x_val, y_val),
          verbose=1)

print('evaluation')
evaluation = model.evaluate(x_test, y_test, batch_size=512)
print('Accuracy:', evaluation[1], 'Loss:', evaluation[0])
