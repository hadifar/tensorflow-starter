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

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize data [images are gray-scale 64*64 pixels]
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten images similar to keras.layers.flatten
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

model = tf.keras.models.Sequential([
    # convert 28*28 into 784
    # tf.keras.layers.Flatten(),
    # MLP layer with 512 neuron
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # some drop out for avoid over-fitting
    tf.keras.layers.Dropout(0.2),
    # prediction layer (10 classes to predict)
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# what is the difference between categorical_crossentropy and sparse_categorical_crossentropy ?
# categorical_crossentropy -> one-hot
# sparse_categorical_crossentropy -> integer-hot

# categorical_crossentropy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, tf.keras.utils.to_categorical(y_train, 10), epochs=5)
print(model.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 10)))

# sparse_categorical_crossentropy
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)
# print(model.evaluate(x_test, y_test))

# evaluate method returns the loss & metrics values for the model in test mode.
