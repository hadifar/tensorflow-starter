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

tf.enable_eager_execution()

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.boston_housing.load_data()

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

(train_data, train_labels) = tf.constant(train_data), tf.constant(train_labels)
(test_data, test_labels) = tf.constant(test_data), tf.constant(test_labels)


class LinearRegression(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1,
                                           kernel_initializer=tf.constant_initializer(0),
                                           bias_initializer=tf.constant_initializer(0))

    def call(self, inputs):
        output = self.dense(inputs)
        return output


def loss_f(model, input, labels):
    pred = model(input)
    return tf.losses.mean_squared_error(tf.expand_dims(labels, axis=1), pred)


model = LinearRegression()
optimizer = tf.train.GradientDescentOptimizer(1e-3)

for e in range(5000):
    with tf.GradientTape() as tape:
        loss = loss_f(model, train_data, train_labels)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(loss_f(model, test_data, test_labels).numpy())
