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
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = tf.constant(X)
y = tf.constant(y)

X = tf.expand_dims(X, axis=1)
y = tf.expand_dims(y, axis=1)


class LinearRegression(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense = tf.layers.Dense(units=1,
                                     kernel_initializer=tf.zeros_initializer(),
                                     bias_initializer=tf.zeros_initializer())

    def __call__(self, inputs):
        outputs = self.dense(inputs)
        return outputs


model = LinearRegression()

optimizer = tf.train.GradientDescentOptimizer(1e-3)

for _ in range(10000):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables)
