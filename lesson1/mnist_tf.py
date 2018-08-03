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
from tensorflow import keras

from lesson1.data_helper import DataHelper

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
m_train = x_train.shape[0]
m_test = x_test.shape[0]

x_train = np.reshape(x_train, [m_train, -1])
x_test = np.reshape(x_test, [m_test, -1])

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

data_helper = DataHelper(data=x_train, label=y_train)

X = tf.placeholder(tf.float32, [None, 784], name="X")
Y_truth = tf.placeholder(tf.float32, [None, 10], name="Y")
W = tf.Variable(tf.zeros([784, 10]), name='weights')
b = tf.Variable(tf.zeros([10]), name='bias')
init = tf.initialize_all_variables()

Y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
cross_entropy = -tf.reduce_sum(Y_truth * tf.log(Y_pred))

is_correct = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_truth, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.004)

train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)
for i in range(10000):
    batch_X, batch_Y = data_helper.next_batch(100)

    train_data = {X: batch_X, Y_truth: batch_Y}
    sess.run(train_step, feed_dict=train_data)

    acc_train, loss_train = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    test_data = {X: x_test, Y_truth: y_test}
    acc_test, loss_test = sess.run([accuracy, cross_entropy], feed_dict=test_data)

    if i % 300 == 0:
        print(i)
        print("training::: accuracy", acc_train, "loss", loss_train)
        print("testing::: accuracy", acc_test, "loss", loss_test)
        print(50 * '-')
