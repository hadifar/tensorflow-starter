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

from helper.data_helper import DataHelper

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255
# x_train = x_train[:20]
# x_test = x_test[:20]
# y_train = y_train[:20]
# y_test = y_test[:20]

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

data_helper = DataHelper(x_train, y_train)

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='image')
Y = tf.placeholder(tf.float32, shape=[None, 10], name='label')
pKeep = tf.placeholder(tf.float32)

K = 6
L = 12
M = 18

# convolution network
W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=0.1))
B1 = tf.Variable(tf.ones([K]) / 10)
W2 = tf.Variable(tf.truncated_normal([4, 4, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L]) / 10)
W3 = tf.Variable(tf.truncated_normal([3, 3, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M]) / 10)
# flatten convolution
W4 = tf.Variable(tf.truncated_normal([M * 8 * 8, 128], stddev=0.1))
B4 = tf.Variable(tf.ones([128]) / 10)
W5 = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10]) / 10)

Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding="SAME") + B2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding="SAME") + B3)
Y3 = tf.reshape(Y3, [-1, M * 8 * 8])
Y3 = tf.nn.dropout(Y3, pKeep)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
y_logit = tf.matmul(Y4, W5) + B5
Y_pred = tf.nn.softmax(y_logit)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_logit, labels=Y)
cross_entropy = tf.reduce_mean(cross_entropy)

is_correct = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(25000):
    batch_X, batch_Y = data_helper.next_batch(100)

    train_data = {X: batch_X, Y: batch_Y, pKeep: 0.75}
    sess.run(train_step, feed_dict=train_data)

    acc_train, loss_train = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    test_data = {X: x_test, Y: y_test, pKeep: 1.0}
    acc_test, loss_test = sess.run([accuracy, cross_entropy], feed_dict=test_data)

    if i % 100 == 0:
        print(i)
        print("training::: accuracy", acc_train, "loss", loss_train)
        print("testing::: accuracy", acc_test, "loss", loss_test)
        print(50 * '-')
