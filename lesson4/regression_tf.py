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

from lesson1.data_helper import DataHelper

(train_data, train_labels), (test_data, test_labels) = keras.datasets.boston_housing.load_data()

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

data_helper = DataHelper(train_data, train_labels)

feature_num = train_data.shape[1]

x = tf.placeholder(dtype=tf.float32, shape=[None, feature_num], name='feature')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target')

global_step = tf.Variable(0, trainable=False, name='global_step')

l1 = tf.layers.dense(x, 64, activation=tf.nn.relu)
l2 = tf.layers.dense(l1, 64, activation=tf.nn.relu)
pred = tf.layers.dense(l2, 1)

mae = tf.metrics.mean_absolute_error(labels=y, predictions=pred)
loss = tf.losses.mean_squared_error(labels=y, predictions=pred)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while True:
        batch_x, batch_y = data_helper.next_batch(32)

        feed_data = {x: batch_x,
                     y: batch_y}

        _, mae, step = sess.run([optimizer, mae, global_step], feed_dict=feed_data)

        current_step = tf.train.global_step(sess, global_step)

        if step % 10 == 0:
            print('step: {}/{}... '.format(step, 500),
                  'loss: {:.4f}... '.format(loss))

        if current_step >= 500:
            break
