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
import pandas as pd
import tensorflow as tf

from helper.data_helper import DataHelper

train = pd.read_csv('/Users/mac/Downloads/train.csv')
train = train.drop(columns=['ID'])
train_labels = train['medv'].values
train_data = train.drop(columns='medv').values

test_data = pd.read_csv('/Users/mac/Downloads/test.csv')
# test_data = train.drop(columns=[])

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
# test_data = (test_data - mean) / std

data_helper = DataHelper(train_data, train_labels)

feature_num = train_data.shape[1]

x = tf.placeholder(dtype=tf.float32, shape=[None, feature_num], name='feature')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target')

global_step = tf.Variable(0, trainable=False, name='global_step')

l1 = tf.layers.dense(x, 64, activation=tf.nn.relu)
l2 = tf.layers.dense(l1, 64, activation=tf.nn.relu)
pred = tf.layers.dense(l2, 1)

mae, mae_op = tf.metrics.mean_absolute_error(labels=y, predictions=pred)
loss = tf.losses.mean_squared_error(labels=y, predictions=pred)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train_step = optimizer.minimize(loss, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    while True:
        batch_x, batch_y = data_helper.next_batch(128)
        batch_y = np.expand_dims(batch_y, axis=1)
        feed_data = {x: batch_x,
                     y: batch_y}

        _, step, mean_avg_error = sess.run([train_step, global_step, mae_op], feed_dict=feed_data)

        current_step = tf.train.global_step(sess, global_step)

        if step % 1000 == 0:
            print('step: {}/{}... '.format(step, 1000000),
                  'mae: {}'.format(mean_avg_error))

        if current_step >= 1000000:
            break

    # test_labels = np.expand_dims(test_labels, axis=1)
    total_res = []
    ids = []
    for i in range(len(test_data)):
        batch_x = np.expand_dims(test_data.values[i][1:], axis=0)
        batch_x = (batch_x - mean) / std
        # batch_y = np.expand_dims(test_labels[0], axis=0)
        feed_data = {x: batch_x}
        res = sess.run([pred], feed_dict=feed_data)
        total_res.append(float(res[0][0][0]))
        ids.append(int(test_data.values[i][0]))
    # print(total_res)
    sub = pd.DataFrame({'ID': ids, 'medv': total_res}, columns=['ID', 'medv'])
    sub.to_csv('kaggle_sub.csv', index=False)
