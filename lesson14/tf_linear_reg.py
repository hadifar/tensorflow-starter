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

# create dummy data
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
X_data = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
Y_data = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
X_data = np.expand_dims(X_data, axis=1)
Y_data = np.expand_dims(Y_data, axis=1)

x = tf.placeholder(dtype=tf.float32, shape=(1, 1))
y = tf.placeholder(dtype=tf.float32, shape=(1, 1))

# our linear regression model
dense = tf.layers.Dense(units=1)(x)

# our optimizer and loss
loss = tf.losses.mean_squared_error(labels=y, predictions=dense)
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # our training loop
    for index in range(10000):
        batch_x, batch_y = X_data[index % 5], Y_data[index % 5]
        _, mae = sess.run([train_step, loss], feed_dict={x: [batch_x], y: [batch_y]})
        if (index + 1) % 100 == 0:
            print('loss at step {}: {:5.1f}'.format(index, mae))

    # finally save the model with simple_save API
    tf.saved_model.simple_save(session=sess,
                               export_dir='./test/',
                               inputs={'x': x},
                               outputs={'y': dense})
