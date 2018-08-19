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

import time

import numpy as np
import tensorflow as tf

# Step 1: Read in data
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

m_train = train_x.shape[0]
m_test = test_x.shape[0]

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)
train_x = np.reshape(train_x, [m_train, -1])
test_x = np.reshape(test_x, [m_test, -1])

train = train_x.astype(np.float32), train_y.astype(np.float32)
test = test_x.astype(np.float32), test_y.astype(np.float32)

# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)  # if you want to shuffle your data
train_data = train_data.batch(128)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(10000)
test_data = test_data.batch(128)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
X, Y = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)  # initializer for train_data

# create weights and bias for logistic regression
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.get_variable(name='weight',
                    initializer=tf.truncated_normal(shape=[784, 10], mean=0, stddev=0.01))
b = tf.get_variable(name='bias', initializer=tf.zeros([10]))

# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = tf.matmul(X, w) + b

# use cross entropy of softmax of logits as the loss function
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
loss = tf.reduce_mean(loss)

# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(30):
        sss = sess.run(train_init)  # drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)  # drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds / m_test))
writer.close()
