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
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time
import numpy as np
# tf.enable_eager_execution()

from lesson6 import utils

# Define paramaters for the model
learning_rate = 0.008
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000
flat = False

# Step 1: Read in data
mnist_folder = '/Users/mac/PycharmProjects/tensorflow-starter/lesson6/assignment1/data'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=flat)
if not flat:
    train = (np.expand_dims(train[0], 3), train[1])
    test = (np.expand_dims(test[0], 3), test[1])

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)  # if you want to shuffle your data
train_data = train_data.prefetch(2)
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
#############################
########## TO DO ############
#############################
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(10000)
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
X, Y = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)  # initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
#############################
########## TO DO ############
#############################

K = 6
L = 12
M = 24
D = 100

step = tf.placeholder(tf.int32)

W1 = tf.get_variable(name='W1', initializer=tf.truncated_normal([7, 7, 1, K], stddev=0.1))
b1 = tf.get_variable(name='B1', initializer=tf.ones([K]) / 10)

W2 = tf.get_variable(name='W2', initializer=tf.truncated_normal([6, 6, K, L], stddev=0.1))
b2 = tf.get_variable(name='B2', initializer=tf.ones(L) / 10)

W3 = tf.get_variable(name='W3', initializer=tf.truncated_normal([4, 4, L, M], stddev=0.1))
b3 = tf.get_variable(name='B3', initializer=tf.ones(M) / 10)

W4 = tf.get_variable(name='W4', initializer=tf.truncated_normal([7 * 7 * M, D]))
b4 = tf.get_variable(name='B4', initializer=tf.ones(D) / 10)

W5 = tf.get_variable(name='W5', initializer=tf.truncated_normal([D, 10]))
b5 = tf.get_variable(name='B5', initializer=tf.ones(10) / 10)

# output 28*28
conv1 = tf.nn.leaky_relu(tf.nn.conv2d(X, W1, [1, 1, 1, 1], padding="SAME") + b1)
# output 14*14
conv2 = tf.nn.leaky_relu(tf.nn.conv2d(conv1, W2, [1, 2, 2, 1], padding='SAME') + b2)
# output 7*7
conv3 = tf.nn.leaky_relu(tf.nn.conv2d(conv2, W3, [1, 2, 2, 1], padding='SAME') + b3)

flatten = tf.reshape(conv3, shape=[-1, 7 * 7 * 24])

dense1 = tf.nn.leaky_relu(tf.matmul(flatten, W4) + b4)
logits = tf.matmul(dense1, W5) + b5

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# logits = tf.matmul(img, w) + b
#############################
########## TO DO ############
#############################


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
loss = tf.reduce_mean(loss)
#############################
########## TO DO ############
#############################


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer().minimize(loss)
#############################
########## TO DO ############
#############################


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

summary_acc = tf.summary.scalar("training_accuracy", accuracy)
summary_loss = tf.summary.scalar("training_loss", loss)
summary_merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        sss = sess.run(train_init)  # drawing samples from train_data
        total_loss = 0
        total_acc = 0
        n_batches = 0
        try:
            while True:
                _, l, a, m = sess.run([optimizer, loss, accuracy, summary_merged])
                total_loss += l
                total_acc += a
                n_batches += 1
                writer.add_summary(m, i)
        except tf.errors.OutOfRangeError:
            pass

        print('Loss epoch {0}: {1}'.format(i, total_loss / n_batches))
        print('Acc epoch {0}: {1}'.format(i, total_acc / (n_batches * batch_size)))
        print(50 * '-')

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

    print('Accuracy {0}'.format(total_correct_preds / n_test))
writer.close()
