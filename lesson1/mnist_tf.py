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
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../lesson1/data/", one_hot=True)

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, [None, 784], name="X")
    Y_truth = tf.placeholder(tf.float32, [None, 10], name="Y")

    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name='bias')

    init = tf.initialize_all_variables()

    # re_X = tf.reshape(X, [-1, 784])
    Y_pred = tf.nn.softmax(tf.matmul(X, W) + b)

    cross_entropy = -tf.reduce_sum(Y_truth * tf.log(Y_pred))
    is_correct = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_truth, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.004)
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.Session()
    sess.run(init)

    for i in range(10000):
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, Y_truth: batch_Y}
        sess.run(train_step, feed_dict=train_data)

        acc_train, loss_train = sess.run([accuracy, cross_entropy], feed_dict=train_data)

        test_data = {X: mnist.test.images, Y_truth: mnist.test.labels}
        acc_test, loss_test = sess.run([accuracy, cross_entropy], feed_dict=test_data)

        if i % 300 == 0:
            print("training::: accuracy", acc_train, "loss", loss_train)
            print("testing::: accuracy", acc_test, "loss", loss_test)
            print(50 * '-')
