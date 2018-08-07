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

from lesson1.data_helper import DataHelper

imdb = tf.keras.datasets.imdb

vocab_size = 10000
embed_size = 32
seq_len = 256
batch_size = 512
training_iters = 1000000
epoch = 40

(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=vocab_size)

# train_y = tf.keras.utils.to_categorical(train_y)
# test_y = tf.keras.utils.to_categorical(test_y)

train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, maxlen=seq_len)
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, maxlen=seq_len)

data_helper = DataHelper(train_x, train_y)

# Input
X = tf.placeholder(dtype=tf.int32, shape=[None, seq_len])
Y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

# Model
embedding = tf.Variable(tf.truncated_normal([vocab_size, embed_size]))
inputs = tf.nn.embedding_lookup(embedding, X)  # [batch_size * 256 * 100]

pooling = tf.layers.max_pooling1d(inputs, 2, strides=1, padding="valid")  # [?,32]
global_pooling = tf.reduce_mean(pooling, 1)
dense1 = tf.layers.dense(global_pooling, 16, activation=tf.nn.relu)
logit = tf.layers.dense(dense1, 1)
pred = tf.round(tf.nn.sigmoid(logit))
# pred = tf.cast(pred, tf.int32)
correct_pred = tf.equal(pred, tf.cast(Y, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(Y, tf.float32), logits=logit))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    while step * batch_size < training_iters:

        # We will read a batch of 100 images [100 x 784] as batch_x
        # batch_y is a matrix of [100x10]
        batch_x, batch_y = data_helper.next_batch(batch_size)
        batch_y = np.expand_dims(batch_y, 1)

        # We consider each row of the image as one sequence
        # Reshape data to get 28 seq of 28 elements, so that, batxh_x is [100x28x28]
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

        if step % batch_size == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={X: batch_x, Y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    test_helper = DataHelper(test_x, test_y)
    step = 0
    acc = 0
    while test_helper.epoch_completed == 0:
        step = step + 1
        x_batch, y_batch = test_helper.next_batch(batch_size)
        y_batch = np.expand_dims(y_batch, 1)
        acc = acc + sess.run(accuracy, feed_dict={X: x_batch, Y: y_batch})
        print("Testing Accuracy on step ", step, ' is: ', acc / step)
    print('final accuracy', acc / step)
