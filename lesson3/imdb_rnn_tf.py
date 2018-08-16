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

from lesson1.data_helper import DataHelper

tf.reset_default_graph()

imdb = tf.keras.datasets.imdb

vocabulary_size = 10000
embedding_size = 64
seq_len = 256
learning_rate = 0.001
rnn_size = 64
batch_size = 512

nb_epoch = 300

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocabulary_size)

train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256)

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

data_helper = DataHelper(train_data, train_labels)

# Input data
x = tf.placeholder(tf.int32, shape=[None, seq_len])
y = tf.placeholder(tf.int32, shape=[None, 2])
global_step = tf.Variable(0, trainable=False)

# Model
embeddings = tf.get_variable("embedding", [vocabulary_size, embedding_size])
inputs = tf.nn.embedding_lookup(embeddings, x)

# define RNN layer
rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
outputs, states = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state=initial_state)
# RNN outputs: [batch_size * seq_len * hidden_size]
# split and extract only last output
output = tf.reshape(tf.split(outputs, seq_len, axis=1, name='split')[-1], [batch_size, -1])

# Dense layer
dense1 = tf.layers.dense(output, 16, activation='relu')
logit = tf.layers.dense(output, 2)
pred = tf.nn.softmax(logit)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logit))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(cost, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while True:

        batch_x, batch_y = data_helper.next_batch(batch_size)

        _, step = sess.run([train_step, global_step], feed_dict={x: batch_x, y: batch_y})

        current_step = tf.train.global_step(sess, global_step)

        if step % 10 == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(current_step) + ", Mini batch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))

        if current_step > nb_epoch:
            break

    test_helper = DataHelper(test_data, test_labels)
    step = 0
    acc = 0
    while test_helper.epoch_completed == 0:
        step = step + 1
        x_batch, y_batch = test_helper.next_batch(batch_size)
        cur_acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
        acc = acc + cur_acc
    print('final accuracy', acc / step)
