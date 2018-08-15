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

imdb = tf.keras.datasets.imdb

training_iters = 150000
vocabulary_size = 10000
embedding_size = 32
seq_len = 256
learning_rate = 0.01
rnn_size = 32
batch_size = 512

nb_epoch = 300

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocabulary_size)

word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256, padding='post',
                                                           value=word_index["<PAD>"])
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256, padding='post',
                                                          value=word_index["<PAD>"])

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

rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
outputs, states = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state=initial_state)
output = tf.reshape(tf.split(outputs, seq_len, axis=1, name='split')[-1], [batch_size, -1])

weights = tf.Variable(tf.truncated_normal([32, 2], stddev=0.1))
biases = tf.Variable(tf.ones([2]) / 10)

logit = tf.matmul(output, weights) + biases
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
            print("Iter " + str(current_step) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))

        if current_step > nb_epoch:
            break

    feed_data = {x: test_data, y: test_labels}
    acc = sess.run([accuracy], feed_dict=feed_data)
    print('accuracy: {}'.format(acc))
