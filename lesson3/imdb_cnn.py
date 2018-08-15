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

vocabulary_size = 10000
embedding_size = 64
seq_len = 256
batch_size = 512
nb_epoch = 100
dropout = 0.25

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocabulary_size)

word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=seq_len, padding='post',
                                                           value=word_index["<PAD>"])
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=seq_len, padding='post',
                                                          value=word_index["<PAD>"])

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

data_helper = DataHelper(train_data, train_labels)

x = tf.placeholder(tf.int32, shape=[None, seq_len])
y = tf.placeholder(tf.int32, shape=[None, 2])

global_step = tf.Variable(0, trainable=False)

# Model
embeddings = tf.get_variable("embedding", [vocabulary_size, embedding_size])
inputs = tf.nn.embedding_lookup(embeddings, x)

conv1 = tf.layers.conv1d(inputs, 64, 16, strides=1, padding='same', activation=tf.nn.relu)
conv1 = tf.layers.dropout(conv1, rate=dropout)

conv2 = tf.layers.conv1d(conv1, 32, 32, strides=1, padding='same', activation=tf.nn.relu)
conv2 = tf.layers.dropout(conv2, rate=dropout)

conv3 = tf.layers.conv1d(conv2, 16, 48, strides=1, padding='same', activation=tf.nn.relu)
conv3 = tf.layers.dropout(conv3, rate=dropout)

conv4 = tf.layers.conv1d(conv3, 8, 64, strides=1, padding='same', activation=tf.nn.relu)

pool1 = tf.layers.max_pooling1d(conv4, pool_size=2, strides=1)

flatten = tf.layers.flatten(pool1)

dense1 = tf.layers.dense(flatten, 32, activation=tf.nn.relu)
dense1 = tf.layers.dropout(dense1, rate=dropout)

dense2 = tf.layers.dense(dense1, 16, activation=tf.nn.relu)
dense2 = tf.layers.dropout(dense2, rate=dropout)

logit = tf.layers.dense(dense2, 2)
pred = tf.nn.softmax(logit)

is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logit)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(cost, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while True:
        batch_x, batch_y = data_helper.next_batch(batch_size)
        feed_data = {x: batch_x, y: batch_y, }
        _, step, acc = sess.run([train_step, global_step, accuracy], feed_dict=feed_data)

        current_step = tf.train.global_step(sess, global_step)

        if step % 10 == 0:
            print('step: {}/{}... '.format(step, nb_epoch),
                  'accuracy: {}'.format(acc))

        if current_step >= nb_epoch:
            break

    feed_data = {x: test_data, y: test_labels}
    acc = sess.run([accuracy], feed_dict=feed_data)
    print('accuracy: {}'.format(acc))
