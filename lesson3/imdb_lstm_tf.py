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
batch_size = 256
num_layers = 1

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

# Model
embeddings = tf.get_variable("embedding", [vocabulary_size, embedding_size])
inputs = tf.nn.embedding_lookup(embeddings, x)

rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
stacked_rnn = tf.contrib.rnn.MultiRNNCell([rnn_cell] * num_layers)
initial_state = stacked_rnn.zero_state(batch_size, tf.float32)
outputs, states = tf.nn.dynamic_rnn(stacked_rnn, inputs, initial_state=initial_state)
output = tf.reshape(tf.split(outputs, seq_len, axis=1, name='split')[-1], [batch_size, -1])

weights = tf.Variable(tf.truncated_normal([32, 2], stddev=0.1))
biases = tf.Variable(tf.ones([2]) / 10)

logit = tf.matmul(output, weights) + biases
pred = tf.nn.softmax(logit)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logit))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    while step * batch_size < training_iters:

        # We will read a batch of 100 images [100 x 784] as batch_x
        # batch_y is a matrix of [100x10]
        batch_x, batch_y = data_helper.next_batch(batch_size)

        # We consider each row of the image as one sequence
        # Reshape data to get 28 seq of 28 elements, so that, batxh_x is [100x28x28]
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % batch_size == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    test_helper = DataHelper(test_data, test_labels)
    step = 0
    acc = 0
    while test_helper.epoch_completed == 0:
        step = step + 1
        x_batch, y_batch = test_helper.next_batch(batch_size)
        acc = acc + sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
        print("Testing Accuracy on step ", step, ' is: ', accuracy)
    print('final accuracy', acc / step)
