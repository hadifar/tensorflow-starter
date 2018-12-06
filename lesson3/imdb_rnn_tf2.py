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

vocabulary_size = 10000
embedding_size = 64
rnn_size = 64
batch_size = 512

# download dataset
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=vocabulary_size)

# add zero padding to our data
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256)

# convert our np.data to tf.data.Dataset and create two iterators for test & train
training_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).repeat(5).shuffle(1024).batch(
    batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).repeat(1).batch(batch_size)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

train_init_op = iterator.make_initializer(training_dataset)
test_init_op = iterator.make_initializer(test_dataset)

# Input data
x, y = iterator.get_next()

# it is a counter
global_step = tf.Variable(0, trainable=False)

# embedding matrix
embeddings_matrix = tf.get_variable("embedding", [vocabulary_size, embedding_size])
embed = tf.nn.embedding_lookup(embeddings_matrix, x)  # batch x seq x embed_size

# our RNN variables
Wx = tf.get_variable(name='Wx', shape=[embedding_size, rnn_size])
Wh = tf.get_variable(name='Wh', shape=[rnn_size, rnn_size])
bias_rnn = tf.get_variable(name='brnn', initializer=tf.zeros([rnn_size]))


# single step in RNN
def rnn_step(prev_hidden_state, x):
    return tf.tanh(tf.matmul(x, Wx) + tf.matmul(prev_hidden_state, Wh) + bias_rnn)

# our unroll function
# notice that our inputs should be transpose
hidden_states = tf.scan(fn=rnn_step,
                        elems=tf.transpose(embed, perm=[1, 0, 2]),
                        initializer=tf.zeros([batch_size, rnn_size]))

# covert to previous shape
outputs = tf.transpose(hidden_states, perm=[1, 0, 2])

# extract last hidden
last_rnn_output = outputs[:, -1, :]

# dense layers variables
Wd1 = tf.get_variable(name='dense1', shape=(rnn_size, 16))
Wd2 = tf.get_variable(name='dense2', shape=(rnn_size, 2))

dense1 = tf.nn.relu(tf.matmul(last_rnn_output, Wd1))
logit = tf.matmul(last_rnn_output, Wd2)
pred = tf.nn.softmax(logit)

# calculate accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# define optimizer, cost, and training step
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit))
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(cost, global_step=global_step)

# training and testing loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op)

    while True:
        try:
            _, step, c, acc = sess.run([train_step, global_step, cost, accuracy])
            if step % 50 == 0:
                print("Iter " + str(step) +
                      ", batch loss {:.6f}".format(c) +
                      ", batch Accuracy= {:.5f}".format(acc))
        except tf.errors.OutOfRangeError:
            print('training is finished...')
            break

    step = 0
    acc = 0
    sess.run(test_init_op)
    while True:
        try:
            step = step + 1
            acc = acc + sess.run(accuracy)
        except tf.errors.OutOfRangeError:
            break
    print('final accuracy', acc / step)
