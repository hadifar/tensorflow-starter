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

from lesson5.vgg import VGG19

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = x_train.astype(np.float32), tf.keras.utils.to_categorical(y_train.astype(np.float32))
    x_test, y_test = x_test.astype(np.float32), tf.keras.utils.to_categorical(y_test.astype(np.float32))

    with tf.name_scope('data'):
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.shuffle(len(x_train))
        train_data = train_data.batch(128)

        test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_data = test_data.shuffle(len(x_test))
        test_data = test_data.batch(128)

        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)

        X, Y = iterator.get_next()

        train_init = iterator.make_initializer(train_data)  # initializer for train_data
        test_init = iterator.make_initializer(test_data)  # initializer for train_data

    global_step = tf.get_variable(name='global_step', initializer=0, trainable=False)

    # X = tf.reshape(X, shape=[-1, 32, 32, 3])
    with tf.name_scope('vgg'):
        vgg = VGG19()
        vgg_out = vgg.load(X)

    with tf.name_scope('my_model'):
        d1 = tf.layers.dense(vgg_out, 100, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 100, activation=tf.nn.relu)
        logit = tf.layers.dense(d2, 10)
        pred = tf.nn.softmax(logit)

    is_correct = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logit))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss, global_step=global_step)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(train_init)

        total_loss = 0.0

        writer = tf.summary.FileWriter("transfer", sess.graph)

        for index in range(25000):
            try:
                _, loss_batch, step = sess.run([train_step, loss, global_step])
                total_loss += loss_batch
                if (index + 1) % 100 == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / 100))
                    print('Average acc at step {}: {:5.1f}'.format(index, sess.run(accuracy)))
                    total_loss = 0.0
            except tf.errors.OutOfRangeError:
                sess.run(train_init)

            # test the model
        sess.run(test_init)  # drawing samples from test_data
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch = sess.run(accuracy)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy {0}'.format(total_correct_preds / len(x_test)))
        writer.close()
