# -*- coding: utf-8 -*-
#
# Copyright 2019 Amir Hadifar. All Rights Reserved.
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

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_eval = x_train[0:10000]
y_eval = y_train[0:10000]

x_train = x_train[10000:]
y_train = y_train[10000:]


def neural_net_model(inputs, mode):
    with tf.variable_scope('ConvModel'):
        inputs = inputs / 255
        input_layer = tf.reshape(inputs, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=20,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=40,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        flatten = tf.reshape(pool2, [-1, 4 * 4 * 40])
        dense1 = tf.layers.dense(inputs=flatten, units=256, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
        dense2 = tf.layers.dense(inputs=dropout, units=10)
        return dense2


def model_fn(features, labels, mode):
    logits = neural_net_model(features, mode)
    class_prediction = tf.argmax(logits, axis=-1)
    preds = class_prediction

    loss = None
    train_op = None
    eval_metric_ops = {}

    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.int32), logits=logits))

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=preds,
                name='accuracy')
        }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=class_prediction,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def train_input_fn():
    return tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(1000, reshuffle_each_iteration=True) \
        .repeat(count=None) \
        .batch(128)


def eval_input_fn():
    return tf.data.Dataset.from_tensor_slices((x_eval, y_eval)).batch(128)


def test_input_fn():
    return tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)


def main(_):
    config = tf.estimator.RunConfig(model_dir='./tmp/',
                                    save_summary_steps=100,
                                    log_step_count_steps=100,
                                    save_checkpoints_steps=500)

    model_estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)
    model_estimator.train(train_input_fn, max_steps=5000)
    result = model_estimator.evaluate(test_input_fn)
    print(result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
