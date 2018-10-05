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
import os

import tensorflow as tf

from lesson12 import general_utils


class PairSimilarity(object):

    def __init__(self, data, embed_weights, gru_size=128, vocab_size=100000):
        self.data = data
        self.gru_size = gru_size
        self.vocab_size = vocab_size
        self.global_step = tf.get_variable(name='global_step', initializer=0, trainable=False)
        self.embed = tf.get_variable(name='embed', shape=embed_weights.shape, trainable=False,
                                     initializer=tf.constant_initializer(embed_weights))

    def _create_input(self):
        train, test, dev = self.data
        (q1_dev, q2_dev, label_dev) = dev
        (q1_test, q2_test, label_test) = test
        (q1_train, q2_train, label_train) = train

        with tf.name_scope('data'):
            train_dataset = tfdata_generator(q1_train, q2_train, label_train, is_training=True)
            dev_dataset = tfdata_generator(q1_dev, q2_dev, label_dev, is_training=False)
            test_dataset = tfdata_generator(q1_test, q2_test, label_test, is_training=False)

            # create one iterator and initialize it with different datasets
            self.iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                            train_dataset.output_shapes)

            self.train_init = self.iterator.make_initializer(train_dataset)  # initializer for train_data
            self.test_init = self.iterator.make_initializer(test_dataset)  # initializer for train_data

            self.sent1, self.sent2, self.label = self.iterator.get_next()

    def _create_model(self):
        with tf.name_scope('model'):
            self.left_out = self._create_tower(self.sent1, reuse=False)
            self.right_out = self._create_tower(self.sent2, reuse=True)

            sub = tf.subtract(self.left_out, self.right_out)
            sum = tf.add(self.left_out, self.right_out)
            # dot = tf.tensordot(self.left_out, self.right_out, axes=1)

            concat = tf.concat([sub, sum], axis=1)
            concat = tf.layers.dropout(concat)
            dense = tf.layers.dense(inputs=concat, units=128, activation='relu')

            self.logit = tf.layers.dense(inputs=dense, units=1)
            self.pred = tf.nn.sigmoid(self.logit)

            is_correct = tf.equal(tf.round(self.pred), self.label)
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def _create_tower(self, inp, reuse=False):
        with tf.variable_scope('embedding', reuse=reuse):
            vector = tf.nn.embedding_lookup(params=self.embed, ids=inp)

        with tf.variable_scope('rnn', reuse=reuse):
            fw_cell = [tf.nn.rnn_cell.GRUCell(self.gru_size, reuse=tf.get_variable_scope().reuse)]
            fw_cells = tf.nn.rnn_cell.MultiRNNCell(fw_cell)

            bw_cell = [tf.nn.rnn_cell.GRUCell(self.gru_size, reuse=tf.get_variable_scope().reuse)]
            bw_cells = tf.nn.rnn_cell.MultiRNNCell(bw_cell)

            (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(fw_cells,
                                                                        bw_cells,
                                                                        inputs=vector,
                                                                        dtype=tf.float32)
            output = tf.concat([fw_output, bw_output], axis=2)
            output = output[:, -1, :]

        with tf.variable_scope('dense', reuse=reuse):
            dense1 = tf.layers.dense(inputs=output, units=128, activation='relu', reuse=tf.get_variable_scope().reuse)
            return dense1

    def _create_loss(self):
        with tf.name_scope('loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit)
            self.loss = tf.reduce_mean(loss)

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

    def _create_summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_input()
        self._create_model()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

    def train(self, epoch=2500):

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.train_init)

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            writer = tf.summary.FileWriter(logdir='graphs/pair_sim/', graph=sess.graph)

            initial_step = self.global_step.eval()
            total_loss = 0.0
            total_acc = 0
            for index in range(initial_step, initial_step + epoch):
                try:
                    _, acc_batch, loss_batch, summary, step = sess.run(
                        [self.optimizer, self.accuracy, self.loss, self.summary_op, self.global_step])

                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    total_acc += acc_batch
                    if (index + 1) % 128 == 0:
                        print('step {}: loss: {:5.2f} -- acc: {:5.2f}'.format(index,
                                                                              total_loss / 128,
                                                                              total_acc / 128))
                        total_loss = 0.0
                        total_acc = 0.0
                        saver.save(sess, 'checkpoints/siamese', index)

                except tf.errors.OutOfRangeError:
                    sess.run(self.train_init)

            writer.close()

    def inference(self):
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            sess.run(self.test_init)

            total_correct_preds = 0
            try:
                while True:
                    accuracy_batch = sess.run(self.accuracy)
                    total_correct_preds += accuracy_batch
            except tf.errors.OutOfRangeError:
                pass
            print('Accuracy {0}'.format(total_correct_preds / 10000))


def tfdata_generator(sent1, sent2, labels, is_training):
    dataset = tf.data.Dataset.from_tensor_slices((sent1, sent2, labels))
    if is_training:
        dataset = dataset.shuffle(1000)  # depends on sample size

    dataset = dataset.batch(64)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def main():
    embedding, word_index, train, test, dev = general_utils.load_data()
    nb_vocab = min(len(word_index), 100000) + 1
    model = PairSimilarity(data=[train, test, dev], embed_weights=embedding, gru_size=128, vocab_size=nb_vocab)
    model.build_graph()
    model.train()
    model.inference()


if __name__ == '__main__':
    tf.app.run()
