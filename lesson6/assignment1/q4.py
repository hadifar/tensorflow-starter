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
from tensorboard.plugins import projector

from lesson6.assignment1 import word2vec_utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('DOWNLOAD_URL', 'http://mattmahoney.net/dc/text8.zip', 'download link for text8')
tf.flags.DEFINE_integer('EXPECTED_BYTES', 31344016, 'expected byte to download')
tf.flags.DEFINE_float('LEARNING_RATE', 0.01, 'learning rate')
tf.flags.DEFINE_integer('EMBED_SIZE', 128, 'embedding dimension')
tf.flags.DEFINE_integer('VOCAB_SIZE', 50000, 'size of corpus vocabulary')
tf.flags.DEFINE_integer('NEG_SAMPLES', 64, 'size of negative examples')
tf.flags.DEFINE_integer('EPOCH', 100000, 'number of epochs')
tf.flags.DEFINE_integer('BATCH_SIZE', 128, 'batch size')
tf.flags.DEFINE_integer('SKIP_WINDOW', 1, 'the context window')
tf.flags.DEFINE_string('VISUAL_FLD', 'visualization', 'visualization folder')
tf.flags.DEFINE_integer('SKIP_STEP', 5000, 'show evaluation every step')


class CBow(object):

    def __init__(self, dataset, vocab_size, embed_size, neg_samples, learning_rate, skip_step):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.embed_dim = embed_size
        self.neg_samples = neg_samples
        self.learning_rate = learning_rate
        self.skip_step = skip_step
        self.global_step = tf.get_variable(name='global_step', trainable=False, initializer=0)

    def _import_data(self):
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()

    def _create_embedding(self):
        with tf.name_scope('embedding'):
            self.embed_matrix = tf.get_variable(name='embed',
                                                initializer=tf.random_uniform([self.vocab_size, self.embed_dim],
                                                                              minval=-1,
                                                                              maxval=1))
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words)

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.nce_wights = tf.get_variable(name='nce_weights',
                                              initializer=tf.truncated_normal(shape=[self.vocab_size, self.embed_dim],
                                                                              stddev=1.0 / self.embed_dim ** 0.5))

            self.nce_biases = tf.get_variable(name='nce_biases',
                                              initializer=tf.zeros([self.vocab_size]))

            self.loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_wights, self.nce_biases, self.target_words, self.embed,
                                                      num_sampled=self.neg_samples, num_classes=self.vocab_size))

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._import_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, epoch):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer)

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            writer = tf.summary.FileWriter('graphs/word2vec/lr' + str(self.learning_rate), sess.graph)
            initial_step = self.global_step.eval()
            total_loss = 0.0

            for index in range(initial_step, initial_step + epoch):
                try:
                    _, loss_batch, summary = sess.run(
                        [self.optimizer, self.loss, self.summary_op])

                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print('Average loss at step {}: {:5.1f}'.format(index, total_loss / self.skip_step))
                        total_loss = 0.0
                        saver.save(sess, 'checkpoints/skip-gram', index)

                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)

            writer.close()

    def visualization(self, num_visualize, visual_fld):
        """ run "'tensorboard --logdir='visualization'" to see the embeddings """

        # create the list of num_variable most common words to visualize
        word2vec_utils.most_common_words(visual_fld, num_visualize)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            final_embed_matrix = sess.run(self.embed_matrix)

            # you have to store embeddings in a new variable
            embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter(visual_fld)

            # add embedding to the config file
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # link this tensor to its metadata file, in this case the first NUM_VISUALIZE words of vocab
            embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

            # saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, os.path.join(visual_fld, 'model.ckpt'), 1)


def main(_):
    def gen():
        yield from word2vec_utils.batch_gen(FLAGS.DOWNLOAD_URL, FLAGS.EXPECTED_BYTES, FLAGS.VOCAB_SIZE,
                                            FLAGS.BATCH_SIZE, FLAGS.SKIP_WINDOW, FLAGS.VISUAL_FLD)

    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32),
                                             (tf.TensorShape([FLAGS.BATCH_SIZE]),
                                              tf.TensorShape([FLAGS.BATCH_SIZE, 1])))

    model = CBow(dataset, FLAGS.VOCAB_SIZE, FLAGS.EMBED_SIZE, FLAGS.NEG_SAMPLES, FLAGS.LEARNING_RATE, FLAGS.SKIP_STEP)

    model.build_graph()

    model.train(FLAGS.EPOCH)

    model.visualization(3000, FLAGS.VISUAL_FLD)


if __name__ == '__main__':
    tf.app.run()
