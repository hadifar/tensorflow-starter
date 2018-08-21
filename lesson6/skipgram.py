import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from lesson6 import word2vec_utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('DOWNLOAD_URL', 'http://mattmahoney.net/dc/text8.zip', 'download link for text8')
tf.flags.DEFINE_integer('EXPECTED_BYTES', 31344016, 'expected byte to download')
tf.flags.DEFINE_integer('NUM_VISUALIZE', 3000, 'number of tokens to visualize')
tf.flags.DEFINE_integer('VOCAB_SIZE', 50000, 'size of vocabulary')
tf.flags.DEFINE_integer('BATCH_SIZE', 128, 'size of batch')
tf.flags.DEFINE_integer('EMBED_SIZE', 128, 'dimension of the word embedding vectors')
tf.flags.DEFINE_integer('SKIP_WINDOW', 1, 'the context window')
tf.flags.DEFINE_float('NUM_SAMPLED', 64, 'number of negative examples to sample')
tf.flags.DEFINE_float('LEARNING_RATE', 1.0, 'learning rate')
tf.flags.DEFINE_integer('NUM_TRAIN_STEPS', 100000, 'max steps of training')
tf.flags.DEFINE_string('VISUAL_FLD', 'visualization', 'visualization folder')
tf.flags.DEFINE_integer('SKIP_STEP', 5000, 'show evaluation every step')


class SkipGram(object):

    def __init__(self, dataset, vocab_size, embed_dim, neg_sample, learning_rate, skip_step=5000):
        self.neg_sample = neg_sample
        self.vocab_size = vocab_size
        self.dataset = dataset
        self.embed_dim = embed_dim
        self.lr = learning_rate
        self.skip_step = skip_step

    def _import_data(self):
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()

    def _create_embedding(self):
        """ Step 2: in word2vec, it's actually the weights that we care about """
        with tf.name_scope('embedding'):
            self.embed_matrix = tf.get_variable(name='embed',
                                                initializer=tf.random_uniform([self.vocab_size, self.embed_dim],
                                                                              minval=-1,
                                                                              maxval=1))
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embedding')

    def _create_loss(self):
        """ Step 3 + 4: define the inference + the loss function """
        with tf.name_scope('loss'):
            self.nec_weight = tf.get_variable(name='nce_weight',
                                              initializer=tf.truncated_normal([self.vocab_size, self.embed_dim],
                                                                              stddev=1.0 / self.embed_dim ** 0.5))
            self.nec_bias = tf.get_variable(name='nce_bias',
                                            initializer=tf.zeros([self.vocab_size]))

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nec_weight,
                                                      biases=self.nec_bias,
                                                      labels=self.target_words,
                                                      inputs=self.embed,
                                                      num_classes=self.vocab_size,
                                                      num_sampled=self.neg_sample))

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
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

            total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter('graphs/word2vec/lr' + str(self.lr), sess.graph)

            for index in range(epoch):
                try:
                    _, loss_batch, summary = sess.run([self.optimizer, self.loss, self.summary_op])
                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print('Average loss at step {}: {:5.1f}'.format(index, total_loss / self.skip_step))
                        total_loss = 0.0
                        saver.save(sess, 'checkpoints/skip-gram', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)

            writer.close()

    def visualization(self, visual_fld, num_visualize):
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

    dataset = tf.data.Dataset.from_generator(gen,
                                             (tf.int32, tf.int32),
                                             (tf.TensorShape([FLAGS.BATCH_SIZE]),
                                              tf.TensorShape([FLAGS.BATCH_SIZE, 1])))

    skipgram = SkipGram(dataset=dataset,
                        vocab_size=FLAGS.VOCAB_SIZE,
                        embed_dim=FLAGS.EMBED_SIZE,
                        neg_sample=FLAGS.NUM_SAMPLED,
                        learning_rate=FLAGS.LEARNING_RATE)

    skipgram.build_graph()
    skipgram.train(FLAGS.NUM_TRAIN_STEPS)
    skipgram.visualization(FLAGS.VISUAL_FLD, FLAGS.NUM_VISUALIZE)


if __name__ == '__main__':
    tf.app.run()
