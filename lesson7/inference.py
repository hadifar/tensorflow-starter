import os

import numpy as np
import tensorflow as tf

from lesson7.char_rnn import CharRNN
from lesson7.utils import TextReader, pick_top_n

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state')
tf.flags.DEFINE_integer('num_layers', 2, 'number of the lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', '../lesson7/default/converter.pkl', 'converter path')
tf.flags.DEFINE_string('checkpoint_path', '../lesson7/default/model', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '',
                       'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 1000, 'max length to generate')


def main(_):
    converter = TextReader(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(
            FLAGS.checkpoint_path)

    char_rnn = CharRNN(
        converter.vocab_size,
        sample=True,
        lstm_size=FLAGS.lstm_size,
        num_layers=FLAGS.num_layers,
        use_embedding=FLAGS.use_embedding,
        embedding_size=FLAGS.embedding_size)

    start = converter.text_to_arr(FLAGS.start_string)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_path)
        samples = [c for c in start]
        new_state = sess.run(char_rnn.initial_state)
        preds = np.ones((converter.vocab_size,))

        for c in start:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed_dict = {
                char_rnn.inputs: x,
                char_rnn.keep_prob: 1,
                char_rnn.initial_state: new_state
            }
            preds, new_state = sess.run(
                [char_rnn.prediction, char_rnn.final_state],
                feed_dict=feed_dict)

        c = pick_top_n(preds, converter.vocab_size)

        samples.append(c)

        for i in range(FLAGS.max_length):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed_dict = {
                char_rnn.inputs: x,
                char_rnn.keep_prob: 1,
                char_rnn.initial_state: new_state
            }
            preds, new_state = sess.run(
                [char_rnn.prediction, char_rnn.final_state],
                feed_dict=feed_dict)
            c = pick_top_n(preds, converter.vocab_size)
            # 添加生成的新字符
            samples.append(c)

        samples = np.array(samples)
        print(converter.arr_to_text(samples))


if __name__ == '__main__':
    tf.app.run()
