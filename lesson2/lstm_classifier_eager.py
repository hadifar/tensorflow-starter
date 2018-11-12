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

tf.enable_eager_execution()

file_name = 'sentiment.csv'
# read csv data
csv_dataset = tf.data.experimental.CsvDataset(file_name,
                                              record_defaults=["", ""],
                                              header=True,
                                              select_cols=[1, 2])
# change label-sentence to sentence-label
csv_dataset = csv_dataset.map(lambda x, y: (y, x))
# convert string labels into numeric value
csv_dataset = csv_dataset.map(lambda x, y: (x, tf.cond(tf.equal(y, 'positive'), lambda: 1, lambda: 0)))
csv_dataset = csv_dataset.map(lambda x, y: (x, tf.one_hot(y, 2)))

# convert string words into IDs
table = tf.contrib.lookup.index_table_from_file(vocabulary_file="vocab.txt", num_oov_buckets=1)
csv_dataset = csv_dataset.map(lambda x, y: (tf.string_split([x]).values, y))
csv_dataset = csv_dataset.map(lambda x, y: (tf.cast(table.lookup(x), tf.int32), y))

# add padding to batches
csv_dataset = csv_dataset.shuffle(10).padded_batch(batch_size=3,
                                                   padded_shapes=(tf.TensorShape([None, ]),
                                                                  tf.TensorShape([None, ])))


class LSTM(tf.keras.Model):
    def __init__(self, vocab_size=100, hid_size=100):
        super(LSTM, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, 128)
        self.dense1 = tf.keras.layers.LSTM(units=hid_size)
        self.pred = tf.keras.layers.Dense(2)

    def __call__(self, inputs):
        emb = self.embed(inputs)
        d1 = self.dense1(emb)
        outputs = self.pred(d1)
        return outputs


def loss_function(labels, preds):
    return tf.losses.sigmoid_cross_entropy(labels, preds)


optimizer = tf.train.AdamOptimizer(0.001)
model = LSTM()

for epoch in range(10000):
    for batch, (sentence, label) in enumerate(csv_dataset):
        with tf.GradientTape() as tape:
            pred = model(sentence)
            loss = loss_function(label, pred)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    if epoch % 24 == 0:
        print('Epoch {}: loss: {:.4f}'.format(epoch + 1, loss))
