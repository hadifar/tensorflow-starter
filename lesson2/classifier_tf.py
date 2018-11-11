# # # -*- coding: utf-8 -*-
# # #
# # # Copyright 2018 Amir Hadifar. All Rights Reserved.
# # #
# # # Licensed under the Apache License, Version 2.0 (the "License");
# # # you may not use this file except in compliance with the License.
# # # You may obtain a copy of the License at
# # #
# # #     http://www.apache.org/licenses/LICENSE-2.0
# # #
# # # Unless required by applicable law or agreed to in writing, software
# # # distributed under the License is distributed on an "AS IS" BASIS,
# # # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # # See the License for the specific language governing permissions and
# # # limitations under the License.
# # ==============================================================================
import tensorflow as tf

tf.enable_eager_execution()

file_name = '../lesson2/sentiment.csv'

# read csv data
csv_dataset = tf.data.experimental.CsvDataset(file_name,
                                              record_defaults=["", ""],
                                              header=True,
                                              select_cols=[1, 2])
# convert string labels into numeric value
csv_dataset = csv_dataset.map(lambda x, y: (tf.cond(tf.equal(x, 'positive'), lambda: 1, lambda: 0), y))

# convert string words into IDs
table = tf.contrib.lookup.index_table_from_file(vocabulary_file="vocab.txt")
csv_dataset = csv_dataset.map(lambda x, y: (x, tf.string_split([y]).values))
csv_dataset = csv_dataset.map(lambda x, y: (x, table.lookup(y)))


class MLP(tf.keras.Model):
    def __init__(self, vocab_size=1000, embed_dim=128):
        super(MLP, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pred = tf.keras.layers.Dense(1)

    def __call__(self, inputs):
        emb = self.embed(inputs)
        emb = tf.reduce_mean(emb, axis=0)
        emb = tf.expand_dims(emb, axis=0)
        outputs = self.pred(emb)
        return outputs


def loss_function(labels, preds):
    labels = tf.reshape(labels, (1, 1))
    preds = tf.round(preds)
    return tf.losses.softmax_cross_entropy(labels, preds)


optimizer = tf.train.AdamOptimizer()
model = MLP()

for epoch in range(1000):

    for batch, (label, sentence) in enumerate(csv_dataset):
        with tf.GradientTape() as tape:
            pred = model(sentence)
            loss = loss_function(label, pred)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if batch % 10 == 0:
            print('Epoch {}: loss: {:.4f}'.format(epoch + 1, loss))

# import pandas as pd
#
# df = pd.read_csv('../lesson2/sentiment.csv', sep=',', names=['id', 'label', 'sentence'], skiprows=1)
# df = df['sentence']
# vocab = set()
# for item in df.values.tolist():
#     vocab.update(item.split())
#
# with open('vocab.txt', mode='w') as out_file:
#     for item in vocab:
#         out_file.write(item + '\n')
