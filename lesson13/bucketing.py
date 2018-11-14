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
batch_size = 8
num_buckets = 4
output_buffer_size = batch_size * 1000
reshuffle_each_iteration = True
random_seed = 1023
src_max_len = 12

# read csv data
csv_dataset = tf.data.experimental.CsvDataset(file_name,
                                              record_defaults=["", ""],
                                              header=True,
                                              select_cols=[1, 2])
# change label-sentence to sentence-label
csv_dataset = csv_dataset.map(lambda x, y: (y, x))

# filter labels where not in {positive, negative}
csv_dataset = csv_dataset.filter(lambda x, y: tf.logical_or(tf.equal(y, 'positive'), tf.equal(y, 'negative')))

# convert string labels into numeric value
csv_dataset = csv_dataset.map(lambda x, y: (x, tf.where(tf.equal(y, 'positive'), 1, 0)))

# convert labels to one_hot encoding
csv_dataset = csv_dataset.map(lambda x, y: (x, tf.one_hot(y, depth=2)))

# shuffle data
csv_dataset = csv_dataset.shuffle(output_buffer_size, random_seed, reshuffle_each_iteration)

# convert string words into IDs
table = tf.contrib.lookup.index_table_from_file(vocabulary_file="vocab.txt", num_oov_buckets=1)
csv_dataset = csv_dataset.map(lambda x, y: (tf.string_split([x]).values, y))
csv_dataset = csv_dataset.map(lambda x, y: (tf.cast(table.lookup(x), tf.int32), y))

# filter zero length input sequences and labels
csv_dataset = csv_dataset.filter(lambda x, y: tf.size(x) > 0)

# calculate length of sequences
csv_dataset = csv_dataset.map(lambda x, y: (x, y, tf.size(x)))


def key_func(unused_1, unused_2, src_len):
    # Calculate bucket_width by maximum source sequence length.
    # Pairs with length [0, bucket_width) go to bucket 0, length
    # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
    # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
    if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
    else:
        bucket_width = 10

    # Bucket sentence pairs by the length of their source sentence and target
    # sentence.
    bucket_id = src_len // bucket_width
    return tf.to_int64(tf.minimum(num_buckets, bucket_id))


def reduce_func(unused_key, windowed_data):
    # add padding to batches
    return windowed_data.padded_batch(batch_size=batch_size,
                                      padded_shapes=(tf.TensorShape([None, ]),
                                                     tf.TensorShape([None, ]),
                                                     tf.TensorShape([])))


csv_dataset = csv_dataset.apply(tf.data.experimental.group_by_window(
    key_func=key_func,
    reduce_func=reduce_func, window_size=batch_size))

for i, c in enumerate(csv_dataset):
    print(c[0].numpy())
    print(20 * '*')
