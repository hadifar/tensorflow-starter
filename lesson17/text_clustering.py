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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.manifold import TSNE

encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
sentences = [
    'this is good',
    'this is perfectly fine',
    'good day will come!',
    'these days are perfect',

    'driving license is expired',
    'driving car carelessly is dangerous',
    'drive car or walk down to the street',
    'run , ride , drive',

    'hello world',
    'hello world wide web',
    'hello machine learning',
    'hello deep learning'
]
k = 3
max_iter = 1


def extract_feature(session, data):
    return session.run(encoder(data))


def initial_cluster_centroids(X, k):
    return X[0:k, :]


def assign_cluster(X, centroids):
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    return mins


def recompute_centroids(X, Y):
    sums = tf.unsorted_segment_sum(X, Y, k)
    counts = tf.unsorted_segment_sum(tf.ones_like(X), Y, k)
    return sums / counts


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    X = extract_feature(sess, sentences)
    centroids = initial_cluster_centroids(X, k)
    print(np.array(X).shape)

    i = 0
    while i < max_iter:
        i += 1
        Y = assign_cluster(X, centroids)
        centroids = sess.run(recompute_centroids(X, Y))
        if i % 10 == 0:
            print("iteration: {}".format(i))

    labels = sess.run(assign_cluster(X, centroids))
    X = TSNE(n_components=2).fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.show()

    all_sentence = []
    for i, sent in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            normalize_a = tf.nn.l2_normalize(tf.squeeze(X[i]), 0)
            normalize_b = tf.nn.l2_normalize(tf.squeeze(X[j]), 0)
            cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
            print('sent1 :', sent)
            print('sent2 :', sent2)
            print('cosine', sess.run(cos_similarity))