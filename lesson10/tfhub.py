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
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import metrics

# download universal-sentence-encoder model from tf-hub module
url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(url)

# read dataset (download it from Kaggle.com)
data = pd.read_csv('spam.csv', encoding='latin-1')

# drop unwanted columns, and rename the column name appropriately.
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1": "label", "v2": "text"})
# change string label to integer label
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

y = list(data['label_num'])
x = list(data['text'])

# split data into test and train
x_train = np.asarray(x[:5000])
y_train = np.asarray(y[:5000])

x_test = np.asarray(x[5000:])
y_test = np.asarray(y[5000:])


# build lambda layer for tf-hub
def UniversalEmbedding(sen):
    return embed(tf.squeeze(tf.cast(sen, tf.string)))


# build Keras model
input_text = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
embedding = tf.keras.layers.Lambda(UniversalEmbedding, output_shape=(512,))(input_text)
dense = tf.keras.layers.Dense(256, activation='relu')(embedding)
pred = tf.keras.layers.Dense(2, activation='softmax')(dense)
model = tf.keras.models.Model(inputs=[input_text], outputs=pred)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# train model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    model.fit(x_train, y_train, epochs=1, batch_size=128)

    # prediction
    predicts = model.predict(x_test, batch_size=128)

# confusion matrix
predicts = predicts.argmax(axis=1)
metrics.confusion_matrix(y_test, predicts)
print(metrics.classification_report(y_test, predicts))
