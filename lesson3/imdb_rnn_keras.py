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
from tensorflow import keras
from tensorflow.python.keras.models import load_model

print('load data')
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
word_index = keras.datasets.imdb.get_word_index()

print('preprocessing...')
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=256)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=256)

x_val = x_train[:10000]
y_val = y_train[:10000]

x_train = x_train[10000:]
y_train = y_train[10000:]

print('build model')
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('train model')
model.fit(x_train,
          y_train,
          epochs=5,
          batch_size=512,
          validation_data=(x_val, y_val),
          verbose=1)

print('save trained model...')
model.save('sentiment_keras.h5')
del model

print('load model...')
model = load_model('sentiment_keras.h5')

print('evaluation')
evaluation = model.evaluate(x_test, y_test, batch_size=512)
print('Loss:', evaluation[0], 'Accuracy:', evaluation[1])

sample = 'this is new sentence and this bad bad sentence'
sample_label = 0
inps = [word_index[word] for word in sample.split() if word in word_index]
inps = keras.preprocessing.sequence.pad_sequences([inps], maxlen=256)
print('Accuracy:', model.evaluate(inps, [sample_label], batch_size=1)[1])
print('Sentiment score: {}'.format(model.predict(inps)[0][0]))
