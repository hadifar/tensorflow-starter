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
from tensorflow import keras


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


(train_data, train_labels), (test_data, test_labels) = keras.datasets.boston_housing.load_data()

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print(train_data.shape)
print(test_data.shape)
print(50 * '-')

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(1))

optimizer = keras.optimizers.Nadam(clipvalue=5)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model.compile(optimizer, loss='mse', metrics=['mae'])
history = model.fit(train_data, train_labels, batch_size=128, epochs=500, validation_split=0.2, callbacks=[early_stop])

plot_history(history)
prediction = model.evaluate(test_data, test_labels)
print(prediction)
