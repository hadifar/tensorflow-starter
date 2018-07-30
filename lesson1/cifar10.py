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

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train, x_test = x_train / 255, x_test / 255
    img_shape = x_train[0].shape
    print('Cifar10 images size', img_shape)

    img_input = keras.layers.Input(shape=img_shape)
    conv1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(img_input)
    pol1 = keras.layers.MaxPooling2D(2)(conv1)
    conv2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(pol1)
    pol2 = keras.layers.MaxPooling2D(2)(conv2)
    conv3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pol2)
    pol3 = keras.layers.MaxPooling2D(2)(conv3)
    flatten = keras.layers.Flatten()(pol3)
    dens1 = keras.layers.Dense(512, activation='relu')(flatten)
    dens2 = keras.layers.Dense(128, activation='relu')(dens1)
    output = keras.layers.Dense(10, activation='softmax')(dens2)

    model = keras.Model(img_input, output)
    print(model.summary())

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20)
    print(model.evaluate(x_test, y_test))
