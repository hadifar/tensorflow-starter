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
from tensorflow.keras.applications import VGG19
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    img_shape = x_train[0].shape

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    train_gen = ImageDataGenerator(featurewise_std_normalization=True,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

    test_gen = ImageDataGenerator(featurewise_std_normalization=True)

    train_gen.fit(x_train)
    test_gen.fit(x_test)

    inp = keras.layers.Input(shape=(32, 32, 3), name='image_input')

    vgg_model = VGG19(weights='imagenet', include_top=False)
    for layer in vgg_model.layers:
        layer.trainable = False

    vgg_out = vgg_model(inp)

    x = keras.layers.Flatten(name='flatten')(vgg_out)
    x = keras.layers.Dense(512, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(512, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(10, activation='softmax', name='predictions')(x)

    # Create your own model
    my_model = keras.models.Model(inputs=inp, outputs=x)

    # In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training

    my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    my_model.summary()

    my_model.fit_generator(train_gen.flow(x_train, y_train, batch_size=64),
                           steps_per_epoch=len(x_train) / 64, epochs=30, verbose=2)

    print(my_model.evaluate_generator(test_gen.flow(x_test, y_test, batch_size=64)))
