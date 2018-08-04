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

import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

matplotlib.pyplot.ioff()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    img_shape = x_train[0].shape
    print('Cifar10 images size', img_shape)

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
    drop1 = keras.layers.Dropout(0.2)(dens2)
    output = keras.layers.Dense(10, activation='softmax')(drop1)

    model = keras.Model(img_input, output)
    print(model.summary())

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(train_gen.flow(x_train, y_train, batch_size=64),
                        steps_per_epoch=len(x_train) / 64, epochs=30, verbose=2)

    print(model.evaluate(x_test, y_test))

    # visualize layers
    # successive_outputs = [layer.output for layer in model.layers[1:]]
    # visualization_model = Model(img_input, successive_outputs)
    #
    # x = x_test[1]
    # x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
    #
    # # Rescale by 1/255
    # # x /= 255
    #
    # # Let's run our image through our network, thus obtaining all
    # # intermediate representations for this image.
    # successive_feature_maps = visualization_model.predict(x)
    #
    # # These are the names of the layers, so can have them as part of our plot
    # layer_names = [layer.name for layer in model.layers[1:]]
    #
    # # Now let's display our representations
    # for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    #     if len(feature_map.shape) == 4:
    #         # Just do this for the conv / maxpool layers, not the fully-connected layers
    #         n_features = feature_map.shape[-1]  # number of features in feature map
    #         # The feature map has shape (1, size, size, n_features)
    #         size = feature_map.shape[1]
    #         # We will tile our images in this matrix
    #         display_grid = np.zeros((size, size * n_features))
    #         for i in range(n_features):
    #             # Postprocess the feature to make it visually palatable
    #             x = feature_map[0, :, :, i]
    #             x -= x.mean()
    #             x /= x.std()
    #             x *= 64
    #             x += 128
    #             x = np.clip(x, 0, 255).astype('uint8')
    #             # We'll tile each filter into this big horizontal grid
    #             display_grid[:, i * size: (i + 1) * size] = x
    #         # Display the grid
    #         scale = 20. / n_features
    #         plt.figure(figsize=(scale * n_features, scale))
    #         plt.title(layer_name)
    #         plt.grid(False)
    #         plt.imshow(display_grid, aspect='auto', cmap='viridis')
    #
    # plt.show()
