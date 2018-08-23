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
import scipy.io
import tensorflow as tf

from lesson8 import utils

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783


class VGG(object):
    def __init__(self, input_img):
        utils.download(VGG_DOWNLOAD_LINK, VGG_FILENAME, EXPECTED_BYTES)
        self.vgg_layers = scipy.io.loadmat(VGG_FILENAME)['layers']
        self.input_img = input_img
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    def _weights(self, layer_idx, expected_layer_name):
        """ Return the weights and biases at layer_idx already trained by VGG
        """
        W = self.vgg_layers[0][layer_idx][0][0][2][0][0]
        b = self.vgg_layers[0][layer_idx][0][0][2][0][1]
        layer_name = self.vgg_layers[0][layer_idx][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b.reshape(b.size)

    def conv2d_relu(self, prev_layer, layer_idx, layer_name):
        """ Create a convolution layer with RELU using the weights and
        biases extracted from the VGG model at 'layer_idx'. You should use
        the function _weights() defined above to extract weights and biases.
        _weights() returns numpy arrays, so you have to convert them to TF tensors.
        Don't forget to apply relu to the output from the convolution.
        Inputs:
            prev_layer: the output tensor from the previous layer
            layer_idx: the index to current layer in vgg_layers
            layer_name: the string that is the name of the current layer.
                        It's used to specify variable_scope.
        Hint for choosing strides size:
            for small images, you probably don't want to skip any pixel
        """
        W_, b_ = self._weights(layer_idx, layer_name)
        W, b = tf.convert_to_tensor(W_), tf.convert_to_tensor(b_)
        conv2d = tf.nn.conv2d(prev_layer, W, strides=[1, 1, 1, 1], padding="SAME")
        out = tf.nn.relu(tf.nn.bias_add(conv2d, b))
        setattr(self, layer_name, out)

    def avgpool(self, prev_layer, layer_name):
        """ Create the average pooling layer. The paper suggests that
        average pooling works better than max pooling.

        Input:
            prev_layer: the output tensor from the previous layer
            layer_name: the string that you want to name the layer.
                        It's used to specify variable_scope.
        Hint for choosing strides and kszie: choose what you feel appropriate
        """
        out = tf.nn.avg_pool(prev_layer, [1, 5, 5, 1], [1, 2, 2, 1], padding="SAME")
        setattr(self, layer_name, out)

    def load(self):
        self.conv2d_relu(self.input_img, 0, 'conv1_1')
        self.conv2d_relu(self.conv1_1, 2, 'conv1_2')
        self.avgpool(self.conv1_2, 'avgpool1')
        self.conv2d_relu(self.avgpool1, 5, 'conv2_1')
        self.conv2d_relu(self.conv2_1, 7, 'conv2_2')
        self.avgpool(self.conv2_2, 'avgpool2')
        self.conv2d_relu(self.avgpool2, 10, 'conv3_1')
        self.conv2d_relu(self.conv3_1, 12, 'conv3_2')
        self.conv2d_relu(self.conv3_2, 14, 'conv3_3')
        self.conv2d_relu(self.conv3_3, 16, 'conv3_4')
        self.avgpool(self.conv3_4, 'avgpool3')
        self.conv2d_relu(self.avgpool3, 19, 'conv4_1')
        self.conv2d_relu(self.conv4_1, 21, 'conv4_2')
        self.conv2d_relu(self.conv4_2, 23, 'conv4_3')
        self.conv2d_relu(self.conv4_3, 25, 'conv4_4')
        self.avgpool(self.conv4_4, 'avgpool4')
        self.conv2d_relu(self.avgpool4, 28, 'conv5_1')
        self.conv2d_relu(self.conv5_1, 30, 'conv5_2')
        self.conv2d_relu(self.conv5_2, 32, 'conv5_3')
        self.conv2d_relu(self.conv5_3, 34, 'conv5_4')
        self.avgpool(self.conv5_4, 'avgpool5')
