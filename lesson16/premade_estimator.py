# -*- coding: utf-8 -*-
#
# Copyright 2019 Amir Hadifar. All Rights Reserved.
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

"""
The following code taken from: https://github.com/tensorflow
"""

import tensorflow as tf

import lesson16.iris_data as iris_data

# Feature columns describe how to use the input.
my_feature_columns = []
for key in iris_data.train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3,
    # The directory which model saved
    model_dir='./tmp'
)

# Train the Model.
classifier.train(input_fn=lambda: iris_data.train_input_fn(), steps=1000)

# Evaluate the model.
eval_result = classifier.evaluate(input_fn=lambda: iris_data.eval_input_fn())

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
