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

# our data
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
# normilize our data
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

# initilization
a, b = 0, 0
# number of iteration
num_epoch = 10000
learning_rate = 1e-3
# our main loop
for e in range(num_epoch):
    y_pred = a * X + b  # our line equation
    grad_a = (y_pred - y).dot(X)  # calculate gradient for a
    grad_b = (y_pred - y).sum()  # calculate gradient for b
    # Update parameters.
    a = a - learning_rate * grad_a
    b = b - learning_rate * grad_b
print(a, b)
