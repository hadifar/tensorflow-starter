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
import os

import tensorflow as tf
from tensor2tensor import problems
from tensor2tensor.utils import metrics
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# Setup some directories
data_dir = os.path.expanduser("../lesson16/t2t/data")
tmp_dir = os.path.expanduser("../lesson16/t2t/tmp")
train_dir = os.path.expanduser("../lesson16/t2t/train")
checkpoint_dir = os.path.expanduser("../lesson16/t2t/checkpoints")
tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
tf.gfile.MakeDirs(train_dir)
tf.gfile.MakeDirs(checkpoint_dir)
gs_data_dir = "../lesson16/t2t/tensor2tensor-data"
gs_ckpt_dir = "../lesson16/t2t/tensor2tensor-checkpoints/"

# Create your own model

# Fetch the MNIST problem
mnist_problem = problems.problem("image_mnist")
# The generate_data method of a problem will download data and process it into
# a standard format ready for training and evaluation.
mnist_problem.generate_data(data_dir, tmp_dir)


class MySimpleModel(t2t_model.T2TModel):

    def body(self, features):
        inputs = features["inputs"]
        filters = self.hparams.hidden_size
        h1 = tf.layers.conv2d(inputs, filters,
                              kernel_size=(5, 5), strides=(2, 2))
        h2 = tf.layers.conv2d(tf.nn.relu(h1), filters,
                              kernel_size=(5, 5), strides=(2, 2))
        return tf.layers.conv2d(tf.nn.relu(h2), filters,
                                kernel_size=(3, 3))


hparams = trainer_lib.create_hparams("basic_1", data_dir=data_dir, problem_name="image_mnist")
hparams.hidden_size = 64
model = MySimpleModel(hparams, Modes.TRAIN)


# Prepare for the training loop

# In Eager mode, opt.minimize must be passed a loss function wrapped with
# implicit_value_and_gradients
@tfe.implicit_value_and_gradients
def loss_fn(features):
    _, losses = model(features)
    return losses["training"]


# Setup the training data
BATCH_SIZE = 128
mnist_train_dataset = mnist_problem.dataset(Modes.TRAIN, data_dir)
mnist_train_dataset = mnist_train_dataset.repeat(None).batch(BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()

# Train
NUM_STEPS = 500

for count, example in enumerate(tfe.Iterator(mnist_train_dataset)):
    example["targets"] = tf.reshape(example["targets"], [BATCH_SIZE, 1, 1, 1])  # Make it 4D.
    loss, gv = loss_fn(example)
    optimizer.apply_gradients(gv)

    if count % 50 == 0:
        print("Step: %d, Loss: %.3f" % (count, loss.numpy()))
    if count >= NUM_STEPS:
        break

model.set_mode(Modes.EVAL)
mnist_eval_dataset = mnist_problem.dataset(Modes.EVAL, data_dir)

# Create eval metric accumulators for accuracy (ACC) and accuracy in
# top 5 (ACC_TOP5)
metrics_accum, metrics_result = metrics.create_eager_metrics(
    [metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5])

for count, example in enumerate(tfe.Iterator(mnist_eval_dataset)):
    if count >= 200:
        break

    # Make the inputs and targets 4D
    example["inputs"] = tf.reshape(example["inputs"], [1, 28, 28, 1])
    example["targets"] = tf.reshape(example["targets"], [1, 1, 1, 1])

    # Call the model
    predictions, _ = model(example)

    # Compute and accumulate metrics
    metrics_accum(predictions, example["targets"])

# Print out the averaged metric values on the eval data
for name, val in metrics_result().items():
    print("%s: %.2f" % (name, val))
