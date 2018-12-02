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
import tensorflow as tf

v1 = tf.Variable(8., name="v1")
v2 = tf.Variable(1., name="v2")
a = tf.add(v1, v2)

ckpnt = tf.train.Checkpoint(my=v1)
# status = ckpnt.restore(tf.train.latest_checkpoint('./test/')).assert_consumed()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # status.initialize_or_restore(sess)
    print(sess.run(v1))
    ckpnt.save('./test/myVar', sess)
    # ckpnt.save('./test/myVar', sess)
