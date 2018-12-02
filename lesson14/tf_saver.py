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

v1 = tf.Variable(80., name="v1")
v2 = tf.Variable(10., name="v2")
a = tf.add(v1, v2)

saver1 = tf.train.Saver(var_list={'myVar1': v1, 'myVar2': v2})
saver2 = tf.train.Saver(var_list={'myVar1': v1})
saver3 = tf.train.Saver(var_list=tf.trainable_variables())
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, './saver_test/myVar')
    saver.restore(sess, './saver_test/myVar')
    print(sess.run(v1))
