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
import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
num_episodes = 500
num_exploration_episodes = 100
max_len_episode = 1000
batch_size = 32
learning_rate = 1e-3
gamma = 1.
initial_epsilon = 1.
final_epsilon = 0.01


class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)


env = gym.make('CartPole-v1')
model = QNetwork()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
state = env.reset()
replay_buffer = deque(maxlen=10000)
epsilon = initial_epsilon

for episode_id in range(num_episodes):
    state = env.reset()
    epsilon = max(initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
                  final_epsilon)
    for t in range(max_len_episode):
        env.render()
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = model.predict(tf.constant(np.expand_dims(state, axis=0), dtype=tf.float32)).numpy()
            action = action[0]

        next_state, reward, done, info = env.step(action)

        reward = -10. if done else reward

        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
            break

        if len(replay_buffer) >= batch_size:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                [np.array(a, dtype=np.float32) for a in zip(*random.sample(replay_buffer, batch_size))]
            q_value = model(tf.constant(batch_next_state, dtype=tf.float32))
            y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)

            with tf.GradientTape() as tape:
                loss = tf.losses.mean_squared_error(labels=y,
                                                    predictions=tf.reduce_sum(
                                                        model(tf.constant(batch_state)) * tf.one_hot(batch_action,
                                                                                                     depth=2),
                                                        axis=1))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

env.env.close()
