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
import tensorflow.keras as keras

text = open('../lesson7/data/all_poem.txt').read()

unique = sorted(set(text))
vocab_size = len(unique)
# creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(unique)}
idx2char = {i: u for i, u in enumerate(unique)}

input_text = []
target_text = []
max_length = 50
batch_size = 32

for f in range(0, len(text) - max_length, max_length):
    inps = text[f:f + max_length]
    targ = text[f + 1:f + 1 + max_length]

    input_text.append([char2idx[i] for i in inps])
    target_text.append([char2idx[t] for t in targ])

input_text = np.array(input_text)
target_text = np.array(target_text)

stateful_size = (len(input_text) // batch_size) * batch_size
input_text = input_text[:stateful_size]
target_text = target_text[:stateful_size]


class MyModel(keras.Sequential):
    def __init__(self, vocab_size=vocab_size, batch_siz3=128, embed_size=100, rnn_dim=100, seq_len=40):
        super(MyModel, self).__init__()

        self.add(keras.layers.Embedding(vocab_size, embed_size, batch_input_shape=(batch_siz3, None)))

        self.add(keras.layers.GRU(rnn_dim, stateful=True, return_sequences=True))

        self.add(keras.layers.TimeDistributed(keras.layers.Dense(vocab_size, activation='softmax')))


# def create_model():
# inp = keras.layers.Input((40,), batch_size=batch_size)
# out = MyModel()(
# return keras.Model(inp, out)


model = MyModel(batch_siz3=batch_size)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=input_text, y=np.expand_dims(target_text, 2), shuffle=False, epochs=1000, batch_size=batch_size)

config = model.get_config()
config[0]['config']['batch_input_shape'] = (1, None)
sample_model = keras.Sequential.from_config(config)
sample_model.trainable = False

num_generate = 25

# You can change the start string to experiment
start_string = 'ر'
# converting our start string to numbers(vectorizing!)
input_eval = [char2idx[s] for s in start_string]
input_eval = np.expand_dims(input_eval, 0)

# empty string to store our results
text_generated = ''

# low temperatures results in more predictable text.
# higher temperatures results in more surprising text
# experiment to find the best setting
temperature = 1.0

ids = [i for i in range(vocab_size)]
# hidden state shape == (batch_size, number of rnn units); here batch size == 1
for i in range(num_generate):
    predictions = sample_model.predict(input_eval)

    # using a multinomial distribution to predict the word returned by the model
    # predictions = predictions / temperature
    # np.random.choice([i for i in range(1001)], p=predictions[0][0])
    predicted_id = np.random.choice(ids, p=predictions[0][0])

    # We pass the predicted word as the next input to the model
    # along with the previous hidden state
    input_eval = np.expand_dims([predicted_id], 0)

    text_generated += idx2char[predicted_id]

print(start_string + text_generated)
