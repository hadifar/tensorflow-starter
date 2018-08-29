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

text = open('../lesson9/text.txt').read()

unique = sorted(set(text))

# creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(unique)}
idx2char = {i: u for i, u in enumerate(unique)}

input_text = []
target_text = []
max_length = 40
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
    def __init__(self, vocab_size=10000, batch_size=128, embed_size=100, rnn_dim=100, seq_len=40):
        super(MyModel, self).__init__()

        self.embed = keras.layers.Embedding(vocab_size, embed_size)

        self.gru = keras.layers.GRU(rnn_dim, stateful=True, return_sequences=True)

        self.time_distributed = keras.layers.TimeDistributed(keras.layers.Dense(vocab_size, activation='softmax'))

    def call(self, inputs):
        x = self.embed(inputs)
        x = self.gru(x)
        x = self.time_distributed(x)
        return x


def create_model():
    inp = keras.layers.Input((40,), batch_size=batch_size)
    out = MyModel()(inp)
    return keras.Model(inp, out)


model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=input_text, y=np.expand_dims(target_text, 2), shuffle=False, epochs=1, batch_size=batch_size)

num_generate = 1000

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

# hidden state shape == (batch_size, number of rnn units); here batch size == 1
for i in range(num_generate):
    predictions = model.predict(input_eval)

    # using a multinomial distribution to predict the word returned by the model
    predictions = predictions / temperature
    predicted_id = np.random.multinomial(predictions, pvals=[1 / 10000] * 10000)[0][0].numpy()

    # We pass the predicted word as the next input to the model
    # along with the previous hidden state
    input_eval = np.expand_dims([predicted_id], 0)

    text_generated += idx2char[predicted_id]

print(start_string + text_generated)
