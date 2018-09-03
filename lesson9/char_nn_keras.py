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

text = open('../lesson9/data/test.txt').read()

unique = sorted(set(text))
len_of_char = len(unique)
# creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(unique)}
idx2char = {i: u for i, u in enumerate(unique)}

input_text = []
target_text = []
max_length = 60
batch_size = 32

for f in range(0, len(text) - max_length, max_length):
    inps = text[f:f + max_length]
    targ = text[f + 1:f + 1 + max_length]

    input_text.append([char2idx[i] for i in inps])
    target_text.append([char2idx[t] for t in targ])

input_text = np.array(input_text)
target_text = np.array(target_text)


class MyModel(keras.Sequential):
    def __init__(self, vocab=len_of_char, embed_size=100, rnn_dim=100):
        super(MyModel, self).__init__()

        self.add(keras.layers.Embedding(vocab, embed_size))

        self.add(keras.layers.GRU(rnn_dim, return_sequences=True))

        self.add(keras.layers.Dense(vocab, activation='softmax'))


model = MyModel()
model.compile(optimizer=keras.optimizers.Nadam(lr=1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
print(50 * '-')

model.fit(x=input_text, y=np.expand_dims(target_text, 2), epochs=1, batch_size=batch_size)

num_generate = 1000

# You can change the start string to experiment
start_string = 'به نام خدا'
# converting our start string to numbers(vectorizing!)
input_eval = np.zeros([max_length])
input_eval[0:len(start_string)] = [char2idx[s] for s in start_string]
input_eval = np.expand_dims(input_eval, 0)

# empty string to store our results
text_generated = ''

model.reset_states()

ids = [i for i in range(len_of_char)]
for i in range(num_generate):
    predictions = model.predict(input_eval)

    predicted_id = np.random.choice(list(char2idx.values()), 1, p=predictions[0][0])[0]

    input_eval = input_eval[1:] + predicted_id
    # input_eval[0] = predicted_id

    text_generated += idx2char[predicted_id]

print(start_string + text_generated)
