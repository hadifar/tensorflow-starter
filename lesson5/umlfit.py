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
from fastai.text import *
import html
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = '../lesson5/data/'
train = []
with open(os.path.join(path, 'train.ft.txt'), 'r') as file:
    for line in file:
        train.append(file.readline())

test = []
with open(os.path.join(path, 'test.ft.txt'), 'r') as file:
    for line in file:
        test.append(file.readline())

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

trn_texts, trn_labels = [text[10:] for text in train], [text[:10] for text in train]
trn_labels = [0 if label == '__label__1' else 1 for label in trn_labels]
val_texts, val_labels = [text[10:] for text in test], [text[:10] for text in test]
val_labels = [0 if label == '__label__1' else 1 for label in val_labels]

col_names = ['labels', 'text']

df_trn = pd.DataFrame({'text': trn_texts, 'labels': trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text': val_texts, 'labels': val_labels}, columns=col_names)

df_trn.to_csv(path + 'train.csv', header=False, index=False)
df_val.to_csv(path + 'test.csv', header=False, index=False)

CLASSES = ['neg', 'pos']
with open('data/classes.txt', 'w') as outfile:
    outfile.writelines(f'{o}\n' for o in CLASSES)

trn_texts, val_texts = train_test_split(np.concatenate([trn_texts, val_texts]), test_size=0.1)

df_trn = pd.DataFrame({'text': trn_texts, 'labels': [0] * len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({'text': val_texts, 'labels': [0] * len(val_texts)}, columns=col_names)

df_trn.to_csv(path + 'train.csv', header=False, index=False)
df_val.to_csv(path + 'test.csv', header=False, index=False)

chunksize = 24000
re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls=1):
    labels = df.iloc[:, range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls + 1, len(df.columns)):
        texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_
        labels += labels_
    return tok, labels


df_trn = pd.read_csv(path + 'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(path + 'test.csv', header=None, chunksize=chunksize)
