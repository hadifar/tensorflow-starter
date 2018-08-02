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
import pandas as pd

if __name__ == '__main__':
    file = '/Users/mac/Downloads/Ganjoor.0.75.Data_androidgozar.com/Ganjoor.0.75_androidgozar.com/verse.csv'
    df = pd.read_csv(file, sep='\t')
    # all_ferdosi_poem = []
    # cat = range(1321, 1940)
    # i = 0
    # for item in df.iterrows():
    #     i = i + 1
    #     if item[1][0] in cat:
    #         all_ferdosi_poem.append(item[1].values[3])
    #
    #     if i % 1000:
    #         print(i)

    df1 = df.values[:, [3]]

    df1 = pd.DataFrame(df1)

    df1.to_csv('../all_poem.csv', index=False, sep='\t',header=False)
