#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:09:23 2019

@author: ouxf17
"""

import numpy as np
import pandas as pd
import sys
import random
import matplotlib.pyplot as plt

np.random.seed(0)


arg = sys.argv
raw_data_file = arg[1]
embed_data_file = arg[2]

data_raw = pd.read_csv(raw_data_file, header=None)
data_embed = pd.read_csv(embed_data_file, header=None)
data_raw = data_raw.reset_index(drop=True)
data_embed = data_embed.reset_index(drop=True)
N = data_embed.shape[0]


number_label = dict.fromkeys(range(10))
for key in number_label.keys():
    number_label[key] = []

for i in range(N):
    label = data_raw.iloc[i, 1]
    number_label[label].append(i)


data2 = data_embed.iloc[number_label[2] + number_label[6] + number_label[7]]
data3 = data_embed.iloc[number_label[6] + number_label[7]]
data2.to_csv('digits-embedding2.csv', index=False)
data3.to_csv('digits-embedding3.csv', index=False)




fig, axs = plt.subplots(1,10)
for i in range(10):
    sample_idx = random.sample(number_label[i], 1)
    sample_image = data_raw.iloc[sample_idx, 2:].to_numpy().reshape((28,28))
    ax = axs[i]
    ax.imshow(sample_image)
    ax.set_title(str(i))
    ax.axis('off')
plt.savefig("digit_view")



# visualize
sample_idxs = np.random.randint(0, N, size=1000)
sample_imgs = data_embed.iloc[sample_idxs, 2:].to_numpy()
sample_labels = data_embed.iloc[sample_idxs, 1].to_numpy()

df = pd.DataFrame(dict(x=sample_imgs[:, 0], y=sample_imgs[:, 1], label=sample_labels))
groups = df.groupby('label')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.scatter(group.x, group.y, label=name)
ax.legend()
plt.savefig("digit_embed_view")