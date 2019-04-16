import numpy as np
import pandas as pd
import sys
import random
import matplotlib.pyplot as plt
from kmeans import kmeans

np.random.seed(0)

arg = sys.argv
dataFilename1 = arg[1]
dataFilename2 = arg[2]
dataFilename3 = arg[3]


Ks = (2, 4, 8, 16, 32)

wc_ssd = np.zeros(shape=(3, len(Ks)))
sc_score = np.zeros(shape=(3, len(Ks)))

for i in range(len(Ks)):
    K = Ks[i]
    wc_ssd[0, i], sc_score[0, i], _, _ = kmeans(dataFilename1, K)
    wc_ssd[1, i], sc_score[1, i], _, _ = kmeans(dataFilename2, K)
    wc_ssd[2, i], sc_score[2, i], _, _ = kmeans(dataFilename3, K)

plt.figure()
plt.plot(range(len(Ks)), wc_ssd[0, :], label='dataset1')
plt.plot(range(len(Ks)), wc_ssd[1, :], label='dataset2')
plt.plot(range(len(Ks)), wc_ssd[2, :], label='dataset3')
plt.xticks(range(len(Ks)), Ks)
plt.legend()
plt.savefig('wc_ssd_dataset123')


plt.figure()
plt.plot(range(len(Ks)), sc_score[0, :], label='dataset1')
plt.plot(range(len(Ks)), sc_score[1, :], label='dataset2')
plt.plot(range(len(Ks)), sc_score[2, :], label='dataset3')
plt.xticks(range(len(Ks)), Ks)
plt.legend()
plt.savefig('sc_score_dataset123')




