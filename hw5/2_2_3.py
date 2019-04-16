import numpy as np
import pandas as pd
import sys
import random
import matplotlib.pyplot as plt
from kmeans import kmeans

# np.random.seed(0)

arg = sys.argv
dataFilename1 = arg[1]
dataFilename2 = arg[2]
dataFilename3 = arg[3]


Ks = (2, 4, 8, 16, 32)
random_seeds = np.random.randint(1000, size=10)

wc_ssds = np.zeros(shape=(3, 10, len(Ks)))
sc_scores = np.zeros(shape=(3, 10, len(Ks)))

for j in range(len(random_seeds)):
    random_seed = random_seeds[j]
    for i in range(len(Ks)):
        K = Ks[i]
        wc_ssds[0, j, i], sc_scores[0, j, i], _, _ = kmeans(dataFilename1, K, random_seed=random_seed)
        wc_ssds[1, j, i], sc_scores[1, j, i], _, _ = kmeans(dataFilename2, K, random_seed=random_seed)
        wc_ssds[2, j, i], sc_scores[2, j, i], _, _ = kmeans(dataFilename3, K, random_seed=random_seed)

aver_wc_ssds = np.mean(wc_ssds, axis=1).squeeze()
std_error_wc_ssds = np.std(wc_ssds, axis=1).squeeze() / np.sqrt(10)

plt.figure()
plt.errorbar(range(len(Ks)), aver_wc_ssds[0, :], yerr=std_error_wc_ssds[0, :], label='dataset1')
plt.errorbar(range(len(Ks)), aver_wc_ssds[1, :], yerr=std_error_wc_ssds[1, :], label='dataset2')
plt.errorbar(range(len(Ks)), aver_wc_ssds[2, :], yerr=std_error_wc_ssds[2, :], label='dataset3')
plt.xticks(range(len(Ks)), Ks)
plt.legend()
plt.savefig('wc_ssd_cv_dataset123')



aver_sc_scores = np.mean(sc_scores, axis=1).squeeze()
std_error_sc_scores = np.std(sc_scores, axis=1).squeeze() / np.sqrt(10)
print("standard error of sc:")
print(std_error_sc_scores)

plt.figure()
plt.errorbar(range(len(Ks)), aver_sc_scores[0, :], yerr=std_error_sc_scores[0, :], label='dataset1')
plt.errorbar(range(len(Ks)), aver_sc_scores[1, :], yerr=std_error_sc_scores[1, :], label='dataset2')
plt.errorbar(range(len(Ks)), aver_sc_scores[2, :], yerr=std_error_sc_scores[2, :], label='dataset3')
plt.xticks(range(len(Ks)), Ks)
plt.legend()
plt.savefig('sc_score_cv_dataset123')
