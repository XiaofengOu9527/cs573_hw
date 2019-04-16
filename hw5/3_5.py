import numpy as np
import pandas as pd
import sys
import random
import matplotlib.pyplot as plt
import scipy
import scipy.cluster.hierarchy as hierarchy

arg = sys.argv
dataFilename = arg[1]



np.random.seed(0)

data = pd.read_csv(dataFilename, header=None)
data_matrix = data.iloc[:, 2:].to_numpy()
data_label = data.iloc[:, 1].to_numpy()
N = data_matrix.shape[0]

labels = dict.fromkeys(range(10))
for key in labels.keys():
    labels[key] = []

for i in range(N):
    label = data_label[i]
    labels[label].append(i)


sample_labels = dict.fromkeys(range(10))
sample_idxs = []
for key in labels.keys():
    sample_labels[key] = list(np.random.choice(labels[key], 10))
    sample_idxs = sample_idxs + sample_labels[key]


sample_data_matrix = data_matrix[sample_idxs, :]
sample_data_label = data_label[sample_idxs]
N = sample_data_matrix.shape[0]


label_true_idxs = dict.fromkeys(range(10))
label_true_ct = np.zeros(10)

for key in label_true_idxs.keys():
    label_true_idxs[key] = []

for j in range(N):
    label = int(sample_data_label[j])
    label_true_idxs[label].append(j)
    label_true_ct[label] += 1
label_true_ct /= N

methods = {'single': 8, 'complete': 8, 'average': 8}


for method in methods.keys():
    Z = hierarchy.linkage(sample_data_matrix, method=method)

    K = methods[method]
    pred_label = hierarchy.fcluster(Z, K, criterion='maxclust') - 1

    label_pred_ct = np.zeros(K)     # predicting distribution
    label_pred_idxs = dict.fromkeys(range(K))
    # indices of each cluster
    for key in label_pred_idxs.keys():
        label_pred_idxs[key] = []

    for j in range(N):
        label = int(pred_label[j])
        label_pred_idxs[label].append(j)
        label_pred_ct[label] += 1

    label_pred_ct /= N

    # compute NMI
    Mutual_info = np.zeros(shape=(K, 10))      # mutual information matrix
    for i in range(K):
        for j in range(10):
            M = len(set(label_pred_idxs[i]).intersection(label_true_idxs[j])) / N
            if M == 0:
                Mutual_info[i,j] = 0
            else:
                Mutual_info[i,j] = M * (np.log2(M) - np.log2(label_pred_ct[i]) - np.log2(label_true_ct[j]))

    MI_score = np.sum(Mutual_info)

    true_entropy = - np.sum(label_true_ct[np.nonzero(label_true_ct)] * np.log2(label_true_ct[np.nonzero(label_true_ct)]))

    pred_entropy = - np.sum(label_pred_ct[np.nonzero(label_pred_ct)] * np.log2(label_pred_ct[np.nonzero(label_pred_ct)]))

    NMI_score = MI_score / (true_entropy + pred_entropy)


    print('NMI of {} linkage with K={}: {}'.format(method, str(K), str(round(NMI_score,2))))