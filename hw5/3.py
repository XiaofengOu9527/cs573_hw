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




Ks = (2, 4, 8, 16, 32)
methods = ['single', 'complete', 'average']
wc_ssd = np.zeros(shape=(3, len(Ks)))
sc_score = np.zeros(shape=(3, len(Ks)))

for t in range(len(methods)):
    method = methods[t]
    Z = hierarchy.linkage(sample_data_matrix, method=method)

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram with {} linkage'.format(method))
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hierarchy.dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig('dendrogram_{}_linkage'.format(method))

    
    for i in range(len(Ks)):
        K = Ks[i]
        pred_label = hierarchy.fcluster(Z, K, criterion='maxclust') - 1

        label_pred_ct = np.zeros(K)     # predicting distribution
        label_pred_idxs = dict.fromkeys(range(K))


        # indices of each cluster
        for key in label_pred_idxs.keys():
            label_pred_idxs[key] = []

        for j in range(N):
            label = int(pred_label[j])
            label_pred_idxs[label].append(j)


        # compute WC-SSD
        for k in range(K):
            idxs = label_pred_idxs[k]
            centroid = np.mean(sample_data_matrix[idxs, :], axis=0)
            wc_ssd[t, i] += np.sum(np.square(data_matrix[idxs, :] - centroid))


        # compute SC
        S = np.zeros(N)
        for label in range(K):
            idxs = label_pred_idxs[label]
            for idx in idxs:
                A = np.sum(np.sqrt(np.sum(np.square(sample_data_matrix[idxs, :] - sample_data_matrix[idx, :]), axis=1)))
                B = np.sum(np.sqrt(np.sum(np.square(sample_data_matrix - sample_data_matrix[idx, :]), axis=1))) - A
                A /= len(idxs)
                if len(idxs) < N:
                        B /= (N - len(idxs))
                else:
                        B = 0
                S[idx] = (B - A) / max(A, B)
        sc_score[t, i] = np.mean(S)

# plot sc and wc_ssd
plt.figure()
plt.plot(range(len(Ks)), wc_ssd[0, :], label='single')
plt.plot(range(len(Ks)), wc_ssd[1, :], label='complete')
plt.plot(range(len(Ks)), wc_ssd[2, :], label='average')
plt.xticks(range(len(Ks)), Ks)
plt.legend()
plt.savefig('wc_ssd_single_complete_average_linkage')

plt.figure()
plt.plot(range(len(Ks)), sc_score[0, :], label='single')
plt.plot(range(len(Ks)), sc_score[1, :], label='complete')
plt.plot(range(len(Ks)), sc_score[2, :], label='average')
plt.xticks(range(len(Ks)), Ks)
plt.legend()
plt.savefig('sc_score_single_complete_average_linkage')