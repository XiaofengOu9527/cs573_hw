#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ouxf17
"""

import numpy as np
import pandas as pd
import sys
import random
import matplotlib.pyplot as plt




def kmeans(dataFilename, K, random_seed=0):
    np.random.seed(random_seed)

    data = pd.read_csv(dataFilename, header=None)
    N = data.shape[0]

    max_iter_ct = 50
    centroid_update_tol = 1e-6

    data_matrix = data.iloc[:, 2:].to_numpy()
    data_true_label = data.iloc[:, 1].to_numpy()
    data_pred_label = np.zeros(N)


    centroid = data_matrix[np.random.randint(0, N, size=K), :]

    iter_ct = 0
    flag = True
    while flag:
        # assign label to data points
        label_pred_ct = np.zeros(K)     # model distribution
        for i in range(N):
            data_point = data_matrix[i,:]
            dist = np.sum(np.square(centroid - data_point), axis=1)
            label = int(np.argmin(dist))
            data_pred_label[i] = label
            label_pred_ct[label] += 1

        iter_ct += 1
        if iter_ct >= max_iter_ct:
            flag = False
            break
        else:
            # update centroid
            old_centroid = centroid
            
            centroid = np.zeros(shape=(K, 2))
            for i in range(N):
                label = int(data_pred_label[i])
                centroid[label, :]  += data_matrix[i, :]
            
            # new centroid
            centroid /= label_pred_ct[:, None]

            if np.max(np.abs(old_centroid - centroid)) < centroid_update_tol:
                # print('stop at {} iteration'.format(str(iter_ct)))
                iter_ct = max_iter_ct



    label_true_ct = np.zeros(10)     # true distribution
    label_pred_ct = np.zeros(K)     # predicting distribution

    label_true_idxs = dict.fromkeys(range(10))
    label_pred_idxs = dict.fromkeys(range(K))

    for key in label_pred_idxs.keys():
        label_pred_idxs[key] = []

    for key in label_true_idxs.keys():
        label_true_idxs[key] = []
        
    for i in range(N):
        pred_label = int(data_pred_label[i])
        true_label = int(data_true_label[i])

        label_pred_idxs[pred_label].append(i)
        label_true_idxs[true_label].append(i)

        label_true_ct[true_label] += 1
        label_pred_ct[pred_label] += 1

    label_true_ct /= N
    label_pred_ct /= N

    # # compute true WC-SSD
    # true_wc_ssd = 0
    # for k in range(K):
    #     idxs = label_true_idxs[k]
    #     true_centroid = np.mean(data_matrix[idxs, :], axis=0)
    #     true_wc_ssd += np.sum(np.square(data_matrix[idxs, :] - true_centroid))
    # print("True WC-SSD: {}".format(str(round(true_wc_ssd,2))))


    # compute WC-SSD
    wc_ssd = 0
    for label in range(K):
        idxs = label_pred_idxs[label]
        label_centroid = centroid[label, :]
        wc_ssd += np.sum(np.square(data_matrix[idxs, :] - label_centroid))

    # compute SC
    S = np.zeros(N)
    for label in range(K):
        idxs = label_pred_idxs[label]
        for idx in idxs:
            A = np.sum(np.sqrt(np.sum(np.square(data_matrix[idxs, :] - data_matrix[idx, :]), axis=1)))
            B = np.sum(np.sqrt(np.sum(np.square(data_matrix - data_matrix[idx, :]), axis=1))) - A
            A /= len(idxs)
            if len(idxs) < N:
                B /= (N - len(idxs))
            else:
                B = 0
            S[idx] = (B - A) / max(A, B)

    sc_score = np.mean(S)
    
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
    

    # return outcomes
    return round(wc_ssd, 2), round(sc_score, 2), round(NMI_score, 2), data_pred_label




if __name__ == '__main__':
    arg = sys.argv
    dataFilename = arg[1]
    K = int(arg[2])
    wc_ssd, sc_score, NMI_score, _ = kmeans(dataFilename, K)
    print("WC-SSD: {}".format(str(round(wc_ssd,2))))
    print("SC: {}".format(str(round(sc_score,2))))
    print("NMI: {}".format(str(round(NMI_score,2))))