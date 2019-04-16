#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:39:47 2019

@author: ouxf17
"""

import sys
import pandas as pd
import numpy as np
import trees
import matplotlib.pyplot as plt
from scipy import stats

arg = sys.argv
trainingDataFilename = arg[1]


data_orin = pd.read_csv(trainingDataFilename)
data_shuffled = data_orin.sample(frac=1, random_state=18)
data = data_shuffled.sample(frac=0.5, random_state=32)


depths = (3, 5, 7, 9)
example_limit = 50
num_tree = 30

# divide data into 10 folds
fold = 10
n = data.shape[0]
n_fold = int(n / fold)
train_accuracy = np.zeros(shape=(len(depths), fold, 3))

for i in range(fold):
    data_fold = data.iloc[i*n_fold : (i+1)*n_fold].reset_index(drop=True)

    for j in range(len(depths)):
        depth = depths[j]

        _, _, _, train_accuracy[j, i, 0], _ = trees.decisionTree(data_fold, data_fold, flag='all', depth_limit=depth, example_limit=example_limit)

        _, _, _, train_accuracy[j, i, 1], _ = trees.bagging(data_fold, data_fold, num_tree=num_tree, depth_limit=depth, example_limit=example_limit)

        _, _, _, train_accuracy[j, i, 2], _ = trees.randomForest(data_fold, data_fold, num_tree=num_tree, depth_limit=depth, example_limit=example_limit)



aver_accuracy = np.mean(train_accuracy, axis=1)
aver_accuracy = aver_accuracy.squeeze()

std_error = np.std(train_accuracy, axis=1) / np.sqrt(fold)
std_error = std_error.squeeze()

plt.errorbar(range(len(depths)), aver_accuracy[:, 0], yerr=std_error[:, 0], label='decision tree')

plt.errorbar(range(len(depths)), aver_accuracy[:, 1], yerr=std_error[:, 1], label='bagging')

plt.errorbar(range(len(depths)), aver_accuracy[:, 2], yerr=std_error[:, 2], label='random forest')

plt.xticks(range(len(depths)), depths)
plt.legend()
plt.savefig('models_accuracy_depth')


""" compare performance between DT and BT """
for j in range(len(depths)):
        depth = depths[j]
        (t_stat, p_value) = stats.ttest_ind(train_accuracy[j, :, 0], train_accuracy[j, :, 1])
        print("p value for the t test at depth {}: {}".format(str(depth), str(p_value)))