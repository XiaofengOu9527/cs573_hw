#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:50:59 2019

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
data = data_orin.sample(frac=1, random_state=18)

depth_limit = 8
example_limit = 50
num_tree = 30

# within each fold, use fractions
t_fracs = (0.05, 0.075, 0.1, 0.15, 0.2)

fold = 10
n = data.shape[0]
n_fold = int(n / fold)
train_accuracy = np.zeros(shape=(len(t_fracs), fold, 3))

for i in range(fold):
    data_fold = data.iloc[i*n_fold : (i+1)*n_fold].reset_index(drop=True)

    for j in range(len(t_fracs)):
        t_frac = t_fracs[j]
        trainSet = data_fold.sample(frac=t_frac, random_state=32).reset_index(drop=True)

        _, _, _, train_accuracy[j, i, 0], _ = trees.decisionTree(trainSet, trainSet, flag='all', depth_limit=depth_limit, example_limit=example_limit)

        _, _, _, train_accuracy[j, i, 1], _ = trees.bagging(trainSet, trainSet, num_tree=num_tree, depth_limit=depth_limit, example_limit=example_limit)

        _, _, _, train_accuracy[j, i, 2], _ = trees.randomForest(trainSet, trainSet, num_tree=num_tree, depth_limit=depth_limit, example_limit=example_limit)



aver_accuracy = np.mean(train_accuracy, axis=1)
aver_accuracy = aver_accuracy.squeeze()
print(aver_accuracy[:, 1])

std_error = np.std(train_accuracy, axis=1) / np.sqrt(fold)
std_error = std_error.squeeze()

plt.errorbar(range(len(t_fracs)), aver_accuracy[:, 0], yerr=std_error[:, 0], label='decision tree')

plt.errorbar(range(len(t_fracs)), aver_accuracy[:, 1], yerr=std_error[:, 1], label='bagging')

plt.errorbar(range(len(t_fracs)), aver_accuracy[:, 2], yerr=std_error[:, 2], label='random forest')

plt.xticks(range(len(t_fracs)), t_fracs)
plt.legend()
plt.savefig('models_accuracy_tfracs')


""" compare performance between DT and BT """
print("compare performance between DT and BT")
for j in range(len(t_fracs)):
        t_frac = t_fracs[j]
        (t_stat, p_value) = stats.ttest_ind(train_accuracy[j, :, 0], train_accuracy[j, :, 1])
        print("p value for the t test at t_frac {}: {}".format(str(t_frac), str(p_value)))