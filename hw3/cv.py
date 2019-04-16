#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:26:03 2019

@author: ouxf17
"""

import pandas as pd
import sys
import numpy as np
from lr_svm import lr, svm 
from nbc import nbc
from nbc_eval import nbc_eval
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
#  read the commmand line input arguments
# =============================================================================
if __name__ == '__main__':
	arg = sys.argv
	input_file = arg[1]
	nbc_input_file = arg[2]




# =============================================================================
# read and shuffle the data
# =============================================================================
orin_trainingData = pd.read_csv(input_file)
orin_nbc_trainingData = pd.read_csv(nbc_input_file)

random_state = 18
frac = 1
trainingData = orin_trainingData.sample(random_state=random_state, frac=frac)
nbc_trainingData = orin_nbc_trainingData.sample(random_state=random_state, frac=frac)
trainingData = trainingData.reset_index(drop=True)
nbc_trainingData = nbc_trainingData.reset_index(drop=True)




# =============================================================================
# split the data into folds
# =============================================================================
S = dict()
S_nbc = dict()
fold = 10;
num_each_fold = int(trainingData.shape[0] / fold)

for j in range(fold):
	lower = j * num_each_fold
	upper = (j + 1) * num_each_fold
	S[j] = trainingData.iloc[lower:upper]
	S_nbc[j] = nbc_trainingData.iloc[lower:upper]
	



# =============================================================================
# train the models on folds
# =============================================================================
t_fracs = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
nbc_accuracy = dict()
lr_accuracy = dict()
svm_accuracy = dict()

nbc_aver_accuracy = np.zeros(len(t_fracs))
nbc_stderr = np.zeros(len(t_fracs))

lr_aver_accuracy = np.zeros(len(t_fracs))
lr_stderr = np.zeros(len(t_fracs))

svm_aver_accuracy = np.zeros(len(t_fracs))
svm_stderr = np.zeros(len(t_fracs))

for k in range(len(t_fracs)):
	t_frac = t_fracs[k]
	nbc_accuracy[t_frac] = np.zeros(fold)
	lr_accuracy[t_frac] = np.zeros(fold)
	svm_accuracy[t_frac] = np.zeros(fold)
	
	for j in range(fold):
		test_set = S[j]
		S_C = pd.concat([S[i] for i in range(fold) if i not in [j]], ignore_index=True)
		train_set = S_C.sample(random_state=32, frac=t_frac)
		
		nbc_test_set = S_nbc[j]
		nbc_S_C = pd.concat([S_nbc[i] for i in range(fold) if i not in [j]], ignore_index=True)
		nbc_train_set = nbc_S_C.sample(random_state=32, frac=t_frac)
		
		# train each models (NBC, LR, SVM) from train_set
		# then apply the model to test_set and measure accuracy
		_, _, lr_accuracy[t_frac][j] = lr(train_set, test_set, display=0)
		_, _, svm_accuracy[t_frac][j] = svm(train_set, test_set, display=0)
		
		(success_prob, failure_prob, success_attri_prob, failure_attri_prob) = nbc(nbc_train_set)
		nbc_accuracy[t_frac][j] = nbc_eval(nbc_test_set, success_prob, success_attri_prob, failure_attri_prob)
		
	
	# compute average accuracy and standard error over 10 trials	(for each model)
	nbc_aver_accuracy[k] = np.mean(nbc_accuracy[t_frac])
	nbc_stderr[k] = np.std(nbc_accuracy[t_frac]) / np.sqrt(fold)
	
	lr_aver_accuracy[k] = np.mean(lr_accuracy[t_frac])
	lr_stderr[k] = np.std(lr_accuracy[t_frac]) / np.sqrt(fold)
	
	svm_aver_accuracy[k] = np.mean(svm_accuracy[t_frac])
	svm_stderr[k] = np.std(svm_accuracy[t_frac]) / np.sqrt(fold)
	


# =============================================================================
# print(nbc_aver_accuracy)
# print(lr_aver_accuracy)
# print(svm_aver_accuracy)
# =============================================================================
	
# =============================================================================
# plot the learning curve
# =============================================================================

x_axis = 4680 * np.array(t_fracs)

plt.errorbar(range(len(t_fracs)), nbc_aver_accuracy, yerr=nbc_stderr, label='nbc')
plt.errorbar(range(len(t_fracs)), lr_aver_accuracy, yerr=lr_stderr, label='lr')
plt.errorbar(range(len(t_fracs)), svm_aver_accuracy, yerr=svm_stderr, label='svm')
plt.xticks(range(len(t_fracs)), x_axis)
plt.legend()
plt.savefig('models_accuracy')

# =============================================================================
# t test on the performance of LR and SVM
# =============================================================================
(t_stat, p_value) = stats.ttest_ind(lr_aver_accuracy, svm_aver_accuracy)
print("p value for the t test: ", p_value)