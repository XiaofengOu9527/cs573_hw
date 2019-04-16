#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 21:00:33 2019

@author: ouxf17
"""

import pandas as pd
import sys
import nbc
from nbc_eval import nbc_eval

random_seed = 47

""" read the commmand line input arguments""" 
arg = sys.argv
training_file = arg[1]
test_file = arg[2]
t_frac = float(arg[3])

""" train the model """
(success_prob, failure_prob, success_attri_prob, failure_attri_prob) = nbc.nbc(t_frac, training_file)



""" apply the model to training and testing dataset """

""" training set """
trainingSet = pd.read_csv(training_file)
training_data = trainingSet.sample(frac=t_frac, random_state=random_seed)

attris = training_data.columns.values.tolist()
attris = [attri for attri in attris if attri not in ["decision"]]

# =============================================================================
# corr_predict = 0
# 
# attris = training_data.columns.values.tolist()
# attris = [attri for attri in attris if attri not in ["decision"]]
# 
# for index, row in training_data.iterrows():
#     p_succ_predict = 1
#     p_fail_predict = 1
#     
#     for attri in attris:
#         p_succ_predict *= success_attri_prob[attri].get(row[attri], 0)
#         p_fail_predict *= failure_attri_prob[attri].get(row[attri], 0)
#         
#     p_succ_predict *= success_prob
#     p_fail_predict *= failure_prob
# 
#     if p_succ_predict > p_fail_predict:
#         decision = 1
#     else:
#         decision = 0   
# 
#     if decision == row["decision"]:
#         corr_predict += 1
# 
# print("Training Accuracy: ", round(corr_predict / training_data.shape[0], 2))
# =============================================================================
acc = nbc_eval(training_data, success_prob, success_attri_prob, failure_attri_prob)
print("Training Accuracy: ", round(acc, 3))

""" test set"""
testing_data = pd.read_csv(test_file)
# =============================================================================
# corr_predict = 0
# 
# for index, row in testing_data.iterrows():
#     p_succ_predict = 1
#     p_fail_predict = 1
#     
#     for attri in attris:
#         p_succ_predict *= success_attri_prob[attri].get(row[attri], 0)
#         p_fail_predict *= failure_attri_prob[attri].get(row[attri], 0)
#         
#     p_succ_predict *= success_prob
#     p_fail_predict *= failure_prob
# 
#     if p_succ_predict > p_fail_predict:
#         decision = 1
#     else:
#         decision = 0   
# 
#     if decision == row["decision"]:
#         corr_predict += 1
# 
# print("Testing Accuracy: ", round(corr_predict / testing_data.shape[0], 2))
# =============================================================================
acc = nbc_eval(testing_data, success_prob, success_attri_prob, failure_attri_prob)
print("Testing Accuracy: ", round(acc, 3))