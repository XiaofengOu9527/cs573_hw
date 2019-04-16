#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 18:24:21 2019

@author: ouxf17
"""

import subprocess
import matplotlib.pyplot as plt

B = [2, 5, 10, 50, 100, 200]
# =============================================================================
# B = [2]
# =============================================================================
test_accuracy = []
training_accuracy = []

for bins in B:
    print("Bin size: ", bins)
    
    discretize_file = 'dating-'+str(bins)+'-binned.csv'
    training_file = 'trainingSet-'+str(bins)+'-binned.csv'
    test_file = 'testSet-'+str(bins)+'-binned.csv'
    t_frac = 1.0
    
    discretize_cmd = 'python ' + 'discretize.py ' + 'dating.csv ' + str(bins) + ' ' + discretize_file + ' 0'
    
    split_cmd = 'python' + ' ' + 'split.py' + ' ' + discretize_file + ' ' + training_file + ' ' + test_file
    
    eval_cmd = 'python' + ' ' + '5_1.py' + ' ' + training_file + ' ' + test_file + ' ' + str(t_frac)
    
    discretize_output = subprocess.run(discretize_cmd.split(), stdout=subprocess.PIPE)
    
    split_output = subprocess.run(split_cmd.split(), stdout=subprocess.PIPE)
    
    eval_output = subprocess.run(eval_cmd.split(), stdout=subprocess.PIPE)
    
    print(eval_output.stdout.decode())
    
# =============================================================================
#     length = len(eval_output.stdout.decode())
# =============================================================================
    length = len(eval_output.stdout.decode())
    test_accuracy.append(float(eval_output.stdout.decode()[length-5:length-1]))
    training_accuracy.append(float(eval_output.stdout.decode()[length-28:length-24]))


fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)

ax.plot(B, training_accuracy, c='b', marker='^', label='Training')
ax.plot(B, test_accuracy, c='r', marker='v', label='Test')

plt.legend(loc=2)
plt.savefig("Bin_number_vs_accuracy")