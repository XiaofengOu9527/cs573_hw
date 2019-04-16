#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 21:50:18 2019

@author: ouxf17
"""

import subprocess
import matplotlib.pyplot as plt

F = [0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1]
test_accuracy = []
training_accuracy = []

for t_frac in F:
    print("Fraction of training samples ", t_frac)
    
    train_cmd = 'python' + ' ' + '5_1.py' + ' ' + 'trainingSet.csv' + ' ' + 'testSet.csv' + ' ' + str(t_frac)
                
    train_output = subprocess.run(train_cmd.split(), stdout=subprocess.PIPE)
    
    print(train_output.stdout.decode())
    
    length = len(train_output.stdout.decode())
    training_accuracy.append(float(train_output.stdout.decode()[length-28:length-24]))
    test_accuracy.append(float(train_output.stdout.decode()[length-5:length-1]))
    


# =============================================================================
# plt.plot(training_accuracy, label='training accuracy')
# plt.plot(test_accuracy, label='test accuracy')
# plt.show()
# =============================================================================

fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)

ax.plot(F, training_accuracy, c='b', marker='^', label='Training')
ax.plot(F, test_accuracy, c='r', marker='v', label='Test')

plt.legend(loc=2)
plt.savefig("training_fraction")