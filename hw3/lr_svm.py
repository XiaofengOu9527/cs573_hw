#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:49:52 2019

@author: ouxf17
"""

import pandas as pd
import sys
import numpy as np


# =============================================================================
# =============================================================================
# # logistics regression
# =============================================================================
# =============================================================================

def lr(trainingSet, testSet, display=1):
	step = 0.01
	lamd = 0.01
	max_iter_ct = 500
	tol = 1e-6
	
	attris = trainingSet.columns.values.tolist()
	features = [feature for feature in attris if feature not in ['decision']]
	
	X_train = trainingSet[features].values
	X_train = np.column_stack((X_train, np.ones((X_train.shape[0], 1))))
	X_test = testSet[features].values
	X_test = np.column_stack((X_test, np.ones((X_test.shape[0], 1))))
	
	Y_train = trainingSet['decision'].values
	Y_train.shape = (len(Y_train), 1)
	Y_test = testSet['decision'].values
	Y_test.shape = (len(Y_test), 1)
	
	weights = np.zeros(shape=(X_train.shape[1], 1))

# =============================================================================
# 	training process
# =============================================================================
	iter_ct = 0
	while iter_ct < max_iter_ct:
		gradient = np.zeros(weights.shape)
		
		Y_predict = 1 / (1 + np.exp(-np.matmul(X_train, weights)))
		

		gradient += lamd * weights + np.matmul(X_train.T, Y_predict - Y_train)
		
		if np.linalg.norm(gradient) < tol:
			break
		
		iter_ct += 1
		weights -= step * gradient
		
		
# =============================================================================
# 	evaluation on trainingSet and testSet
# =============================================================================
	
	# trainingSet
	n_train = trainingSet.shape[0]
	Y_predict = 1 / (1 + np.exp(-np.matmul(X_train, weights)))
	Y_predict = (Y_predict > 0.5)
	n_success = np.sum(Y_predict == Y_train)	
	if display:
		print("Training Accuracy LR: ", round(n_success / n_train, 2))
	train_accuracy = n_success / n_train	

	# testSet
	n_test = testSet.shape[0]
	Y_predict = 1 / (1 + np.exp(-np.matmul(X_test, weights)))
	Y_predict = (Y_predict > 0.5)
	n_success = np.sum(Y_predict == Y_test)
	if display:
		print("Testing Accuracy LR: ", round(n_success / n_test, 2))
	test_accuracy = n_success / n_test
	
	return weights, train_accuracy, test_accuracy







# =============================================================================
# =============================================================================
# # support vector machine with soft margin 
# =============================================================================
# =============================================================================

def svm(trainingSet, testSet, display=1):
	step = 0.5
	lamd = 0.01
	max_iter_ct = 500
	tol = 1e-6
	
	attris = trainingSet.columns.values.tolist()
	features = [feature for feature in attris if feature not in ['decision']]
	
	X_train = trainingSet[features].values
	X_test = testSet[features].values
	
	Y_train = trainingSet['decision'].values
	Y_train.shape = (len(Y_train), 1)
	Y_train = 2 * Y_train - 1
	Y_test = testSet['decision'].values
	Y_test.shape = (len(Y_test), 1)
	Y_test = 2 * Y_test - 1
	
	weights = np.zeros(shape=(X_train.shape[1], 1))
	bias = np.zeros(1)
# =============================================================================
# 	training process
# =============================================================================	
	iter_ct = 0
	n_train = X_train.shape[0]
	while iter_ct < max_iter_ct:
		
		Y_predict = np.matmul(X_train, weights) + bias
		C = - np.multiply(Y_train, (np.multiply(Y_predict, Y_train) < 1))
		gradient_w = lamd * weights + np.matmul(X_train.T, C) / n_train
		gradient_b = np.sum(C) / n_train
 		
		if np.linalg.norm(gradient_w) + np.linalg.norm(gradient_b) < tol:
			break
		
		iter_ct += 1
		weights -= step * gradient_w
		bias -= step * gradient_b
		
# =============================================================================
# 	evaluation on trainingSet and testSet
# =============================================================================
	
	# trainingSet
	n_train = X_train.shape[0]
	Y_predict = np.matmul(X_train, weights) + bias
	Y_predict = 2 * (Y_predict > 0) - 1
	n_success = np.sum(Y_predict == Y_train)
	if display:
		print("Training Accuracy SVM: ", round(n_success / n_train, 2))
	train_accuracy = n_success / n_train
	
	
	# testSet
	n_test = X_test.shape[0]
	Y_predict = np.matmul(X_test, weights) + bias
	Y_predict = 2 * (Y_predict > 0) - 1
	n_success = np.sum(Y_predict == Y_test)
	if display:
		print("Testing Accuracy SVM: ", round(n_success / n_test, 2))
	test_accuracy = n_success / n_test
	
	return (weights, bias), train_accuracy, test_accuracy
	
	





# =============================================================================
# =============================================================================
# # 	main execution
# =============================================================================
# =============================================================================
if __name__ == '__main__':
	trainingDataFilename = sys.argv[1]
	testDataFilename = sys.argv[2]
	modelIdx = int(sys.argv[3])   	# modelIdx = 1 = LR, modelIdx = 2 = SVM
	
	trainingData = pd.read_csv(trainingDataFilename)
	testData = pd.read_csv(testDataFilename)
	
	if modelIdx == 1:
		_, _, _ = lr(trainingData, testData)
	elif modelIdx == 2:
		_, _, _ = svm(trainingData, testData)
	else:
		raise ValueError('please enter 1 for logistics, 2 for svm!!!')	