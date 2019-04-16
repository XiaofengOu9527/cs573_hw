#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:47:49 2019
@author: ouxf17
"""

import sys
import pandas as pd
import math
import random
import numpy as np


class DecisionTreeNode:
	def __init__(self, data=None, indices=None, level=1, attributes=[], 
              name=None, parent=None, children=[], label=None):
		""" 
		CLASS MEMBERS:
			data: dataframe
			indices: indices of rows that belong to this node
			name: for nonleaf nodes
			label: for leaf(terminal) nodes 
	    """
		self.data = data
		self.indices = indices
		self.attributes = attributes
		self.name = name
		self.parent = parent
		self.children = children
		self.level = level
		self.label = label
		
# =============================================================================
# select the attribute to split based on gini gain		
# =============================================================================
	def selectAttri(self, target_attri):
		n = len(self.attributes)
		gini_index = np.zeros(n) 
		split_count = np.zeros((n, 2, 2))
		
		for idx in self.indices:
			row = self.data.iloc[idx]
			curr_label = int(row[target_attri])
			
			for i in range(n):
				split_value = int(row[self.attributes[i]])       
				split_count[i, split_value, curr_label] += 1
		
		for i in range(n):
			for split_value in [0, 1]:
				count = split_count[i, split_value, :]
				total = np.sum(count)
				if total > 0:
					gini_index[i] += (1 - np.sum(np.square(count / total))) * total
		
		gini_index /= len(self.indices)
		
		attri = self.attributes[int(np.argmin(gini_index))]
		self.name = attri
		
		return attri

# =============================================================================
# split the node according to stopping criteria or the attribute chosen by selectAttri()
# =============================================================================

	def split(self, target_attri, depth_limit=8, example_limit=50):
		if self.level >= depth_limit or len(self.indices) < example_limit or len(self.attributes) == 0:
		# leaf node, terminate here, determine the label using majority vote
			negative_ct = 0
			positive_ct = 0
			for idx in self.indices:
				target_label = self.data.iloc[idx][target_attri]
				if target_label == 0:
					negative_ct += 1
				elif target_label == 1:
					positive_ct += 1
			
			if negative_ct > positive_ct:
				self.label = 0
			else:
				self.label = 1
				
			self.name= None
				
			return self.label, None, None
        
		# non-leaf node, split
		attri = self.selectAttri(target_attri)
		
		left_child_idxs = []
		right_child_idxs = []
		
		for idx in self.indices:
			if self.data.iloc[idx][attri] == 0:
				left_child_idxs.append(idx)
			elif self.data.iloc[idx][attri] == 1:
				right_child_idxs.append(idx)
		
		children_attris = [children_attri for children_attri in self.attributes 
					 if children_attri not in [attri]]	
		
		left_child = DecisionTreeNode(data=self.data, indices=left_child_idxs, 
								level=self.level+1, attributes=children_attris,
								parent=self, children=[])
		
		right_child = DecisionTreeNode(data=self.data, indices=right_child_idxs, 
								 level=self.level+1, attributes=children_attris,
								 parent=self, children=[])
		
		self.children.append(left_child)
		self.children.append(right_child)
		self.name = attri
	
		return attri, left_child, right_child

# =============================================================================
# 	evaluate the decision tree, should be called when the instance is the root of the tree
# =============================================================================

	def evaluateData(self, data):
		n = data.shape[0]
		outcome = np.zeros((n,1))
		
		for row_index, row in data.iterrows():
			curr_node = self
			terminal = False
			while not terminal:
				if curr_node.name == None:  # reach leaf node
					outcome[row_index, 0] = curr_node.label
					terminal = True
				else:						# split at current node
					attri = curr_node.name
					
					if row[attri] == 0:
						curr_node = curr_node.children[0]
					elif row[attri] == 1:
						curr_node = curr_node.children[1]
					else:
						raise ValueError("value error at {} of {} row".format(attri, str(row_index)))
		
		return outcome



# =============================================================================
# =============================================================================
# # main functions 
# =============================================================================
# =============================================================================

def decisionTree(trainingSet, testSet, flag='all', depth_limit=8, example_limit=50):

	all_attris = trainingSet.columns.values.tolist()
	attris = [attri for attri in all_attris if attri not in ['decision']]

	# pick features that constitute the decision tree
	if flag == 'all':
		tree_attris = attris
	elif flag == 'part':
		m = len(attris)
		k = math.floor(math.sqrt(m))
		tree_attris = [attris[i] for i in random.sample(range(m), k)]
	
	
	# train on trainingSet
	root = DecisionTreeNode(data=trainingSet, indices=range(trainingSet.shape[0]),
						 level=1, attributes=tree_attris)
	
	
	nodes_to_split = [root]	# push the root into stack
	
	while len(nodes_to_split) > 0:
		curr_node = nodes_to_split.pop()	# pop from stack the next node to split 
		attri, left_child, right_child = curr_node.split('decision', depth_limit=depth_limit, example_limit=example_limit)
		
		if left_child != None and right_child != None:
			nodes_to_split.append(left_child)
			nodes_to_split.append(right_child)
	
	# training accuracy
	training_predict = root.evaluateData(trainingSet) 
	true_label = trainingSet["decision"].to_numpy()
	true_label = np.reshape(true_label, newshape=training_predict.shape)
	train_accuracy = np.sum(training_predict == true_label) / true_label.shape[0]

	# testing accuracy
	test_predict = root.evaluateData(testSet) 
	true_label = testSet["decision"].to_numpy()
	true_label = np.reshape(true_label, newshape=test_predict.shape)
	test_accuracy = np.sum(test_predict == true_label) / true_label.shape[0]

	
	return root, training_predict, test_predict, train_accuracy, test_accuracy





def bagging(trainingSet, testSet, num_tree=30, depth_limit=8, example_limit=50):
	trees = dict.fromkeys(range(num_tree))
	n_train = trainingSet.shape[0]
	n_test = testSet.shape[0]
	train_predict_aggre = np.zeros((n_train, num_tree))
	test_predict_aggre = np.zeros((n_test, num_tree))
	
	for i in range(num_tree):
		data = trainingSet.sample(frac=1, replace=True)		# bootstrap sampling
		trees[i], train_predict, test_predict, _, _ = decisionTree(trainingSet=data, testSet=testSet, flag='all', depth_limit=depth_limit, example_limit=example_limit)
		
		train_predict_aggre[:, i] = train_predict.squeeze()
		test_predict_aggre[:, i] = test_predict.squeeze()
	
	# training accuracy
	train_predict = np.sum(train_predict_aggre, axis=1)
	train_predict = (train_predict > num_tree / 2)
	train_label = trainingSet["decision"].to_numpy()
	train_label = np.reshape(train_label, newshape=train_predict.shape[0])
	train_accuracy = np.sum(train_predict == train_label) / train_label.shape[0]
	
	# testing accuracy
	test_predict = np.sum(test_predict_aggre, axis=1)
	test_predict = (test_predict > num_tree / 2)
	test_label = testSet["decision"].to_numpy()
	test_label = np.reshape(test_label, newshape=test_predict.shape[0])
	test_accuracy = np.sum(test_predict == test_label) / test_label.shape[0]
	
	return trees, train_predict, test_predict, train_accuracy, test_accuracy



def randomForest(trainingSet, testSet, num_tree=30, depth_limit=8, example_limit=50):
	trees = dict.fromkeys(range(num_tree))
	n_train = trainingSet.shape[0]
	n_test = testSet.shape[0]
	train_predict_aggre = np.zeros((n_train, num_tree))
	test_predict_aggre = np.zeros((n_test, num_tree))
	
	for i in range(num_tree):
		data = trainingSet.sample(frac=1, replace=True)		# bootstrap sampling	
		trees[i], train_predict, test_predict, _, _ = decisionTree(trainingSet=data, testSet=testSet, flag='part', depth_limit=depth_limit, example_limit=example_limit)	
		train_predict_aggre[:, i] = train_predict.squeeze()
		test_predict_aggre[:, i] = test_predict.squeeze()
	
	# training accuracy
	train_predict = np.sum(train_predict_aggre, axis=1)
	train_predict = (train_predict > num_tree / 2)
	train_label = trainingSet["decision"].to_numpy()
	train_label = np.reshape(train_label, newshape=train_predict.shape[0])
	train_accuracy = np.sum(train_predict == train_label) / train_label.shape[0]
	
	# testing accuracy
	test_predict = np.sum(test_predict_aggre, axis=1)
	test_predict = (test_predict > num_tree / 2)
	test_label = testSet["decision"].to_numpy()
	test_label = np.reshape(test_label, newshape=test_predict.shape[0])
	test_accuracy = np.sum(test_predict == test_label) / test_label.shape[0]
	
	return trees, train_predict, test_predict, train_accuracy, test_accuracy




# =============================================================================
#  read the commmand line input arguments
# =============================================================================
if __name__ == '__main__':
	arg = sys.argv
	trainingDataFilename = arg[1]
	testDataFilename = arg[2]
	modelIdx = int(arg[3])
	
	trainingSet = pd.read_csv(trainingDataFilename)
	testSet = pd.read_csv(testDataFilename)
	
	if modelIdx == 1:
		tree, train_predict, test_predict, train_accuracy, test_accuracy = decisionTree(trainingSet, testSet, flag='all')
		print("Training accuracy DT: ", str(round(train_accuracy,2)))
		print("Testing accuracy DT: ", str(round(test_accuracy,2)))

	elif modelIdx == 2:
		trees, train_predict, test_predict, train_accuracy, test_accuracy = bagging(trainingSet, testSet)
		print("Training accuracy BAGGING: ", str(round(train_accuracy,2)))
		print("Testing accuracy BAGGING: ", str(round(test_accuracy,2)))

	elif modelIdx == 3:
		trees, train_predict, test_predict, train_accuracy, test_accuracy= randomForest(trainingSet, testSet)
		print("Training accuracy RF: ", str(round(train_accuracy,2)))
		print("Testing accuracy RF: ", str(round(test_accuracy,2)))

	else:
		raise ValueError("please enter 1 for decision tree, 2 for bagging, 3 for random forest!")