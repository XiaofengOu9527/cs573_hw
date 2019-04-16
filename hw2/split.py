#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 21:07:11 2019

@author: ouxf17
"""

import pandas as pd
import sys


""" read the commmand line input arguments""" 
arg = sys.argv
input_file = arg[1]
training_file = arg[2]
test_file = arg[3]

test_frac = 0.2
random_seed = 47



""" read the csv file"""
data = pd.read_csv(input_file)


""" split the data """
test_data = data.sample(frac=test_frac, random_state=random_seed)
train_data = data.drop(test_data.index)



""" output the dataframe to csv files """
test_data.to_csv(test_file, index=False)
train_data.to_csv(training_file, index=False)