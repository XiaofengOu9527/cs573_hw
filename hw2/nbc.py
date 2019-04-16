#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 16:55:14 2019

@author: ouxf17
"""
import pandas as pd

random_seed = 47

def nbc(t_frac, training_file):
    training_data = pd.read_csv(training_file)
    training_samples = training_data.sample(frac=t_frac, random_state=random_seed)
    
    attris = training_samples.columns.values.tolist()
    attris = [attri for attri in attris if attri not in ["decision"]]
    
    total_ct = training_samples.shape[0]
    success_ct = 0
    failure_ct = 0
    
    success_attri_prob = {}
    failure_attri_prob = {}
    
    """ initilization """
    for attri in attris:
        success_attri_prob[attri] = {}
        failure_attri_prob[attri] = {}
        
        
    """ training """ 
    """ add Laplace correction later """
    for index, row in training_samples.iterrows():
        if row["decision"] == 1:
            success_ct += 1
            for attri in attris:
                success_attri_prob[attri][row[attri]] = success_attri_prob[attri].get(row[attri], 0) + 1
        else:
            failure_ct += 1
            for attri in attris:
                failure_attri_prob[attri][row[attri]] = failure_attri_prob[attri].get(row[attri], 0) + 1

    
    for attri in attris:
        for val in success_attri_prob[attri]:
            success_attri_prob[attri][val] /= success_ct
            
        for val in failure_attri_prob[attri]:
            failure_attri_prob[attri][val] /= failure_ct
            
    success_prob = success_ct / total_ct
    
    return (success_prob, 1 - success_prob, success_attri_prob, failure_attri_prob)