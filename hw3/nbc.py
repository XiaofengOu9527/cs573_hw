#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 16:55:14 2019

@author: ouxf17
"""

def nbc(trainingSet):
    
    attris = trainingSet.columns.values.tolist()
    attris = [attri for attri in attris if attri not in ["decision"]]
    
    total_ct = trainingSet.shape[0]
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
    for index, row in trainingSet.iterrows():
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