#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:09:00 2019

@author: ouxf17
"""

import pandas as pd
import sys


""" read the commmand line input arguments""" 
arg = sys.argv
input_file = arg[1]
n = int(arg[2])



""" read the csv file"""
data = pd.read_csv(input_file, nrows=n)  # n should be 6500



""" part i """
drop_cols = ['race', 'race_o', 'field']
data = data.drop(drop_cols, axis=1)


""" part ii """
convt_cateval = ["gender"]

attri_val = {}
attri_numval = {}
attri_val_ct = {}

for attri in convt_cateval:
    attri_val[attri] = set()
    
for index, row in data.iterrows():
    for attri in convt_cateval:
        attri_val[attri].add(row[attri])

for attri in convt_cateval:
    attri_val[attri] = list(attri_val[attri])
    attri_val[attri].sort()
    
    n_val = len(attri_val[attri])
    attri_val_ct[attri] = n_val
    
    num_val = [x for x in range(n_val)]
    
    attri_numval[attri] = dict(zip(attri_val[attri], num_val))

for index, row in data.iterrows():
    for attri in convt_cateval:
        data.loc[index, attri]= attri_numval[attri][row[attri]]


preference_scores_of_participants = [
        "attractive_important",
        "sincere_important",
        "intelligence_important",
        "funny_important",
        "ambition_important",
        "shared_interests_important"]
preference_scroes_of_partners = [
        "pref_o_attractive",
        "pref_o_sincere",
        "pref_o_intelligence",
        "pref_o_funny",
        "pref_o_ambitious",
        "pref_o_shared_interests"]

mean_participants = dict.fromkeys(preference_scores_of_participants, 0)
mean_partners = dict.fromkeys(preference_scroes_of_partners, 0)

for index, row in data.iterrows():
    total = 0
    for attri in preference_scores_of_participants:
        total += row[attri]
    
    for attri in preference_scores_of_participants:
        data.loc[index, attri] = row[attri] / total
        mean_participants[attri] += data.loc[index, attri]
        
    total = 0
    for attri in preference_scroes_of_partners:
        total += row[attri]
        
    for attri in preference_scroes_of_partners:
        data.loc[index, attri] = row[attri] / total
        mean_partners[attri] += data.loc[index, attri]




""" part iii """
discrete_columns = ['gender', 'race', 'race_o',
                    'samerace', 'field', 'decision']
all_columns = data.columns.values.tolist()
continuous_columns = [attri for attri in all_columns if attri not in discrete_columns]

bins = 2
labels = [0, 1]

for col in continuous_columns:
    data[col] = pd.cut(data[col], bins=bins, labels=labels)




""" part iv """
random_state = 47
frac = 0.2

test_data = data.sample(frac=frac, random_state=random_state)
train_data = data.drop(test_data.index)

test_data.to_csv("testSet.csv", index=False)
train_data.to_csv("trainingSet.csv", index=False)


