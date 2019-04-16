#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:47:49 2019

@author: ouxf17
"""

import pandas as pd
import sys
import numpy as np


""" read the commmand line input arguments""" 
arg = sys.argv
input_file = arg[1]
n = int(arg[2])
nbc_input_file = arg[3]


""" read the csv file"""
data = pd.read_csv(input_file, nrows=n)  # n should be 6500
nbc_data = pd.read_csv(nbc_input_file, nrows=n)



"""
part(i)
"""

strip_quote_attri = ["race", "race_o", "field"]
num_quo_rm = 0

for index, row in data.iterrows():
    for attri in strip_quote_attri:
        length = len(row[attri])
        if row[attri][0] == "'" and row[attri][length-1] == "'":
            num_quo_rm += 1
            data.loc[index, attri] = row[attri][1:length-1]

#print("Quotes removed from ", num_quo_rm, " cells")



lwrca_attri = ["field"]
num_lwrc = 0

for index, row in data.iterrows():
    for attri in lwrca_attri:
        data.loc[index, attri] = row[attri].lower()
        if not row[attri] == data.loc[index, attri] :
            num_lwrc += 1

#print("Standardized ", num_lwrc, " cells to lower case")


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
        
""" mean of each attribute """
for attri in mean_participants:
    mean_participants[attri] /= n
    #print("Mean of " + attri + ": ", round(mean_participants[attri], 2))
    
for attri in mean_partners:
    mean_partners[attri] /= n
    #print("Mean of " + attri + ": ", round(mean_partners[attri], 2))
    
    
    
    
    
    
# =============================================================================
# """ part (ii) one hot encoding """
# =============================================================================


convt_cateval = ["gender", "race", "race_o", "field"]
output_dict = {"gender": "female", 
               "race": "Black/African American", 
               "race_o": "Other",
               "field":"economics"}

attri_val = {}
attri_numval = {}
attri_val_ct = {}


""" initialize """
for attri in convt_cateval:
    attri_val[attri] = set()
    
    
""" collect all attribute values """
for index, row in data.iterrows():
    for attri in convt_cateval:
        attri_val[attri].add(row[attri])


""" sort the values and generate a dict for each value """
for attri in convt_cateval:
    attri_val[attri] = list(attri_val[attri])
    attri_val[attri].sort()
    n_val = len(attri_val[attri])
    
    attri_numval[attri] = dict()
    attri_val_ct[attri] = n_val
    
    val_order = [x for x in range(n_val)]
    
# =============================================================================
#     attri_numval[attri] = dict(zip(attri_val[attri], val_order))
#     
#     for key in attri_numval[attri]:
#         one_hot_vector = np.zeros(n_val-1)
#         
#         if attri_numval[attri][key] < n_val - 1:  # if not the last value
#             one_hot_vector[attri_numval[attri][key]] = 1
#         
#         attri_numval[attri][key] = one_hot_vector
# =============================================================================
    
    for i in range(n_val):
        one_hot_vec = np.zeros(n_val-1)
        if i < n_val - 1:
            one_hot_vec[i] = 1
        
        attri_numval[attri][attri_val[attri][i]] = one_hot_vec
    
""" one hot encoding using get_dummies """
non_dummy_cols = list(set(data.columns) - set(convt_cateval))
data = pd.concat([data[non_dummy_cols], 
				 pd.get_dummies(data[convt_cateval], prefix=convt_cateval)], 
				 axis=1)

# now drop the last reference value
for attri in convt_cateval:
	drop_col = attri + '_' + attri_val[attri][-1]
	data.drop(drop_col, axis=1, inplace=True)


    
# =============================================================================
# """ convert the attribute to one-hot vector """
# for index, row in data.iterrows():
#     for attri in convt_cateval:
#         data.at[index, attri] = attri_numval[attri][row[attri]].astype(int)
# =============================================================================

for attri in output_dict:
    print("Mapped vector for " + output_dict[attri] + " in column " + attri + ":", 
          attri_numval[attri][output_dict[attri]])



""" part (iii) """
frac = 0.2
random_state = 25

""" split the data """
test_data = data.sample(frac=frac, random_state=random_state)
train_data = data.drop(test_data.index)

nbc_test_data = nbc_data.sample(frac=frac, random_state=random_state)
nbc_train_data = data.drop(nbc_test_data.index)

""" output the dataframe to csv files """
test_data.to_csv("testSet.csv", index=False)
train_data.to_csv("trainingSet.csv", index=False)

nbc_test_data.to_csv("nbc_testSet.csv", index=False)
nbc_train_data.to_csv("nbc_trainingSet.csv", index=False)