#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:47:58 2019
preprocess the data in dating-full.csv
@author: ouxf17
"""

import pandas as pd
import sys


""" read the commmand line input arguments""" 
arg = sys.argv
input_file = arg[1]
output_file = arg[2]

""" read the csv file"""
data = pd.read_csv(input_file)
n = data.shape[0]




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

print("Quotes removed from ", num_quo_rm, " cells")
            



"""
part(ii)
"""

lwrca_attri = ["field"]
num_lwrc = 0

for index, row in data.iterrows():
    for attri in lwrca_attri:
        data.loc[index, attri] = row[attri].lower()
        if not row[attri] == data.loc[index, attri] :
            num_lwrc += 1

print("Standardized ", num_lwrc, " cells to lower case")
            



"""
part(iii)
"""

convt_cateval = ["gender", "race", "race_o", "field"]
output_dict = {"gender": "male", 
               "race": "European/Caucasian-American", 
               "race_o": "Latino/Hispanic American",
               "field":"law"}

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
    attri_val_ct[attri] = n_val
    
    num_val = [x for x in range(n_val)]
    
    attri_numval[attri] = dict(zip(attri_val[attri], num_val))
         
    
""" convert the attribute to numerical value """
for index, row in data.iterrows():
    for attri in convt_cateval:
        data.loc[index, attri]= attri_numval[attri][row[attri]]


for attri in output_dict:
    print("Value assigned for " + output_dict[attri] + " in column " + attri + ":", 
          attri_numval[attri][output_dict[attri]])




"""
part(iv)
"""

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
    print("Mean of " + attri + ": ", round(mean_participants[attri], 2))
    
for attri in mean_partners:
    mean_partners[attri] /= n
    print("Mean of " + attri + ": ", round(mean_partners[attri], 2))





""" output the dataframe to a csv file """
data.to_csv(output_file, index=False)

