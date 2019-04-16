#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:12:20 2019

@author: ouxf17
"""

import pandas as pd
import sys
import numpy as np


""" read the commmand line input arguments""" 
arg = sys.argv
input_file = arg[1]
bins = int(arg[2])
output_file = arg[3]
if len(arg) < 5:
    display_seg = 1
else: 
    display_seg = int(arg[4])


""" read the csv file"""
data = pd.read_csv(input_file)
n = data.shape[0]



discrete_valued_columns = ['gender', 'race', 'race_o',
                           'samerace', 'field', 'decision']

columns = data.columns.values.tolist()
continuous_valued_columns = [attri for attri in columns if attri not in discrete_valued_columns]


""" these attributes has range from 0 to 1 """
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


""" these attributes has range from 18 to 58 """
ages = ['age', 'age_o']


""" this attribute has range from -1 to 1 """
interests_correlate = ["interests_correlate"]

""" all the other attributes ranges from 0 to 10 """



bin_num_attri = dict.fromkeys(continuous_valued_columns, None)
for attri in continuous_valued_columns:
    bin_num_attri[attri] = dict.fromkeys(range(bins), 0)

for index, row in data.iterrows():
    for attri in continuous_valued_columns:
# =============================================================================
#         if attri in preference_scores_of_participants:
#             num = bins - 1 - int(np.floor( (1 - row[attri]) / (1 / bins)))
#             
#         elif attri in preference_scroes_of_partners:
#             num = bins - 1 - int(np.floor( (1 - row[attri]) / (1 / bins)))
#                              
#         elif attri in ages:
#             num = bins - 1 - int(np.floor((58 - row[attri]) / (40 / bins)))
# 
#         elif attri in interests_correlate:
#             num = bins - 1 - int(np.floor((1 - row[attri]) / (2 / bins)))
#                 
#         else:
#             num = bins - 1 - int(np.floor((10 - row[attri]) / (10 / bins)))
#         
#         # special cases
#         if num > bins - 1:
#             num = bins - 1
#         elif num < 0:
#             num = 0
# =============================================================================
        if (attri in preference_scores_of_participants) or (attri in preference_scroes_of_partners):
            seg_inteval = np.arange(0, bins + 1) * (1 / bins)
        
        elif attri in ages:
            seg_inteval = np.arange(0, bins + 1) * (40 / bins) + 18
            
        elif attri in interests_correlate:
            seg_inteval = np.arange(0, bins + 1) * (2 / bins) - 1
    
        else:
            seg_inteval = np.arange(0, bins + 1) * (10 / bins)
            
        for j in range(0, bins):
            if j == 0:
                if seg_inteval[j] <= row[attri] <= seg_inteval[j+1]:
                    num = j;
                    break
            elif j == bins - 1:
                if seg_inteval[j] < row[attri]:
                    num = j;
                    break
            else:
                if seg_inteval[j] < row[attri] <= seg_inteval[j+1]:
                    num = j;
                    break

        data.loc[index, attri] = num
        bin_num_attri[attri][num] += 1
        
if display_seg:
    for attri in continuous_valued_columns:
        print(attri + ": ", list(bin_num_attri[attri].values()))
        



""" output the dataframe to a csv file """
data.to_csv(output_file, index=False)
                

