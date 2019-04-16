#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 20:24:28 2019

@author: ouxf17
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt



""" read the commmand line input arguments""" 
arg = sys.argv
input_file = arg[1]

""" read the csv file"""
data = pd.read_csv(input_file)
n = data.shape[0]


""" initialization """
rating_of_partner_from_participant = ["attractive_partner",
                                      "sincere_partner",
                                      "intelligence_partner",
                                      "funny_partner",
                                      "ambition_partner",
                                      "shared_interests_partner"]

distinct_val = dict.fromkeys(rating_of_partner_from_participant, None)
succ_distinct_val = dict.fromkeys(rating_of_partner_from_participant, None)
succ_rate = dict.fromkeys(rating_of_partner_from_participant, None)




for attri in distinct_val:
    distinct_val[attri] = {}
    succ_distinct_val[attri] = {}
    succ_rate[attri] = {}


for index, row in data.iterrows():
    if row["decision"] == 1:
        succ = 1
    else:
        succ = 0
            
    for attri in rating_of_partner_from_participant:
        distinct_val[attri][row[attri]] = distinct_val[attri].get(row[attri], 0) + 1
        succ_distinct_val[attri][row[attri]] = succ_distinct_val[attri].get(row[attri], 0) + succ
    

""" number of distinct value in each of the six attribute """
for attri in rating_of_partner_from_participant:
    print("Number of distinct value in " + attri + ": ", len(distinct_val[attri]))
    
    
    
    
for attri in rating_of_partner_from_participant:
    vals = list(distinct_val[attri])
    vals.sort()
    rate = []
    
    for val in vals:
        succ_rate[attri][val] = succ_distinct_val[attri][val] / distinct_val[attri][val]
        succ_rate[attri][val] = round(succ_rate[attri][val], 2)
        rate.append(succ_rate[attri][val])
        
    plt.scatter(vals, rate)
    plt.xlabel(attri + " score")
    plt.ylabel("success rate")
    save_file = attri + "_success_rate"
    plt.savefig(save_file)
    plt.close()

    
    
    
    

        
        
