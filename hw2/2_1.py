#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:44:50 2019
visualizing interesting trends in data
@author: ouxf17
"""

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt


""" read the commmand line input arguments""" 
arg = sys.argv
input_file = arg[1]


""" read the csv file"""
data = pd.read_csv(input_file)
n = data.shape[0]


preference_scores_of_participants = ["attractive_important",
                                     "sincere_important",
                                     "intelligence_important",
                                     "funny_important",
                                     "ambition_important",
                                     "shared_interests_important"]


""" main task """
mean_female = dict.fromkeys(preference_scores_of_participants, 0)
mean_male = dict.fromkeys(preference_scores_of_participants, 0)

num_female = 0
num_male = 0

for index, row in data.iterrows():
    if row["gender"] == 0:
        num_female += 1
        for attri in preference_scores_of_participants:
            mean_female[attri] += row[attri]
                
    else:
        num_male += 1    
        for attri in preference_scores_of_participants:
            mean_male[attri] += row[attri]


for attri in preference_scores_of_participants:
    mean_female[attri] /= num_female
    mean_male[attri] /= num_male
    




""" visualization """
n_groups = len(preference_scores_of_participants)
mean_female = mean_female.values()
mean_male = mean_male.values()
attributes = ["attractive", "sincere", "intelligence", "funny", "ambition", "shared_interest"]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8

rects1 = plt.bar(index, mean_female, bar_width, alpha=opacity, color='fuchsia', label='female')
rects2 = plt.bar(index + bar_width, mean_male, bar_width, alpha=opacity, color='aqua', label='male')
            
plt.xlabel('Attributes')
plt.ylabel('Mean')
plt.title('Attribute mean by gender')
plt.xticks(index + 0.5*bar_width, attributes)
plt.legend()

plt.tight_layout()
plt.savefig("barplot")