#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:09:19 2019

@author: ouxf17
"""


def nbc_eval(data, success_prob, success_attri_prob, failure_attri_prob):
	failure_prob = 1 - success_prob
	corr_predict = 0
	
	attris = data.columns.values.tolist()
	attris = [attri for attri in attris if attri not in ["decision"]]
	
	for index, row in data.iterrows():
		p_succ_predict = 1
		p_fail_predict = 1
		
		for attri in attris:
			p_succ_predict *= success_attri_prob[attri].get(row[attri], 0)
			p_fail_predict *= failure_attri_prob[attri].get(row[attri], 0)
			
		p_succ_predict *= success_prob
		p_fail_predict *= failure_prob
		
		if p_succ_predict > p_fail_predict:
			decision = 1
		else:
			decision = 0
			
		if decision == row["decision"]:
			corr_predict += 1
			
	return corr_predict / data.shape[0]