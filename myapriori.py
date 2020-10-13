#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 00:33:00 2020

@author: suryanshsoni
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import apyori
from apyori import apriori


dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
rules = apriori(transactions = transactions, 
                min_support = 0.003, 
                min_confidence = 0.2,
                min_lift = 3, 
                min_length = 2, 
                max_length = 2)


results = list(rules)
print(results)





