#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 08:26:55 2020

@author: kieranodonnell
"""

# Associating Rules
# Eclat

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []

#Need to be a list format instead of a DF
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    #Gets every product from every transaction

# Training the Eclat model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
print(results)


# Setting reuslts into a DF

def sort(results):
    lhs =[tuple(result[2][0][0])[0] for result in results]
    rhs =[tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]

    return list(zip(lhs, rhs, supports))

results_df = pd.DataFrame(sort(results), columns = ["Product 1", "Product 2", "SUPPORT"])

results_df.nlargest(n=20, columns = "SUPPORT")
