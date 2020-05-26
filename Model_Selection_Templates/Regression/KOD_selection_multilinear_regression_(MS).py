#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:41:39 2020

@author: kieranodonnell
"""
#Multiple Linear Regression -  Model Selection and Verification Template

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Import dataset

df = pd.read_csv('Data.csv')
X = df.iloc[:, :-1].values #In this case and subject to change depending on dataset
y = df.iloc[:, -1].values #In this case and subject to change depending on dataset

#Assume data has been cleaned before this next stage

#Split data into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Import and train
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict Test set results
y_pred = regressor.predict(X_test)
a = y_pred.reshape(len(y_pred),1)
b = y_test.reshape(len(y_test),1)
print(np.concatenate((a, b),axis =1))

#Evaluate Model Performance

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

