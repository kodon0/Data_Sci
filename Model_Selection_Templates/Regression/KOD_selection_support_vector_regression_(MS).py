#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:44:23 2020

@author: kieranodonnell
"""

#Support Vector Regression -  Model Selection and Verification Template

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

#Feature scaling - Required for SVR
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))

#Import and train
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

#Predict Test set results
y_pred = scaler_y.inverse_transform(regressor.predict(scaler_X.transform(X_test)))
a = y_pred.reshape(len(y_pred),1)
b = y_test.reshape(len(y_test),1)
print(np.concatenate((a, b),axis =1))

#Evaluate Model Performance

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


