#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:13:57 2020

@author: kieranodonnell
"""

#Multiple linear regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("50_Startups.csv")

#Encode categroical data - can use get dummy variabes (pd.get_dummies())
#Or can use onehot encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')#Set "state" column


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Set X to be encoded
X = np.array(ct.fit_transform(X))

#Split data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Import regressor (same as simple regression -> autodetects extra features)
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()

#Train
mlr.fit(X_train, y_train)

#Predictions based on X_test
y_pred = mlr.predict(X_test)
np.set_printoptions(precision=2) #sets sig fig

#Concat vectors to see comparison of data
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),axis=1))

#Plot test vs predictions for profits
plt.scatter(y_pred, y_test)
plt.xlabel(xlabel = 'Predicted')
plt.ylabel(ylabel = 'Test')

#Making a predicition for  example the profit of a startup with:
# R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = California?

print(mlr.predict([[1.0,0,0,160000,130000,300000]])) #Needs to be a 2D array -> [[x]]

#To get equation coeffs
print(mlr.coef_)
print(mlr.intercept_)
