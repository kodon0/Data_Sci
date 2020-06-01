#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 07:18:45 2020

@author: kieranodonnell
"""


#SVR Example

#Import libraries
#Given: user asks for $165k - is this a true/good salary? user is between level 6 and 7 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/kieranodonnell/Desktop/Udemy/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv')

y = df.iloc[:,-1].values
X = df.iloc[:,1].values

#In this case, test/train split will not be carried out, as data st is very small and desire is
#To get a model that fits the non-linear data perfcetlyperfectly (or as close as possible)

#Features will be scaled

#Convert y into a 2D array for transformation

y = np.reshape(y, (len(y),1))
X = np.reshape(X, (len(X),1))

#Need to scale both X and y
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

#Training
from sklearn.svm import SVR

svr = SVR(kernel = 'rbf') #Using RBF kernel
#Fit
svr.fit(X, y)

#Predict - using .transform to scale into project scale
#and inverse_transform to convert back into original scale
scaler_y.inverse_transform(svr.predict(scaler_X.transform([[6.5]])))

#Visualise data -> scales need to be reversed
plt.scatter(scaler_X.inverse_transform(X),scaler_y.inverse_transform(y) , color = 'green')
plt.plot(scaler_X.inverse_transform(X),scaler_y.inverse_transform(svr.predict((X))), color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (SVR')
plt.show()
