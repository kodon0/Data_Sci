#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:55:39 2020

@author: kieranodonnell
"""


#Random Forest
#Given: user asks for $165k - is this a true/good salary? user is between level 6 and 7 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/kieranodonnell/Desktop/Udemy/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 9 - Random Forest Regression/Python/Position_Salaries.csv')

#Split X from Y
y = df.iloc[:,-1].values.reshape(len(y),1)
X = df.iloc[:, 1].values.reshape(len(X),1)

#Feature scaling not required. Test/train split not used here (dataset too small)

#Import decision tree regressor
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=20)

#Fit X and y
rfr.fit(X, y)

#Predict - predict on level at 6.5 as per previous regression examples
rfr.predict([[6.5]])
print(rfr.predict([[6.5]]))

#Visualise results - modified for higher dimensions
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'orange')
plt.plot(X_grid, rfr.predict(X_grid), color = 'red')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Random Forest)')
plt.show()

#This is usually done for higher dimensional data (which obviously cannot be plotted)
