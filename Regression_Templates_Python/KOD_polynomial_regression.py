#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:26:46 2020

@author: kieranodonnell
"""


#Polynomial regression vs linear regression
#Given: user asks for $165k - is this a true/good salary? user is between level 6 and 7 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df =  pd.read_csv("/Users/kieranodonnell/Desktop/Udemy/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv")

X = df.iloc[:,1:-1].values #Only want 'level' column
y = df.iloc[:,-1].values

#Don't need to encode as we are only looking at level column

#Won't split data into training and test

#Will use linear regression and polynomial regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y)

#Import poly features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)# Transformed features part of preprocessing

lin_reg_2 = LinearRegression() #Call new lin reg
lin_reg_2.fit(X_poly,y) #Fit to polynomial features

#Visualise results
#Linear reg
plt.scatter(X, y, color='green')
plt.plot(X, lin_reg.predict(X), color='red')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.title('Truth or Bluff (Linear Regression')
plt.show()

#Polynomial reg
plt.scatter(X, y, color='green')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.title('Truth or Bluff (Linear Regression')
plt.show()

#We have overfitting, but in this case it is ok

#Now to predict for linear and polynomial

lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) #Needs to be in order of polynomial -> needs to be transformed
