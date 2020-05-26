#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:02:25 2020

@author: kieranodonnell
"""


# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/kieranodonnell/Desktop/Udemy/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)

#Import linear regression
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)

#Predict

y_pred = linreg.predict(X_test)

#Visualise results - training
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,linreg.predict(X_train), color='blue')
plt.title('Salary vs Exp (Training)')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()


#Visualise results - test
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,linreg.predict(X_train), color='blue')
plt.title('Salary vs Exp (Test)')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()


#To predict the salary of an employee with 20 years experince:
print(linreg.predict([[12]])) #Needs to be a 2D array -> [[x]]

#To get equation coeffs
print(linreg.coef_)
print(linreg.intercept_)