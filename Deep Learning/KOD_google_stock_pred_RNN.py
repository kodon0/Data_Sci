#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:11:40 2020

@author: kieranodonnell
"""


'''This script uses LSTM and RNN in an attempt to predict the stock price of a company.
in this case it's Google, and has been already split into train/test. Data has previously been cleaned
This does not take the date index into consideration altough it can be easily modified'''

# Standard libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import training set uisng 'Open' price
df_train = pd.read_csv('/Users/kieranodonnell/Documents/GitHub/Data Sci/Datasets/Google_Stock_Price_Train.csv')
training = df_train.iloc[:,1:2].values # Get numpy array of open price
df_test = pd.read_csv('/Users/kieranodonnell/Documents/GitHub/Data Sci/Datasets/Google_Stock_Price_Test.csv')
real = df_test.iloc[:,1:2].values # Get numpy array of open price


# Scaling normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_scaled = sc.fit_transform(training)

# Produce a df for 60 timesteps with 1 output
X_train = []
y_train = []
for i in range(60,train_scaled.shape[0]):
    X_train.append(train_scaled[i-60:i,0])
    y_train.append(train_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Adding extra dimensionality for capacity for additional indicators -> modify indicator numbers as required
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1], 1))

# Building stacked LSTM RNN with dropout
# 4 layers with 1 output and 50 units per layer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
 
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(rate = 0.2))
regressor.add(Dense(units = 1))
 
 # Compile RNN
 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train,y_train, epochs = 120, batch_size = 60)

# Predictions and evaluation
# Getting preidcted stock price for Jan 2017
df_full = pd.concat((df_train['Open'],df_test['Open']), axis = 0)

# Making appropriate test set
# Bound of input are Jan 3rd 2017 - 60 days and Jan 31st 2017 - 60 days and reshaping
input_vals = df_full[len(df_full)-len(df_test)-60:].values
input_vals = input_vals.reshape(-1,1)
scaled_input = sc.transform(input_vals)
X_test = []
for i in range(60,scaled_input.shape[0]): # set to 80 if required
    X_test.append(scaled_input[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1], 1))

# Predict stock price for window

predicted_price = regressor.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)

# Visualisations

plt.plot(real, color = 'blue', label = 'Real GOOG stock price')
plt.plot(predicted_price, color = 'red', label = 'Predicted GOOG stock price')
plt.title('Real vs Preidcted GOOG stock Price')
plt.xlabel('Time Index')
plt.ylabel('Opening price') 
plt.legend()
plt.show()

# Numerical evalutaion
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(f"RMSE is {np.sqrt(mean_squared_error(predicted_price,real))}")
print(f"MAE is {mean_absolute_error(predicted_price,real)}")