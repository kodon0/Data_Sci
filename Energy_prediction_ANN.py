#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:03:01 2020

@author: kieranodonnell
"""


# Using ANN to predict energy generation of a power plant.
# Original data soruced from UCI Repo: https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
# AT = Ambinet temp (degC), AP = Ambient pressure (mBar), RH = Relative humidity (%)
# V = Exhaust vacuum (cmHg), PE = Net hourly Production (MW)
# Index is hours

# Import libs
import numpy as np
import pandas as pd
import tensorflow as tf
# Verify tensorflow version tf.__version__

# Import data set
df = pd.read_excel('Folds5x2_pp.xlsx')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split into train/test -> EP is y, rest is X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Import ANN
ann = tf.keras.models.Sequential()

# Define ANN -> input + 3 hidden layers + output
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))

# Compile ANN with MSE loss 
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Train on training set
ann.fit(X_train, y_train, batch_size = 64, epochs = 750)

# Assess predictions
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2) # Reduce size of sig figs
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Results can be plotted if desired using matplotlib etc