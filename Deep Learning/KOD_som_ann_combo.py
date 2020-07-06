#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 08:25:29 2020

@author: kieranodonnell
"""


# Combining SOm (unsupervised) and ANN (supervised) for credit card fraud

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/kieranodonnell/Documents/GitHub/Data Sci/Datasets/Credit_Card_Applications.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Scale features

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

# Training SOM
from minisom import MiniSom
som = MiniSom(x = 10 , y = 10 ,input_len = 15, sigma = 1.0, learning_rate = 0.5)

# Randomly initialise vector weights
som.random_weights_init(X)
som.train_random(data = X, num_iteration=350)

# Visualize results of interneuron distances
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
m1 = np.reshape(mappings[(3,8)], (-1, 2))
m2 = np.reshape(mappings[(3,3)], (-1, 2))
frauds = np.concatenate((m1,m2), axis = 0)
frauds = sc.inverse_transform(frauds)

# Going to ANN

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))
  
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
  if dataset.iloc[i,0] in frauds:
    is_fraud[i] = 1
    
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

import tensorflow as tf
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=2, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(customers, is_fraud, batch_size = 1, epochs = 10)
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]