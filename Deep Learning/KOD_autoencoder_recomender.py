#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:12:41 2020

@author: kieranodonnell
"""


# Autoencoders -> for movie reccomendations

# Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data 
from torch.autograd import Variable

# Import data
movies = pd.read_csv('../Datasets/ml-1m/movies.dat',
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')
users = pd.read_csv('../Datasets/ml-1m/users.dat',
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')
movie_ratings = pd.read_csv('../Datasets/ml-1m/ratings.dat',
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')

# Create train/test split
training_set = pd.read_csv('../Datasets/ml-100k/u1.base', delimiter = '\t') 
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('../Datasets/ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Total number of users and movie numbers
users_total = int(max(max(training_set[:,0]), max(test_set[:,0])))
movies_total = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Create data structure such that users in rows and movies in columns

def convert_data(data):
    conv_data = []
    for id_users in range(1, users_total + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(movies_total)
        ratings[id_movies - 1] = id_ratings
        conv_data.append(list(ratings))
    return conv_data

training_set = convert_data(training_set)
test_set = convert_data(test_set)

# Convert to Torch tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Create Autoencoder NN class
class SAE(nn.Module):
    def __init__(self,):
        super(SAE,self).__init__()
        self.fc1 = nn.Linear(movies_total, 25) # Encoding
        self.fc2 = nn.Linear(25,15)
        self.fc3 = nn.Linear(15,25) # Decoding
        self.fc4 = nn.Linear(25, movies_total)
        self.activation = nn.Sigmoid()
    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criteria = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.05, weight_decay = 0.75)

# Training phase with Autoencoder
epochs = 300
for epoch in range(1, epochs+1):
    training_losses = 0
    s = 0.
    for user_id in range(users_total):
        input = Variable(training_set[user_id]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0 # Values with no rating won't count
            loss = criteria(output, target)
            mean_correction = movies_total/float(torch.sum(target.data > 0) + 6.02e-26)
            loss.backward() # Back propagation
            training_losses += np.sqrt(loss.data*mean_correction)
            s += 1.
            optimizer.step()
    print("epoch: "+str(epoch)+", Training set loss: "+str(training_losses/s))
        
    
# Testing on test set with SAE
test_losses = 0
s = 0.
for user_id in range(users_total):
        input = Variable(training_set[user_id]).unsqueeze(0)
        target = Variable(test_set[user_id])
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[(target == 0).unsqueeze(0)] = 0 # Values with no rating won't count
            loss = criteria(output, target)
            mean_correction = movies_total/float(torch.sum(target.data > 0) + 6.02e-26)
            test_losses += np.sqrt(loss.data*mean_correction)
            s += 1.
print("Test set loss: "+str(test_losses/s))