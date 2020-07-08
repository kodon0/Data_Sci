#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:06:39 2020

@author: kieranodonnell
"""


# Restricetd Bolzmann Machines -> for movie reccomendation
# Based of "An introduction to restricted Bolztmann machines" by Fischer et al

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

# Convert ratings from 1-5 to 0-1 (liked: yes or no, for stars > 3). -1 will mean 'unrated'
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set == 3] = 0
training_set[training_set > 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0 
test_set[test_set == 3] = 0
test_set[test_set > 3] = 1

# Create Boltzmann neural net class, hn = number of hidden nodes, vn = number of visible nodes
class RBM():
    def __init__(self,hn, vn):
        self.W = torch.randn(hn,vn) # Size of weights based on v and h
        self.a = torch.randn(1,hn) # Bias for hidden nodes
        self.b = torch.randn(1,vn) # Bias for visible nodes
    def bernoulli_sample_h(self,x):
        wx = torch.mm(x, self.W.t()) # product of visible node x and tensor weight
        activation = wx + self.a.expand_as(wx) # Ensure bias is applied to each row of minibatch
        p_h_given_v = torch.sigmoid(activation) # probability of h given v
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def bernoulli_sample_v(self,y):
        wy = torch.mm(y, self.W) # product of visible node y and tensor weight
        activation = wy + self.a.expand_as(wy) # Ensure bias is applied to each row of minibatch
        p_v_given_h = torch.sigmoid(activation) # probability of v given h
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self,ph0,phk,v0,vk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        
vn = len(training_set[0]) # Number of visible nodes
hn = 123 # Number of hidden nodes
batch_size = 20 # Tunable b

rbm = RBM(hn,vn)

# Train RBM
epoch_n = 20
for epoc in range(1,epoch_n + 1):
    training_losses = 0
    s = 0. # Need float
    for user_id in range(0, users_total - batch_size,batch_size):
        vk = training_set[user_id:user_id+batch_size]
        v0 = training_set[user_id:user_id+batch_size]
        ph0,_ = rbm.bernoulli_sample_h(v0)
        for k in range(epoch_n):
            _,hk = rbm.bernoulli_sample_h(vk)
            _,vk = rbm.bernoulli_sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk,_ = rbm.bernoulli_sample_h(vk)
        rbm.train(ph0,phk,v0,vk)
        training_losses += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s +=1.
    print("epoch: "+str(epoch_n)+", loss: "+str(training_losses/s))
        
# Testing RBM

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))  
            
            
