#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:03:19 2020

@author: kieranodonnell
"""


# K-Fold Cross Validation and Grid Search

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) # Choosing 10 folds
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Apllying Grid Search to find best hyperparameters
from sklearn.model_selection import GridSearchCV
params = [{'C':[0.1,1,10,100,1000],'kernel':['linear']},
          {'C':[0.1,1,10,100,1000],'kernel':['rbf'],'gamma':[0.001, 0.001, 0.01, 0.1, 0.9]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = params,
                           scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_search.fit(X_train, y_train)
top_accuracy = grid_search.best_score_
top_params = grid_search.best_params_
print("Top Scoring Accuracy: {:.2f} %".format(top_accuracy*100))
print("Top Scoring Parameters:", top_params)