#KOD Multi-linear regression

#Import data set

dataset = read.csv('/Users/kieranodonnell/Desktop/Udemy/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/R/50_Startups.csv')

#Encode the categorical data

dataset$State = factor(dataset$State,
                       level = c('New York','California','Florida'),
                       labels = c(1,2,3))

#Split data into test and training

library(caTools)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Regression modelling for fitting

regressor = lm(formula = Profit ~ .,
               data = training_set) #how to express profit as a linear combination -> . is all other vars

summary(regressor) #Show summary

#Predict test set results

