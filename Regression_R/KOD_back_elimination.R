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

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set) #how to express profit as a linear combination -> . is all other vars

summary(regressor) #Show summary -> look at which have highest P values? remove them

#Repeat with highest P value omitted -> summary suggest all but R&D 

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = training_set) 
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = training_set) 
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set) 
summary(regressor)

#Automatic backwards elimination in R:

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
#Predict test set results
y_pred

