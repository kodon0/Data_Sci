#Random Forest Regression 

#Given: user asks for $165k - is this a true/good salary? user is between level 6 and 7 
#Import data
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Feature Scaling
# No train/test split -> want full set for close as possible fitting

#Create DTR model:
#install.packages('randomForest')
library(randomForest)
rfr = randomForest(x = dataset[1],
                   y = dataset$Salary,
                   ntree = 35)

#Fit DTR model:

#Making predictions:
y_pred = predict(rfr, data.frame(Level = 6.5)) #Need to make a new dataframe

#Visualise reuslts with ggplot2
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01) #Optional -> makes smoother curves
ggplot() + 
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = x_grid , y = predict(rfr, newdata = data.frame(Level=x_grid))),
            color = 'green') +
  ggtitle('Random Forest model') +
  xlab('Level') +
  ylab('Salary')

