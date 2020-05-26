#Decision Tree Regression 

#Given: user asks for $165k - is this a true/good salary? user is between level 6 and 7 
#Import data
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Feature Scaling
# No train/test split -> want full set for close as possible fitting

#Create DTR model:
#install.packages('rpart')
library(rpart)
dtr = rpart(formula = Salary ~.,
          data = dataset,
          control = rpart.control(minsplit = 1))

#Fit DTR model:

#Making predictions:
y_pred = predict(dtr, data.frame(Level = 6.5)) #Need to make a new dataframe

#Visualise reuslts with ggplot2
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01) #Optional -> makes smoother curves
ggplot() + 
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = x_grid , y = predict(dtr, newdata = data.frame(Level=x_grid))),
            color = 'green') +
  ggtitle('DTR model') +
  xlab('Level') +
  ylab('Salary')

