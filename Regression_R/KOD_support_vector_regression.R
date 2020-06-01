#Support Vector Regression

#Given: user asks for $165k - is this a true/good salary? user is between level 6 and 7

#Import data
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Feature Scaling
# No train/test split -> want full set for close as possible fitting

#Create SVR model:
#install.packages('e1071')
library(e1071)
svr = svm(formula = Salary ~.,
          data = dataset,
        type = 'eps-regression')

#Fit SVR model:

#Making predictions:
y_pred = predict(svr, data.frame(Level = 6.5)) #Need to make a new dataframe

#Visualise reuslts with ggplot2
library(ggplot2)
x_grid = seq(dataset$Level), max(dataset$Level), 0.1) #Optionoal -> makes smoother curves
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(svr, newdata = dataset)),
            color = 'green') +
  ggtitle('SVR model') +
  xlab('Level')+
  ylab('Salary')
