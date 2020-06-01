#Polynomial regression
#Given: user asks for $165k - is this a true/good salary? user is between level 6 and 7 

#Import data
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Don't ned to split into test and training in this case (only time - only 10 values)

lin_reg = lm(formula = Salary ~ Level,
             data = dataset)
summary(lin_reg)

#Polynomial features
#Need to add new column with features squared and cubed
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~.,
              data = dataset)
summary(poly_reg)

#Visualise reuslts with ggplot2
library(ggplot2)

#linear regression vis
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(lin_reg, newdata = dataset)),
            color = 'green') +
  ggtitle('Truth or Bluff (linear regression)') +
  xlab('Level')+
  ylab('Salary')

#polynomial regression vis
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(poly_reg, newdata = dataset)),
            color = 'green') +
  ggtitle('Truth or Bluff (polynomial regression)') +
  xlab('Level')+
  ylab('Salary')

#Making predictions:
#Linear
y_pred = predict(lin_reg, data.frame(Level = 6.5)) #Need to make a new dataframe

#Polynomial
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4)) #Need to make a new dataframe
