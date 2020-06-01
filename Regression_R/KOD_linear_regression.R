#KOD Simple Linear Regression

#Import dataset

dataset = read.csv('Salary_Data.csv')

#Split data
library(caTools)
set.seed(42)
split = sample.split(dataset$Salary, SplitRatio = 0.8)
training_set = subset(dataset, split ==TRUE)
test_set = subset(dataset, split ==FALSE)

#Build Linear Regression
regressor = lm(formula = Salary ~ YearsExperience, # Salary is proportional to years exp
               data = training_set) #enter summary(regressor) in consol for info on it
                                    #Number of stars shows statistical significance

#Predictions

y_pred = predict(regressor, newdata = test_set) #Choose new data to be the test set


#Visulise results
#install.packages('ggplot2')
library(ggplot2)

ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),#train set plot done with +'s
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
                colour = 'blue') + #regression line
  ggtitle('Salary vs Experience (Training)') +
  xlab('Experience') +
  ylab('Salary')

#Lets look at test set

ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),#train set plot done with +'s
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + #regression line keep training set
  ggtitle('Salary vs Experience (Test)') +
  xlab('Experience') +
  ylab('Salary')

