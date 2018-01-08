# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#fitting LR to training set
regressor = lm(formula = Salary ~ YearsExperience, 
               training_set)
#predit the test set results
y_pred = predict(regressor,newdata = test_set)

#visualize the training set results
#install.packages('ggplot2')
library('ggplot2')
ggplot()+
  geom_point()


