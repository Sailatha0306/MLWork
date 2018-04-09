# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York','California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#fit multiple LR to training set
regressor = lm(formula = Profit ~ .,
               data = training_set)

#predicting the test results
y_pred = predict(regressor,newdata = test_set)

#building the model using backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)
#backward elimination using a function

backward_elimination <- function(x,sl){
  numVars = length(x)
  for(i in c(1:numVars)){
    regressor = lm(formula = Profit ~. , data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars),"Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars),"Pr(>|t|)"] == maxVar)
      x = x[,-j]
    }
    numVars = numVars -1
  }
  return(summary(regressor))
}


sl = 0.05
dataset = dataset[,c(1,2,3,4,5)]
backward_elimination(training_set,sl)