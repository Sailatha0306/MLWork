#polynomial regression
# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

# Fitting the linear regression to dataset
lin_reg = lm( formula = Salary ~.,
              data = dataset)

# Fitting the polynomial regression to dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3

ploy_reg = lm( formula = Salary ~.,
              data = dataset)

#visualize linear regression
#library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level,y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear regression)') +
  xlab('Level')+
  ylab('Salary')

#visualize polynomial regression
#library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level,y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(ploy_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial regression)') +
  xlab('Level')+
  ylab('Salary')

#predict salary using linear regression
y_pred = predict(lin_reg,data.frame(Level = 6.5))


#predict salary using polynomial regression
y_pred = predict(ploy_reg,data.frame(Level = 6.5,
                                     Level2 = 6.5^2,
                                     Level3 = 6.5^3, 
                                     Level4 = 6.5^4))


