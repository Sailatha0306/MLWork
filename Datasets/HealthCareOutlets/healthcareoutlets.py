"""
Created on Wed Feb  7 18:03:46 2018

@author: Ravikiran.Tamiri
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

#dropping useless columns
dataset_train.drop(['male','state','credit_score','outlet_no', 'city', 'zip', 'store_location', 'time_zone', 'latitude', 'longitude', 'location_employee_code','credit_score_range'], axis='columns', inplace=True)
dataset_test.drop(['male','state','credit_score','outlet_no', 'city', 'zip', 'store_location', 'time_zone', 'latitude', 'longitude', 'location_employee_code','credit_score_range'], axis='columns', inplace=True)

#check if there are any null values
dataset_train.isnull().sum()
dataset_test.isnull().sum()

X = dataset_train.iloc[:,:-1].values;
y = dataset_train.iloc[:,9].values;

Test_X = dataset_train.iloc[:,:-1].values;
#Test_Y = dataset_train.iloc[:,:9].values;

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
Test_X = sc_X.transform(Test_X)
##PCA step
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 5)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


## Fitting the Regression Model to the dataset
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
#regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)
Test_Y = regressor.predict(Test_X)
#y_pred_train = regressor.predict(X_train)

#apply k-fold cross validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator=regressor,X = X_train,y = Y_train,cv = 10)
accuracies.mean()
accuracies.std()

Result = dataset_test.iloc[:,0].values
Result = [Result, Test_Y]


