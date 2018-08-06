"""
Created on Tue May 29 12:59:11 2018

@author: Ravikiran.Tamiri
"""

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('sales_data_sample.csv',encoding='latin-1')
#dataset.isnull().sum()

X = dataset.iloc[:,[1,2,9,10,11,20,24]].values
Y = dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 5] = labelencoder.fit_transform(X[:, 5])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [3,5,6])
X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train,Y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

#apply k-fold cross validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator=regressor,X = X_train,y = Y_train,cv = 10)
#accuracies = cross_val_score(estimator=regressor,X = X,y = Y,cv = 10)
accuracies.mean()
accuracies.std()

