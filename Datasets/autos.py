"""
Created on Mon Feb  5 21:19:35 2018

@author: Ravikiran.Tamiri
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('autos.csv',encoding='latin-1')
dataset.drop(['name','dateCrawled','seller', 'offerType', 'abtest','model','nrOfPictures', 'lastSeen', 'postalCode', 'dateCreated'],axis='columns',inplace = 'True')

#remove the duplicates
dedups = dataset.drop_duplicates(['price','vehicleType','yearOfRegistration'
                         ,'gearbox','powerPS','kilometer','monthOfRegistration','fuelType'
                         ,'notRepairedDamage'])
dedups = dedups[
        (dedups.yearOfRegistration <= 2016) 
      & (dedups.yearOfRegistration >= 1950) 
      & (dedups.price >= 100) 
      & (dedups.price <= 150000) 
      & (dedups.powerPS >= 10) 
      & (dedups.powerPS <= 500)]


dedups['notRepairedDamage'].fillna(value='not-declared', inplace=True)
dedups['fuelType'].fillna(value='not-declared', inplace=True)
dedups['gearbox'].fillna(value='not-declared', inplace=True)
dedups['vehicleType'].fillna(value='not-declared', inplace=True)

dedups.isnull().sum()

#dropping all the rows which has any NAN values 
#dataset = dataset.dropna(axis=0)

#check if all the null are dropped
#check the unique values
#dataset.notRepairedDamage.unique()
#dataset.offerType.unique()
#dataset.abtest.unique()
#dataset.notRepairedDamage.unique()

labels = ['gearbox', 'notRepairedDamage', 'brand', 'fuelType', 'vehicleType']
les = {}


from sklearn import preprocessing

for l in labels:
    les[l] = preprocessing.LabelEncoder()
    les[l].fit(dedups[l])
    tr = les[l].transform(dedups[l]) 
    dedups.loc[:, l + '_feat'] = pd.Series(tr, index=dedups.index)
    
labeled = dedups[ ['price'
                        ,'yearOfRegistration'
                        ,'powerPS'
                        ,'kilometer'
                        ,'monthOfRegistration'] 
                    + [x+"_feat" for x in labels]]
    
corr_mat = labeled.corr()

labeled.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
Y = labeled['price']
X = labeled.drop(['price'], axis='columns', inplace=False)

###############################################################################

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)


# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

y_test = y_test.values
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)





