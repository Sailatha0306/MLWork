"""
Created on Wed Feb 28 15:57:20 2018

@author: Ravikiran.Tamiri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords

yelp = pd.read_csv("yelp.csv")

yelp.shape
yelp.head()
yelp.info()

#adding the column 'text_length'
yelp['text_length'] = yelp.text.apply(len)

g = sns.FacetGrid(data=yelp,col='stars')
g.map(plt.hist,'text_length',bins =50)

sns.boxplot(x = 'stars', y = 'text_length', data = yelp)

stars = yelp.groupby('stars').mean()
stars.corr()

sns.heatmap(data=stars.corr(),annot=True)

#good and bad reviews - 1 & 5 stars
yelp_class = yelp[(yelp['stars'] == 1)|(yelp['stars'] == 5)]

X = yelp_class['text']
Y = yelp_class['stars']

import string

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer = text_process).fit(X)

len(bow_transformer.vocabulary_)
r_25 = X[24]
bow_25 = bow_transformer.transform([r_25])

print(bow_transformer.get_feature_names()[11443])

X = bow_transformer.transform(X)

print('Shape of sparse matrix ',X.shape)

print('Non zero occurances ',X.nnz)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

preds = nb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test,preds)
cr = classification_report(y_test,preds)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, preds)




