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

import string
import math

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

dataset = pd.read_csv("AAAI-13AcceptedPapers.csv")
#print(dataset['Title'])
#print(dataset['Abstract'][149])

#X = dataset['High-Level Keyword(s)']
#X = dataset['High-Level Keyword(s)']
#X = dataset['Keywords']
X = dataset['Topics']

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    filtered_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(filtered_words)
    #return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    #return [stemmer.stem(t) for t in filtered_words]

for i in range(0,len(X)):
    X[i] = text_process(X[i])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidvectorizer = TfidfVectorizer()
#tfidvectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,min_df=0.2,use_idf=True, tokenizer=text_process, ngram_range=(1,3))
tfidf_matrix = tfidvectorizer.fit_transform(X)

print(tfidf_matrix.shape)

terms = tfidvectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

#KMeans 

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 150):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(tfidf_matrix)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,150), wcss,'.')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#kmeans = KMeans(n_clusters = 20, init = 'k-means++', random_state = 42)
#y_kmeans = kmeans.fit_predict(tfidf_matrix)




