"""
Created on Fri Mar  2 15:39:26 2018

@author: Ravikiran.Tamiri
"""
import glob,string
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import datetime
print(datetime.datetime.utcnow())

filenames = sorted(glob.glob("*/*.*"))
df = pd.DataFrame()
raw_data = []

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

for filename in filenames:
    with open(filename, "r") as book_file:
            data = book_file.read()
            raw_data.append(data)


X = raw_data

print(datetime.datetime.utcnow())
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    filtered_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    #return ' '.join(filtered_words)
    #return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    stem_words = [stemmer.stem(t) for t in filtered_words]
    return ' '.join(stem_words)

print(datetime.datetime.utcnow())
for i in range(0,len(X)):
    X[i] = text_process(X[i])

print(datetime.datetime.utcnow())

from sklearn.feature_extraction.text import TfidfVectorizer
#tfidvectorizer = TfidfVectorizer()
tfidvectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,min_df=0.2,use_idf=True, tokenizer=text_process, ngram_range=(1,3))
tfidf_matrix = tfidvectorizer.fit_transform(X)

print(tfidf_matrix.shape)
print(datetime.datetime.utcnow())



terms = tfidvectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

#KMeans 
print(datetime.datetime.utcnow())
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 2225):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(tfidf_matrix)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,2225), wcss,'.')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
print(datetime.datetime.utcnow())




#kmeans = KMeans(n_clusters = 20, init = 'k-means++', random_state = 42)
#y_kmeans = kmeans.fit_predict(tfidf_matrix)