#random Forest Decision Tree
# Importing the libraries
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import re

# Importing the dataset
dataset = pd.read_csv('bmv_training_set.csv')
X = dataset.iloc[:10000, 1]
y = dataset.iloc[:10000, 2]

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def text_process(text):
    text = re.sub('\d',"",text)
    for c in string.punctuation:
        text = text.replace(c," ")
    filtered_words = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(filtered_words)

for i in range(0,len(X)):
    X[i] = text_process(X[i])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(X_train, y_train)
predicted_svm = text_clf_svm.predict(X_test)
np.mean(predicted_svm == y_test)
