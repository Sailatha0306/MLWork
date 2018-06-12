# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:00:13 2018

@author: Ravikiran.Tamiri
"""

import gensim

raw_corpus = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",              
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
#list of words to be removed 
stoplist = set('for a of the and to in'.split(' '))

#tokenize words in the list after removing the stopwords 
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in raw_corpus]

from collections import defaultdict

frequency = defaultdict(int)

for text in texts:
    for token in text:
        frequency[token] += 1
        
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

from gensim import corpora
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

print(dictionary.token2id)

bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

from gensim import models
tfidf = models.TfidfModel(bow_corpus)
tfidf[dictionary.doc2bow("system minors".lower().split())]

