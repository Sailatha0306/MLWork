# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:56:22 2018

@author: Ravikiran.Tamiri
"""

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

import nltk
nltk.download('stopwords') # Let's make sure the 'stopword' package is downloaded & updated
nltk.download('wordnet') # Let's also download wordnet, which will be used for lemmatization

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint
from smart_open import smart_open

train_file = 'politics.txt'

with smart_open(train_file, 'rb') as f:
    for n, l in enumerate(f):
        if n < 8:
            print([l])
            
def build_texts(fname):
    with smart_open(fname, 'rb') as f:
        for line in f:
            yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)
            
train_texts = list(build_texts(train_file))

bigram = gensim.models.Phrases(train_texts)

from gensim.utils import lemmatize
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

def process_texts(texts):
    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [bigram[line] for line in texts]
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    texts = [[word for word in lemmatizer.lemmatize(' '.join(line), pos='v').split()] for line in texts]
    return texts

train_texts = process_texts(train_texts)

dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]

#LSI modelling
lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
lsimodel.show_topics(num_topics=5) 
lsitopics = lsimodel.show_topics(formatted=False)


#HDP modelling
hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
hdpmodel.show_topics()
hdptopics = hdpmodel.show_topics(formatted=False)

#LDA modelling
ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
ldatopics = ldamodel.show_topics(formatted=False)

def  evaluate_graph (dictionary, corpus, texts, limit):
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()
    
    return lm_list, c_v

lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=train_texts, limit=10)

pyLDAvis.gensim.prepare(lmlist[2], corpus, dictionary)

lmtopics = lmlist[5].show_topics(formatted=False)

#LDA as LSI

def ret_top_model():
    top_topics = [(0, 0)]
    while top_topics[0][1] < 0.97:
        lm = LdaModel(corpus=corpus, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=train_texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics

lm, top_topics = ret_top_model()

print(top_topics[:5])

lsitopics = [[word for word, prob in topic] for topicid, topic in lsitopics]

hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]

ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]

lmtopics = [[word for word, prob in topic] for topicid, topic in lmtopics]


###NOT COMPLETED###
