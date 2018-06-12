"""Created on Mon Jun 11 17:22:15 2018

@author: Ravikiran.Tamiri
"""

from bs4 import BeautifulSoup
import requests
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest

min_cut = 0.1
max_cut = 0.9 
my_stopwords = set(stopwords.words('english') + list(punctuation))
  
def _compute_frequencies(word_sent):
    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in my_stopwords:
          freq[word] += 1
    # frequencies normalization and fitering
    m = float(max(freq.values()))
    for w in list(freq):
      freq[w] = freq[w]/m
      if freq[w] >= max_cut or freq[w] <= min_cut:
        del freq[w]
    return freq

def summarize(text, n):
    sents = sent_tokenize(text)
    assert n <= len(sents)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    _freq = _compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in _freq:
          ranking[i] += _freq[w]
    sents_idx = _rank(ranking, n)    
    return [sents[j] for j in sents_idx]

def getContentFromURL(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text

def _rank(ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)

def summarizeURL(url, total_pars):
	url_text = getContentFromURL(url).replace(u"Â", u"").replace(u"â", u"")
	final_summary = summarize(url_text.replace("\n"," "), total_pars)
	return " ".join(final_summary)

def process_url(url):
    final_summary = summarizeURL(url, 5)
    print(final_summary)
    
def process_text(text):
    final_summary = summarize(text.replace("\n"," "), 5)
    print(final_summary)

#main
ret = input("Do you want to enter a \n 1.URL \n 2.Text\n")

if ret == 1:
    url = input("Enter a URL\n")
    process_url(url)
elif ret == 2:
    text = input("Enter the text \n")
    process_text(text)
else:
    print("Enter valid input\n")
    
if ret == 1:
    print("its one")
else:
    print("its two")