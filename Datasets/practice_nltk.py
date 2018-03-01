import nltk
nltk.download('punkt')
#tokenizing
####word tokenizers
###sentence tokenizers
#lexicons -words and meanings.
#corporas - medical journals,

from nltk.tokenize import sent_tokenize, word_tokenize

example_token = "Hello there, how are you today? My modelling an."
print(sent_tokenize(example_token))
print(word_tokenize(example_token))

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

words = word_tokenize(example_token)

#filtered_sentences = []
#
#for w in words:
#    if w not in stop_words:
#        filtered_sentences.append(w)

filtered_sentences = [w for w in words if not w in stop_words ]     
print(filtered_sentences)

#stemming-eg: riding = rid
from nltk.stem import PorterStemmer

ps = PorterStemmer()

eg = ["python","pythonly","pythonise"]

for w in eg:
    print(ps.stem(w))

for w in words:
    print(ps.stem(w))

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
nltk.download('state_union')
nltk.download('averaged_perceptron_tagger')

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

cus_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = cus_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>?} """
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
    except Exception as e:
        print(str(e))

process_content()       
