from nltk.stem import SnowballStemmer

# import spacy
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS as stop
import os
import string
import nltk

try:
    stemmer = SnowballStemmer("english")
except:
    nltk.download("wordnet")
    stemmer = SnowballStemmer("english")

try:
    nlp = en_core_web_sm.load()
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = en_core_web_sm.load()

stop.add("phone")

punc = set(string.punctuation)


def clean(doc):
    # lower text and remove punctuations
    s = ""
    for char in doc.lower():
        s += char if char not in punc else " "
    # remove stopwords, adjectives, and adverbs
    normalized = []
    for token in nlp(s):
        if not (
            token.is_space
            or token.is_stop
            or token.pos_ == "ADJ"
            or token.pos_ == "ADV"
        ):
            # stem and lemmatize text
            t = stemmer.stem(token.lemma_)
            normalized.append(t)
    normalized = " ".join(normalized)
    return normalized
