import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import os
import string
from tqdm import tqdm

import nltk

nltk.donwload("stopwords")
nltk.download("wordnet")

df = pd.read_parquet("../scraped_data/reviews.parquet.gzip")

stop = set(stopwords.words("english"))
new_stopwords = ["amazon", "phone", "xr", "iphone", "apple"]
for word in new_stopwords:
    stop.add(word)

punc = set(string.punctuation)
stemmer = SnowballStemmer("english")
lemma = WordNetLemmatizer()


def stem_and_lemma(word):
    return stemmer.stem(lemma.lemmatize(word, pos="v"))


def clean(doc):
    # remove punctuations
    for ch in punc:
        doc = doc.replace(ch, " ")
    # text to lower case text
    low_case_doc = [i for i in doc.lower().split()]
    # remove stopwords
    stopwords_punc_free = " ".join([i for i in low_case_doc if i not in stop])
    # stem and lemmatize text
    normalized = " ".join(
        stem_and_lemma(word.strip()) for word in stopwords_punc_free.split()
    )
    return normalized


text_clean = [clean(doc) for doc in tqdm(df["text"])]
title_clean = [clean(doc) for doc in tqdm(df["title"])]

df["title_clean"] = title_clean
df["text_clean"] = text_clean

path = "../cleaned_data"
if not os.path.exists(path):
    os.makedirs(path)

df.to_parquet(os.path.join(path, "cleaned_reviews.parquet.gzip"), index=False)
