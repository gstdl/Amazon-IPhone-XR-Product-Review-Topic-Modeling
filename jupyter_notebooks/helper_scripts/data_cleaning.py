import pandas as pd
import os
from tqdm import tqdm
from clean_func import clean

df = pd.read_parquet("../scraped_data/reviews.parquet.gzip")

text_clean = [clean(doc) for doc in tqdm(df["text"])]
title_clean = [clean(doc) for doc in tqdm(df["title"])]

df["title_clean"] = title_clean
df["text_clean"] = text_clean

df = df[(df["text_clean"].apply(lambda x: x.strip()) != "")]

path = "../cleaned_data"
if not os.path.exists(path):
    os.makedirs(path)

df.to_parquet(os.path.join(path, "cleaned_reviews.parquet.gzip"), index=False)
