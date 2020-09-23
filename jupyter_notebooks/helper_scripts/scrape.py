import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import os
from tqdm import tqdm


def grab_review_rating(text):
    return float(text.replace(" out of 5 stars", "").strip())


def grab_review_location_and_date(text):
    location = re.sub("Reviewed in | on \d{1,2} \w+ \d{4}", "", text).strip()
    date = re.findall("\d{1,2} \w+ \d{4}", text)[0]
    return location, date


def scrape(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, "html.parser")
    review_titles = html.find_all(
        "a", class_="review-title", attrs={"data-hook": "review-title"}
    )
    review_dates_and_locations = html.find_all(
        "span", class_="review-date", attrs={"data-hook": "review-date"}
    )
    review_texts = html.find_all(
        "span", class_="review-text", attrs={"data-hook": "review-body"}
    )
    review_ratings = html.find_all(
        "i", class_="review-rating", attrs={"data-hook": "review-star-rating"}
    )
    data = []
    for title, date_and_location, text, rating in zip(
        review_titles, review_dates_and_locations, review_texts, review_ratings
    ):
        title = " ".join([i.strip() for i in title.get_text().split()])
        location, date = grab_review_location_and_date(date_and_location.get_text())
        rating = grab_review_rating(rating.get_text())
        text = " ".join([i.strip() for i in text.get_text().split()])
        data.append([title, date, location, rating, text])
    df = pd.DataFrame(data, columns=["title", "date", "location", "rating", "text"])
    df["date"] = pd.to_datetime(df["date"])
    return df


url = "https://www.amazon.in/Apple-iPhone-XR-64GB-White/product-reviews/B07JGXM9WN/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="

for i in tqdm(range(1, 501), desc="scraping"):
    z = scrape(url + str(i))
    if i == 1:
        df = z.copy()
    else:
        df = df.append(z)

df["rating"] = df["rating"].astype("float32")

path = "../scraped_data"
if not os.path.exists(path):
    os.makedirs(path)

filename = os.path.join(path, "reviews.parquet.gzip")
df.to_parquet(filename, index=False)
