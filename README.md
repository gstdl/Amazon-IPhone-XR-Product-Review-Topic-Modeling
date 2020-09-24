## Introduction

Welcome to this repository. This repository contains my code repository in a project related to [Topic Modeling](https://en.wikipedia.org/wiki/Topic_model). Currently, I'm using [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) to build the model.

Keep in mind that this is an unfinished project. It contains code only. Please expect the following updates to this repository:

1. Documentation
2. Topic Modeling using other methods
3. A deployed web app in Heroku

## Data Source

The data used in this project is **scraped** from [Amazon India's website](https://www.amazon.in). The data used is review data of iPhone XR obtained from [this page](https://www.amazon.in/Apple-iPhone-XR-64GB-White/dp/B07JGXM9WN/ref=cm_cr_arp_d_bdcrb_top?ie=UTF8).

## Skills Demonstrated

1. Data Scraping / Crawling (Using BeautifulSoup)
2. Data Visualization (Using Wordcloud, Matplotlib, & Plotly)
3. Data Cleaning (Using Pandas, Numpy, & nltk)
4. Natural Language Processing - Topic Modeling (Using gensim)
5. Object Oriented Programming (in Python)

## Accessing the Jupyter Notebooks

Jupyter Notebooks contains step by step procedures in completing this project. It explains my process thought and reasoning behind certain codes.

Please open the Jupyter Notebooks using [Google Colab](https://colab.research.google.com) or by visiting the URLs listed below. [Google Colab](https://colab.research.google.com) is required because some plots does not render outside [Google Colab](https://colab.research.google.com)'s Python environment. If you insist on using a Python environment outside [Google Colab](https://colab.research.google.com), you'll need to delete one code cell and rerun the Jupyter Notebook.

1. [Building the Data Scraper](https://colab.research.google.com/github/gstdl/Amazon-IPhone-XR-Product-Review-Topic-Modeling/blob/master/jupyter_notebooks/1.%20Building%20the%20Data%20Scraper.ipynb)
2. [Pre-Modeling Data Analysis & Data Cleaning](https://colab.research.google.com/github/gstdl/Amazon-IPhone-XR-Product-Review-Topic-Modeling/blob/master/jupyter_notebooks/2.%20Pre-Modeling%20Data%20Analysis%20%26%20Data%20Cleaning.ipynb)
3. [Topic Modeling (LDA)](https://colab.research.google.com/github/gstdl/Amazon-IPhone-XR-Product-Review-Topic-Modeling/blob/master/jupyter_notebooks/3.%20Topic%20Modeling%20(LDA).ipynb)