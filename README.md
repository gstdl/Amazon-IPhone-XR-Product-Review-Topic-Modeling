## Introduction

Welcome to this repository. This repository contains my code repository in a project related to [Topic Modeling](https://en.wikipedia.org/wiki/Topic_model). Currently, I'm using [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) to build the model.

## Why Should I Do Topic Modeling?

### Scenario

Imagine a scenario where you run a customer service center and your company is receiving at least 100 complains a day and the average complain resolving time is 1 hour per complain. If 1 employee can only work for 10 hour a day, you'll need to employ 10 people to handle all complains with very diverse topics.

As an employer, you have built a mechanism to categorize your customer's complains. However, most of the time, your customer's doesn't care about the mechanism and sends their complain in random pages as long as the page says `"complain"`.

### Problem

1. Customer's complains are not categorized properly even when there's a mechanism to prevent this situation.
2. Employee might spend extra time in resolving complains if they have to face to many diverse topics. In simple words, an employee might spend most of his/her time understanding the varying complains. It will be so much faster if he/she can focus on a particular topic in a row.

### Solution

1. Develop a classification model to identify whether a complain is correctly categorized. This can be completed using Logistic Regression, Support Vector Machine, Tree-based Models, Deep Learning Model, or Ensemble Models. However, this is feasible only when the data is already labelled or annotated.
2. Use an unsupervised machine learning model such as [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation), [Latent Semantic Analysis (LSI)](https://en.wikipedia.org/wiki/Latent_semantic_analysis), [Hierarchical Dirichlet Process (HDP)](https://en.wikipedia.org/wiki/Hierarchical_Dirichlet_process), or other clustering methods. Unlike the first option, this option is always feasible even when the data is unlabelled.
3. A simple if-then rule. If the solutions above doesn't satisfy your topic modeling result or you think that they're overkill, you can always make an if-then rule to categorize your complains.

### Business Impact

Complains received will be pre-sorted based on model output and employees can quickly resolve the issues if they're facing complains with similar topics consequtively. If average complain resolving time is reduced, the number of employees needed to operate the customer service center can be reduced. Thus, minimizing operational cost.

## A Work in Progress

Keep in mind that this is an unfinished project. It contains code only. Please expect the following updates to this repository:

1. Documentation
2. Topic Modeling using other methods
3. A deployed web app in Heroku

You can check this [Kanban Board](https://github.com/users/gstdl/projects/1) for more details.

## Data Source

The data used in this project is **scraped** from [Amazon India's website](https://www.amazon.in). The data used is review data of iPhone XR obtained from [this page](https://www.amazon.in/Apple-iPhone-XR-64GB-White/dp/B07JGXM9WN/ref=cm_cr_arp_d_bdcrb_top?ie=UTF8). The scraped data is saved in [scraped_data folder](scraped_data).

**Why am I using review data when the practical example is customer complains?**

> I'm using review data because complain data is not an easy data obtain and the procedure to build the model will be almost the same.

<!-- **Can I use this data for my personal projects?**

> Yes, you can use this data for your own projects. Besides of topic modeling, you can use it for text classification by using the rating score as label. However, I encourage you to scrape the data by yourself because you'll learn how to scrape data from the internet by doing so. -->

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
2. [Pre-Modeling Data Analysis & Data Cleaning](https://colab.research.google.com/github/gstdl/Amazon-IPhone-XR-Product-Review-Topic-Modeling/blob/master/jupyter_notebooks/2.%20Pre%20Modeling%20Data%20Analysis%20%26%20Data%20Cleaning.ipynb)
3. [Topic Modeling (LDA)](https://colab.research.google.com/github/gstdl/Amazon-IPhone-XR-Product-Review-Topic-Modeling/blob/master/jupyter_notebooks/3.%20Topic%20Modeling%20(LDA).ipynb)