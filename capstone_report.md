# Machine Learning Engineer Nanodegree
## Capstone Project
Jair Miranda  
April 29st, 2021

### Project Overview

Customer segmentation looking to general population data, and Arvato Company clients data,
the idea identify with general population data those who have the greatest chance of becoming customers.
In the project I used a lot of different algorithms and also created pipelines to clean up the data, 
and made some feature engineering.

### Problem Statement

Customer segmentation is a common problem on machine learning projects, mainly if you're looking to improve the
marketing campaign results.
Arvato Financial Solucions has a lot of demographic and customer data, and we ll try to find patterns on this data, 
understanding the difference between customers and non-customers.
We have train, test dataset too, but all 4 datasets have a serious problem with nullity data, and wrong types.
The most interesting thing about this project is that it is a real problem, and we can violate it in any company,
besides that having a good result can generate high profitability for the company

It was hard to deal with all data, heavy for the memory.

### Data and Inputs

There are four data files associated with this project:

- `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
- `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
- `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
- `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

### Solution Statement.

We ll start analyse the data and use catboost to check feature importance
The goal is to create a model or model pipeline to understand what's the change of new lead to be a client for that we ll:
   
    1- Analyse the Data.
    2- Clean Up the Data.
    3- Create Data Wrangler pipeline.
    4- Make Unsupervided Analyses.
    5- Create Unsupervided Pipeline.
    6- Make Supervised Learning.
    7- Create Supervised Pipeline.
    8- Evaluate.

### Metrics

Measuring Performance: AUC (AUROC)
![alt text](https://glassboxmedicine.files.wordpress.com/2019/02/roc-curve-v2.png)

### Analyses

I used Azdias dataset to make analyses

Most of the columns have less than 20% missing values.
I drop those columns that have greater than 20% missing values
And below graph shows the top 50 names of columns along with its missing percent.

![alt text](https://glassboxmedicine.files.wordpress.com/2019/02/roc-curve-v2.png)


