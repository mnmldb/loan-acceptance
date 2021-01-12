#  University of California Los Angeles, Extension<br>Machine Learning (COM SCI-X 450.4), Fall 2020<br>Final Project<br>Loan Acceptance Based on Payback Potential
## Problem Statement
In this project, using data from Home Credit, a consumer finance provider, to build a model that can predict whether a loan applicant should receive a loan based on their chances of default. This is a classification problem as we are going to be using supervised machine learning to predict two discrete output values based on a variety of input values. We will be Home Credit’s training dataset to analyze family, loan and asset history of over 300,000 applicants and using relevant independent variables in the training dataset, we will assess each client’s payback potential, which will act as the basis for our two final discrete output values of either ‘accepted’ or ‘rejected’ for a loan. Since this is a classification problem, we will be using relevant classification algorithms including logistic regression, linear discriminant analysis, k-nearest neighbors, and tree-based methods including random forest.

## Data Source
Data is from [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/overview) `application_train.csv` data set. It consists of 16
categorical features, 34 semi-categorical features, 18 integer features, 52 numeric features, and 1 surrogate key.

## Folder Structure
as of December 2nd
~~~
.
├── train_test_split                    # Data splitting
│      └── train_test_split.R           # Split application_train.csv by 70% for training set and 30% for test set and export as train_raw.csv and test_raw.csv
├── eda                                 # Exploratory data analysis
│      ├── eda.R                        # Variable understanding and selection
│      └── (eda_draft.R                 # Draft for eda.R (not used)
├── feature_engineering                 # Feature engineering
│      └── feature_engineering.R        # Processing on train_raw.csv and test_raw.csv and export as train_model.csv and test_model.csv
├── down_sampling                       # Balance target distribution
│      └── down_sampling.R 
├── modeling                            # Main code for modeling
│      ├── modeling_ds.R                # Main code for modeling with down sampled data set of the algorithms identified
│      ├── (modeling.R)                 # Draft for modeling with all data set (not used)
│      └── (baseline.R)                 # Draft for modeling.R (not used)
├── (raw_data)                          # Store raw data downloaded from Kaggle (git ignored)
│      └── application_train.csv
├── (processed_data)                    # Store processed data (git ignored)
│      ├── train_raw.csv
│      ├── test_raw.csv
│      ├── train_model.csv
│      ├── train_model_ds.csv           # Down sampled training data
│      └── test_model.csv
└── README.md
~~~
