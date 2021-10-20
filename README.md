#  Loan Acceptance Based on Payback Potential
## Problem Statement
In this project, using data from Home Credit, a consumer finance provider, to build a model that can predict whether a loan applicant should receive a loan based on their chances of default. This is a classification problem as we are going to be using supervised machine learning to predict two discrete output values based on a variety of input values. We will be Home Credit’s training dataset to analyze family, loan and asset history of over 300,000 applicants and using relevant independent variables in the training dataset, we will assess each client’s payback potential, which will act as the basis for our two final discrete output values of either ‘accepted’ or ‘rejected’ for a loan. Since this is a classification problem, we will be using relevant classification algorithms including logistic regression, linear discriminant analysis, k-nearest neighbors, and tree-based methods including random forest.

## Data Source
Data is from [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/overview) `application_train.csv` data set. It consists of 16
categorical features, 34 semi-categorical features, 18 integer features, 52 numeric features, and 1 surrogate key. As a result of exploratory data analysis and feature engineering, we used 21 features to train the model in total.

## Folder Structure
~~~
.
├── train_test_split                    # Data splitting
│      └── train_test_split.R           # Split application_train.csv by 70% for training set and 30% for test set and export as train_raw.csv and test_raw.csv
├── eda                                 # Exploratory data analysis
│      ├── eda.R                        # Variable understanding and selection
│      └── (eda_draft.R)                # Draft for eda.R (not used)
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

## Result
We used the Area Under the ROC (AUC) curve as the assessment metrics. Below charts show the ROC curve of the best models in 4 algorithms.

![roc](https://user-images.githubusercontent.com/47055092/138018962-4ed64dae-57ea-4bab-be7b-f978de0332af.png)


Comparing the four algorithms used, we find that Logistic Regression and Linear Discriminant Analysis performed the best as those models had the highest accuracy scores and best-performing ROC curves. Logistic Regression and Linear Discriminant Analysis performed almost identical to each other, which would be attributed to the fact that our dependent variable only has two classes of responses and our sample size is quite large. Furthermore, we find that using three of the most predictive features, as opposed using all the features, provides the most accurate results in both Logistic Regression and Linear Discriminant Analysis.
While K-Nearest Neighbors and Tree-based methods do not perform extremely worse than the first two models, their accuracy scores do not measure up to the first two, perhaps due to their non-parametric approach with a data set that has features with a more linear relationship. These models may also attempt high overfitting due to the type of the data set, which results in a poorly performing model. This is in contrast to our literature review findings that specify random forest as the best predictor for loan borrowing data.
Based on the above findings, Logistic Regression and Linear Discriminant Analysis are recommended for the Home Credit Default Risk data set.

![result](https://user-images.githubusercontent.com/47055092/138019028-98162ddb-5332-4d76-9fb5-f7007fc23ef7.png)