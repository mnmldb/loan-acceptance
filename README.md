#  University of California Los Angeles, Extension<br>Machine Learning (COM SCI-X 450.4), Fall 2020<br>Final Project
## Problem Statement
A number of lending institutions struggle to be sure whether certain clients are trustworthy borrowers or are likely to default on their loan due to a lack of sufficient credit.Such institutions require a robust model that could use machine learning to predict a client’s trustworthiness in borrowing based on a set of known facts about them such as their income and credit card balance. This is a very important problem to address because there are vast numbers of unbanked individuals without a significant credit history who are deserving of loans but incapable of receiving them. Creating a model that could assure lenders of the payback potential of borrowers would create significant opportunities, not only for unbanked borrowers, but also for lenders who can expand their business.

Thus in this project using data from Home Credit, a consumer finance providers, we are aiming to build a machine learning algorithm that can predict our dependent variable of whether a borrower is trustworthy based the data set’s independent variables including:
- Income
- Car Ownership
- Occupation
- Credit Amount
- Home Ownership
- Housing Type
- Type of Loan (Cash vs. Revolving)
- Employment Status
- Number of Children
- Annuity Amount
- Education Level
- Marital Status

## Data Source and Prep
Data is from [Home Credit](https://www.kaggle.com/c/home-credit-default-risk/overview) Default Risk (121 independent variables in the application_train data set plus additional independent variables in supplementary files): *https://www.kaggle.com/c/home-credit-default-risk/overview*

67 out of 121 columns in the data set have missing values. For the models that do not work with missing values, we are going to fill them by some representative values (mean, median, etc.) or forecasted values by other variables. 6 out of 121 columns in the data set consist of categorical variables. The number of levels differs from 2 to 58. We are going to use appropriate encoding methods such as one-hot encoding, label encoding, frequency encoding, and target encoding depending on the feature of each column.

We assume a client’s income and loan credit amount will have a strong linear relationship to payback capability and difficulty. We also believe that employment status and home ownership status will have a somewhat weaker but nonetheless significant influence on the client’s ability to qualify for a loan.

We would like to identify the outliers of numerical variables using histograms and replace them with the less influential variables. It could be very important to detect collinearity by looking at the correlation matrix or referring to the variance inflation factor (VIF). We would also like to create visualization graphs such as scatter plots and bar graphs of historical data to explore the covariance between the most significant variables.

## Modeling and Assessment Strategy
We will be using multilinear regression to predict the payback potential of each client based on various independent variables. As the independent variable is either of 0 (no payback) or 1 (payback), we will see the model performance by area under the ROC curve. We will be using algorithms such as Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, K-Nearest Neighbors, Support Vector Machines, Random Forest, Gradient Boosting Decision Tree (e.g. XGBoost), and Neural Network covered in “An Introduction to Statistical Learning” and popular algorithms in machine learning competitions. Also, we will be testing the ensemble method of several models to see if it would improve the score.

## Folder Structure
~~~
.
├── EDA            # Exploratory data analysis
├── Modeling       # Main code for modeling
└── README.md
~~~