# ===================================================================================================
# 1. Preparation
# ===================================================================================================
# Import packages
library(dplyr)
library(ggplot2)
library(ggsci)
library(makedummies)
library(ROCR)
library(MASS)
library(class)
library(randomForest)
library(tree)
library(leaps)
library(car)
library(glmnet)

# Import training and test data
df_train_all <- read.csv("./processed_data/train_model_ds.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE)
df_test_all <- read.csv("./processed_data/test_model.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE)

# Columns used for modeling
col_use <- c(
    "TARGET"
  , "DAYS_BIRTH" # integer
  , "DAYS_EMPLOYED" # integer
  , "DAYS_ID_PUBLISH" # integer
  , "DAYS_LAST_PHONE_CHANGE" #integer
  , "DAYS_REGISTRATION" # numeric
  , "AMT_INCOME_TOTAL" # numeric
  , "AMT_CREDIT" # numeric
  , "AMT_ANNUITY" # numeric
  , "AMT_GOODS_PRICE" # numeric
  , "EXT_SOURCE_1" # numeric
  , "EXT_SOURCE_2" # numeric
  , "EXT_SOURCE_3" # numeric
  , "CODE_GENDER_F" # character (categorical)
  , "ANOMALY_DAYS_EMPLOYED"
  , "SQUARE_DAYS_BIRTH"
  , "SQUARE_EXT_SOURCE_1"
  , "SQUARE_EXT_SOURCE_2"
  , "SQUARE_EXT_SOURCE_3"
  , "INTER_EXT_SOURCE_1_2"
  , "INTER_EXT_SOURCE_2_3"
  , "INTER_EXT_SOURCE_3_1"
  # , "CODE_GENDER_M" # need to remove due to the collinearity
)

# How to find collinearity
# model_col <- glm(TARGET~., data=df_train, family=binomial)
# vif(model_col)
# Error in vif.default(model_col) : 
#  there are aliased coefficients in the model
# alias(model_col) # Collinearity between CODE_GENDER_F and CODE_GENDER_M

# Create new data frames used for modeling
df_train <- df_train_all[, colnames(df_train_all) %in% col_use]
df_test <- df_test_all[, colnames(df_test_all) %in% col_use]

# ==================================================================================================
# 2. Area under the ROC Curve
# ===================================================================================================
# Function to plot the ROC curve
plot_auc <- function(pred, target) {
  prep <- prediction(pred, target)
  perf <- performance(prep, "tpr", "fpr")
  plot(perf)
}

# Function to return the AUC
calc_auc <- function(pred, target) {
  prep <- prediction(pred, target)
  auc <- performance(prep, "auc")
  auc <- as.numeric(auc@y.values)
  return(auc)
}

# ===================================================================================================
# 3. Logistic Regression
# ===================================================================================================
# Test: all variables
model_lr <- glm(TARGET~., data=df_train, family=binomial)
pred_train_lr <- predict(model_lr, type="response")
pred_test_lr <- predict(model_lr, newdata=df_test, type="response")
calc_auc(pred_train_lr, df_train$TARGET) # score: 0.7329133
calc_auc(pred_test_lr, df_test$TARGET) # score: 0.6425435

# Test: 4 variables
model_lr_4 <- glm(TARGET~DAYS_BIRTH+EXT_SOURCE_1+EXT_SOURCE_2+EXT_SOURCE_3, data=df_train, family=binomial)
pred_train_lr_4 <- predict(model_lr_4, type="response")
pred_test_lr_4 <- predict(model_lr_4, newdata=df_test, type="response")
calc_auc(pred_train_lr_4, df_train$TARGET) # score: 0.7184532
calc_auc(pred_test_lr_4, df_test$TARGET) # score: 0.7107454

# Test: 3 variables
model_lr_3 <- glm(TARGET~EXT_SOURCE_1+EXT_SOURCE_2+EXT_SOURCE_3, data=df_train, family=binomial)
pred_train_lr_3 <- predict(model_lr_3, type="response")
pred_test_lr_3 <- predict(model_lr_3, newdata=df_test, type="response")
calc_auc(pred_train_lr_3, df_train$TARGET) # score: 0.7177558
calc_auc(pred_test_lr_3, df_test$TARGET) # score: 0.7146918
plot_auc(pred_test_lr_3, df_test$TARGET) # plot for the paper

# Test: 2 variables
model_lr_2 <- glm(TARGET~EXT_SOURCE_2+EXT_SOURCE_3, data=df_train, family=binomial)
pred_train_lr_2 <- predict(model_lr_2, type="response")
pred_test_lr_2 <- predict(model_lr_2, newdata=df_test, type="response")
calc_auc(pred_train_lr_2, df_train$TARGET) # score: 0.7061576
calc_auc(pred_test_lr_2, df_test$TARGET) # score: 0.7039289

# Test: 1 variables
model_lr_1 <- glm(TARGET~EXT_SOURCE_2, data=df_train, family=binomial)
pred_train_lr_1 <- predict(model_lr_1, type="response")
pred_test_lr_1 <- predict(model_lr_1, newdata=df_test, type="response")
calc_auc(pred_train_lr_1, df_train$TARGET) # score: 0.6568346
calc_auc(pred_test_lr_1, df_test$TARGET) # score: 0.6463267

#--- Best subset selection with cross validation ---#
num_var <- dim(df_train)[2] - 1 # TARGET excluded

# Prepare folds and matrix to input RSS
k <- 10 # number of folds
set.seed(1)
folds <- sample(1:k, nrow(df_train), replace=TRUE)
cv_errors <- matrix(NA, k, num_var, dimnames=list(NULL, paste(1:num_var)))

# Create function for prediction with regsubsets
predict.regsubsets <- function (object ,newdata ,id ,...){
  form=as.formula(object$call [[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object ,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
  }

# Loop folds and calculate each RSS
for (j in 1:k) {
  best_fit <- regsubsets(TARGET~., data=df_train[folds != j, ], nvmax=num_var)
  for (i in 1:num_var) {
    pred <- predict(best_fit, df_train[folds == j, ], id=i)
    cv_errors[j, i] <- mean((df_train$TARGET[folds == j] - pred) ^ 2)
  }
}

# Confirm the best number of features
mean_cv_errors <- apply(cv_errors ,2,mean)
par(mfrow=c(1,1))
plot(mean_cv_errors ,type='b')
best_var <- which.min(mean_cv_errors) # 7

# Create the best model
reg_best <- regsubsets(TARGET~., data=df_train, nvmax=num_var)
coef(reg_best, best_var) # AMT_CREDIT, AMT_GOODS_PRICE, DAYS_EMPLOYED, EXT_SOURCE_1, EXT_SOURCE_3, INTER_EXT_SOURCE_1_2, CODE_GENDER_F
model_lr_best <- glm(TARGET~AMT_CREDIT+AMT_GOODS_PRICE+DAYS_EMPLOYED+EXT_SOURCE_1+EXT_SOURCE_3+INTER_EXT_SOURCE_1_2+CODE_GENDER_F, data=df_train, family=binomial)
pred_train_lr_best <- predict(model_lr_best, type="response")
pred_test_lr_best <- predict(model_lr_best, newdata=df_test, type="response")
calc_auc(pred_train_lr_best, df_train$TARGET) # score: 0.7281696
calc_auc(pred_test_lr_best, df_test$TARGET) # score: 0.681197
plot_auc(pred_test_lr_best, df_test$TARGET)

#--- Ridge ---#
x <- model.matrix(TARGET~., df_train)[,-1]
y <- df_train$TARGET
grid <- 10 ^ seq(10, -2, length=100)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)

# Set index for validation
set.seed (1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]

# Validation to get the best lambda
set.seed (1)
cv.out.ridge <- cv.glmnet(x[train ,],y[train],alpha=0)
plot(cv.out.ridge)
bestlam_ridge <- cv.out.ridge$lambda.min # 0.8777197

# Train the best ridge model 
model_lr_ridge <- glmnet(x,y,alpha=0, lambda = grid)
predict(model_lr_ridge,type="coefficient", s=bestlam_ridge) # coefficient estimates
df_test_ridge <- df_test[, -(colnames(df_test) %in% col_use)]
pred_train_lr_ridge <- predict(model_lr_ridge,newx=x, s=bestlam_ridge, type = 'response')
pred_test_lr_ridge <- predict(model_lr_ridge,newx=as.matrix(df_test_ridge), s=bestlam_ridge, type = 'response')
calc_auc(pred_train_lr_ridge, df_train$TARGET) # score: 0.723859
calc_auc(pred_test_lr_ridge, df_test$TARGET) # score: 0.6777627

#--- Lasso ---#
lasso.mod <- glmnet(x[train ,],y[train],alpha=1,lambda=grid)
plot(lasso.mod)

# Validation to get the best lambda
set.seed (1)
cv.out.lasso <- cv.glmnet(x[train ,],y[train],alpha=1)
plot(cv.out.lasso)
bestlam_lasso <- cv.out.lasso$lambda.min # 0.01722999

# Train the best lasso model
model_lr_lasso <- glmnet(x,y,alpha=1, lambda = grid)
predict(model_lr_lasso,type="coefficient", s=bestlam_lasso) # coefficient estimates
df_test_lasso <- df_test[, -(colnames(df_test) %in% col_use)]
pred_train_lr_lasso <- predict(model_lr_lasso,newx=x, s=bestlam_lasso, type = 'response')
pred_test_lr_lasso <- predict(model_lr_lasso,newx=as.matrix(df_test_lasso), s=bestlam_lasso, type = 'response')
calc_auc(pred_train_lr_lasso, df_train$TARGET) # score: 0.722326
calc_auc(pred_test_lr_lasso, df_test$TARGET) # score: 0.6781371

#--- The Best Model ---#
summary(model_lr_3)

# ===================================================================================================
# 4. Linear Discriminant Analysis
# ===================================================================================================
# Test: all variables
model_lda <- lda(TARGET~., data=df_train)
model_lda
pred_train_lda <- predict(model_lda, newdata=df_train)
pred_test_lda <- predict(model_lda, newdata=df_test)

# <reference> https://stackoverflow.com/questions/41533811/roc-curve-in-linear-discriminant-analysis-with-r
calc_auc(pred_train_lda$posterior[,2], df_train$TARGET) # score: 0.7327251
calc_auc(pred_test_lda$posterior[,2], df_test$TARGET) # score: 0.6536142
plot_auc(pred_test_lda$posterior[,2], df_test$TARGET)

# Test: 4 variables
model_lda_4 <- lda(TARGET~DAYS_BIRTH+EXT_SOURCE_1+EXT_SOURCE_2+EXT_SOURCE_3, data=df_train)
model_lda_4
pred_train_lda_4 <- predict(model_lda_4, newdata=df_train)
pred_test_lda_4 <- predict(model_lda_4, newdata=df_test)

# <reference> https://stackoverflow.com/questions/41533811/roc-curve-in-linear-discriminant-analysis-with-r
calc_auc(pred_train_lda_4$posterior[,2], df_train$TARGET) # score: 0.7184173
calc_auc(pred_test_lda_4$posterior[,2], df_test$TARGET) # score: 0.7102984
plot_auc(pred_test_lda_4$posterior[,2], df_test$TARGET)

# Test: 4 variables
model_lda_3 <- lda(TARGET~EXT_SOURCE_1+EXT_SOURCE_2+EXT_SOURCE_3, data=df_train)
model_lda_3
pred_train_lda_3 <- predict(model_lda_3, newdata=df_train)
pred_test_lda_3 <- predict(model_lda_3, newdata=df_test)

# <reference> https://stackoverflow.com/questions/31533811/roc-curve-in-linear-discriminant-analysis-with-r
calc_auc(pred_train_lda_3$posterior[,2], df_train$TARGET) # score: 0.7177225
calc_auc(pred_test_lda_3$posterior[,2], df_test$TARGET) # score: 0.7145962
plot_auc(pred_test_lda_3$posterior[,2], df_test$TARGET) # plot for the paper

# Best model correlation with Logistic Regression
cor(pred_test_lr_3, pred_test_lda_3$posterior[, 2]) # 0.9999222
plot(pred_test_lr_3, pred_test_lda_3$posterior[, 2], xlab="Logistic Regression", ylab="LDA")

# ===================================================================================================
# 5. K-Nearest Neighbors
# ===================================================================================================
# Test: all variables
train.X <- df_train[, -which(colnames(df_train) == "TARGET")]
test.X <- df_test[, -which(colnames(df_test) == "TARGET")]
train.Y <- df_train$TARGET

# https://stackoverflow.com/questions/40783331/rocr-error-format-of-predictions-is-invalid

pred_train_knn_1 <- knn(train.X, train.X, train.Y, k=1, prob=TRUE)
pred_test_knn_1 <- knn(train.X, test.X, train.Y, k=1, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_1), df_train$TARGET) # score: 1
calc_auc(as.numeric(pred_test_knn_1), df_test$TARGET) # score: 0.5564606

pred_train_knn_3 <- knn(train.X, train.X, train.Y, k=3, prob=TRUE)
pred_test_knn_3 <- knn(train.X, test.X, train.Y, k=3, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_3), df_train$TARGET) # score: 0.7975632
calc_auc(as.numeric(pred_test_knn_3), df_test$TARGET) # score: 0.578347

pred_train_knn_5 <- knn(train.X, train.X, train.Y, k=5, prob=TRUE)
pred_test_knn_5 <- knn(train.X, test.X, train.Y, k=5, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_5), df_train$TARGET) # score: 0.7501465
calc_auc(as.numeric(pred_test_knn_5), df_test$TARGET) # score: 0.5856481

pred_train_knn_10 <- knn(train.X, train.X, train.Y, k=10, prob=TRUE)
pred_test_knn_10 <- knn(train.X, test.X, train.Y, k=10, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_10), df_train$TARGET) # score: 0.709371
calc_auc(as.numeric(pred_test_knn_10), df_test$TARGET) # score: 0.5943738

pred_train_knn_20 <- knn(train.X, train.X, train.Y, k=20, prob=TRUE)
pred_test_knn_20 <- knn(train.X, test.X, train.Y, k=20, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_20), df_train$TARGET) # score: 0.6916447
calc_auc(as.numeric(pred_test_knn_20), df_test$TARGET) # score: 0.6022083

pred_train_knn_50 <- knn(train.X, train.X, train.Y, k=50, prob=TRUE)
pred_test_knn_50 <- knn(train.X, test.X, train.Y, k=50, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_50), df_train$TARGET) # score: 0.6807305
calc_auc(as.numeric(pred_test_knn_50), df_test$TARGET) # score: 0.6129506

pred_train_knn_100 <- knn(train.X, train.X, train.Y, k=100, prob=TRUE)
pred_test_knn_100 <- knn(train.X, test.X, train.Y, k=100, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_100), df_train$TARGET) # score: 0.6756031
calc_auc(as.numeric(pred_test_knn_100), df_test$TARGET) # score: 0.6217609

pred_train_knn_200 <- knn(train.X, train.X, train.Y, k=200, prob=TRUE)
pred_test_knn_200 <- knn(train.X, test.X, train.Y, k=200, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_200), df_train$TARGET) # score: 0.674016
calc_auc(as.numeric(pred_test_knn_200), df_test$TARGET) # score: 0.6304782

pred_train_knn_300 <- knn(train.X, train.X, train.Y, k=300, prob=TRUE)
pred_test_knn_300 <- knn(train.X, test.X, train.Y, k=300, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_300), df_train$TARGET) # score: 0.6721115
calc_auc(as.numeric(pred_test_knn_300), df_test$TARGET) # score: 0.6305978

pred_train_knn_400 <- knn(train.X, train.X, train.Y, k=400, prob=TRUE)
pred_test_knn_400 <- knn(train.X, test.X, train.Y, k=400, prob=TRUE)
calc_auc(as.numeric(pred_train_knn_400), df_train$TARGET) # score: 0.671257
calc_auc(as.numeric(pred_test_knn_400), df_test$TARGET) # score: 0.6341206

# knn with 500 ties end up with the error
# Error in knn(train.X, test.X, train.Y, k = 500, prob = TRUE) : 
# too many ties in knn

# ROC curve of the best model
plot_auc(as.numeric(pred_test_knn_400), df_test$TARGET)

# ===================================================================================================
# 6. Random Forest
# ===================================================================================================
# Test: all variables
set.seed(1)
model_rf <- randomForest(TARGET~., data=df_train)
pred_train_rf <- predict(model_rf, newdata=df_train)
pred_test_rf <- predict(model_rf, newdata=df_test)
importance(model_rf)

calc_auc(pred_train_rf, df_train$TARGET) # score: 1
calc_auc(pred_test_rf, df_test$TARGET) # score: 0.6528119
plot_auc(pred_test_rf, df_test$TARGET)

# Test: 4 variables
model_rf_s <- randomForest(TARGET~DAYS_BIRTH+EXT_SOURCE_1+EXT_SOURCE_2+EXT_SOURCE_3, data=df_train)
pred_train_rf_s <- predict(model_rf_s, newdata=df_train)
pred_test_rf_s <- predict(model_rf_s, newdata=df_test)
importance(model_rf_s)

calc_auc(pred_train_rf_s, df_train$TARGET) # score: 0.9871939
calc_auc(pred_test_rf_s, df_test$TARGET) # score: 0.6588263
plot_auc(pred_test_rf_s, df_test$TARGET)
