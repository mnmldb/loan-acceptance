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

# Import training and test data
df_train_all <- read.csv("./processed_data/train_model_ds.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE)
df_test_all <- read.csv("./processed_data/test_model.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE)

# Columns used for modeling
col_use <- c(
    "TARGET"
  , "DAYS_BIRTH"
  , "DAYS_EMPLOYED"
  , "DAYS_ID_PUBLISH"
  , "AMT_INCOME_TOTAL"
  , "AMT_CREDIT"
  , "AMT_ANNUITY"
  , "AMT_GOODS_PRICE"
  , "DAYS_REGISTRATION"
  , "EXT_SOURCE_1"
  , "EXT_SOURCE_2"
  , "EXT_SOURCE_3"
  , "ANOMALY_DAYS_EMPLOYED"
  , "INTER_EXT_SOURCE_1_2"
  , "INTER_EXT_SOURCE_2_3"
  , "INTER_EXT_SOURCE_3_1"
  , "CODE_GENDER_F"
  , "CODE_GENDER_M"
)

df_train <- df_train_all[, colnames(df_train_all) %in% col_use]
df_test <- df_test_all[, colnames(df_test_all) %in% col_use]

# ===================================================================================================
# 2. Logistic Regression
# ===================================================================================================
model_lr <- glm(TARGET~., data=df_train, family=binomial)
summary(model_lr)
pred_train_lr <- predict(model_lr, type="response")
pred_test_lr <- predict(model_lr, newdata=df_test, type="response")

prep_lr <- prediction(pred_test_lr, df_test$TARGET)
perf_lr <- performance(prep_lr, "tpr", "fpr")
plot(perf_lr)
auc_lr <- performance(prep_lr, "auc")
auc_lr <- as.numeric(auc_lr@y.values)
print(auc_lr) # score: 0.6777203

# ===================================================================================================
# 3. Linear Discriminant Analysis
# ===================================================================================================
# All features
model_lda <- lda(TARGET~., data=df_train)
model_lda
pred_test_lda <- predict(model_lda, newdata=df_test)

# <reference> https://stackoverflow.com/questions/41533811/roc-curve-in-linear-discriminant-analysis-with-r
prep_lda <- prediction(pred_test_lda$posterior[,2], df_test$TARGET)
perf_lda <- performance(prep_lda, "tpr", "fpr")
plot(perf_lda)
auc_lda <- performance(prep_lda, "auc")
auc_lda <- as.numeric(auc_lda@y.values)
print(auc_lda) # score: 0.6780641

# ===================================================================================================
# 4. K-Nearest Neighbors
# ===================================================================================================
train.X <- df_train[, -which(colnames(df_train) == "TARGET")]
test.X <- df_test[, -which(colnames(df_test) == "TARGET")]
train.Y <- df_train$TARGET

pred_test_knn <- knn(train.X, test.X, train.Y, k=10, prob=TRUE)

prep_knn <- prediction(as.numeric(pred_test_knn), df_test$TARGET)
# https://stackoverflow.com/questions/40783331/rocr-error-format-of-predictions-is-invalid
perf_knn <- performance(prep_knn, "tpr", "fpr")
plot(perf_knn)
auc_knn <- performance(prep_knn, "auc")
auc_knn <- as.numeric(auc_knn@y.values)
print(auc_knn) # score: 0.5924022

# ===================================================================================================
# 5. Random Forest
# ===================================================================================================
set.seed(1)

model_rf <- randomForest(TARGET~., data=df_train)
pred_test_rf <- predict(model_rf, newdata=df_test)
importance(model_rf)

prep_rf <- prediction(pred_test_rf, df_test$TARGET)
perf_rf <- performance(prep_rf, "tpr", "fpr")
plot(perf_rf)
auc_rf <- performance(prep_rf, "auc")
auc_rf <- as.numeric(auc_rf@y.values)
print(auc_rf) # score: 0.6522237

model_rf_all <- randomForest(TARGET~., data=df_train_all)
pred_test_rf_all <- predict(model_rf_all, newdata=df_test_all)
importance(model_rf_all)

prep_rf_all <- prediction(pred_test_rf_all, df_test_all$TARGET)
perf_rf_all <- performance(prep_rf_all, "tpr", "fpr")
plot(perf_rf_all)
auc_rf_all <- performance(prep_rf_all, "auc")
auc_rf_all <- as.numeric(auc_rf_all@y.values)
print(auc_rf_all) # score: 0.6095454
