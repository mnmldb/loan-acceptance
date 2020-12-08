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
# 1. Area under the ROC Curve
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
model_lr <- glm(TARGET~., data=df_train, family=binomial)
summary(model_lr)
pred_train_lr <- predict(model_lr, type="response")
pred_test_lr <- predict(model_lr, newdata=df_test, type="response")

plot_auc(pred_test_lr, df_test$TARGET)
calc_auc(pred_test_lr, df_test$TARGET) # score: 0.6777203

model_lr_s <- glm(TARGET~DAYS_BIRTH+EXT_SOURCE_1+EXT_SOURCE_2+EXT_SOURCE_3, data=df_train, family=binomial)
summary(model_lr_s)
pred_train_lr_s <- predict(model_lr_s, type="response")
pred_test_lr_s <- predict(model_lr_s, newdata=df_test, type="response")

plot_auc(pred_test_lr_s, df_test$TARGET)
calc_auc(pred_test_lr_s, df_test$TARGET) # score: 0.7107454

# ===================================================================================================
# 4. Linear Discriminant Analysis
# ===================================================================================================
model_lda <- lda(TARGET~., data=df_train)
model_lda
pred_test_lda <- predict(model_lda, newdata=df_test)

# <reference> https://stackoverflow.com/questions/41533811/roc-curve-in-linear-discriminant-analysis-with-r
plot_auc(pred_test_lda$posterior[,2], df_test$TARGET)
calc_auc(pred_test_lda$posterior[,2], df_test$TARGET) # score: 0.6780641

model_lda_s <- lda(TARGET~DAYS_BIRTH+EXT_SOURCE_1+EXT_SOURCE_2+EXT_SOURCE_3, data=df_train)
model_lda_s
pred_test_lda_s <- predict(model_lda_s, newdata=df_test)

# <reference> https://stackoverflow.com/questions/41533811/roc-curve-in-linear-discriminant-analysis-with-r
plot_auc(pred_test_lda_s$posterior[,2], df_test$TARGET)
calc_auc(pred_test_lda_s$posterior[,2], df_test$TARGET) # score: 0.7102984

# ===================================================================================================
# 5. K-Nearest Neighbors
# ===================================================================================================
train.X <- df_train[, -which(colnames(df_train) == "TARGET")]
test.X <- df_test[, -which(colnames(df_test) == "TARGET")]
train.Y <- df_train$TARGET

pred_test_knn <- knn(train.X, test.X, train.Y, k=10, prob=TRUE)

# https://stackoverflow.com/questions/40783331/rocr-error-format-of-predictions-is-invalid
plot_auc(as.numeric(pred_test_knn), df_test$TARGET)
calc_auc(as.numeric(pred_test_knn), df_test$TARGET) # score: 0.5924022

# ===================================================================================================
# 6. Random Forest
# ===================================================================================================
set.seed(1)

model_rf <- randomForest(TARGET~., data=df_train)
pred_test_rf <- predict(model_rf, newdata=df_test)
importance(model_rf)

plot_auc(pred_test_rf, df_test$TARGET)
calc_auc(pred_test_rf, df_test$TARGET) # score: 0.6522237

model_rf_all <- randomForest(TARGET~., data=df_train_all)
pred_test_rf_all <- predict(model_rf_all, newdata=df_test_all)
importance(model_rf_all)

plot_auc(pred_test_rf_all, df_test$TARGET)
calc_auc(pred_test_rf_all, df_test$TARGET) # score: 0.6095454

model_rf_s <- randomForest(TARGET~DAYS_BIRTH+EXT_SOURCE_1+EXT_SOURCE_2+EXT_SOURCE_3, data=df_train)
pred_test_rf_s <- predict(model_rf_s, newdata=df_test)
importance(model_rf_s)

plot_auc(pred_test_rf_s, df_test$TARGET)
calc_auc(pred_test_rf_s, df_test$TARGET) # score: 0.6588263
