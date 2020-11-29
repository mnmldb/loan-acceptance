### 1. Preparation ###
# Delete all variables
rm(list=ls())

# Set working directory 
setwd("~/Documents/UCLA_Extension/Final_Project")

# Import home credit data
df_all <- read.csv("../Raw_Data/application_train.csv", header=T, sep=",", na.strings=c('', 'NULL', '""')) # need na.strings to capture all missing values pattern
dim(df_all) # 307511 x 122

# Set column names not used for baseline model
del_columns <- c(
  "FONDKAPREMONT_MODE"
  , "HOUSETYPE_MODE" 
  , "WALLSMATERIAL_MODE"
  , "EMERGENCYSTATE_MODE"
  , "OWN_CAR_AGE"
  , "EXT_SOURCE_1"
  , "EXT_SOURCE_3"
  , "APARTMENTS_AVG"
  , "BASEMENTAREA_AVG"
  , "YEARS_BEGINEXPLUATATION_AVG"
  , "YEARS_BUILD_AVG"
  , "COMMONAREA_AVG"
  , "ELEVATORS_AVG"
  , "ENTRANCES_AVG"
  , "FLOORSMAX_AVG"
  , "FLOORSMIN_AVG"
  , "LANDAREA_AVG"
  , "LIVINGAPARTMENTS_AVG"
  , "LIVINGAREA_AVG"
  , "NONLIVINGAPARTMENTS_AVG"
  , "NONLIVINGAREA_AVG"
  , "APARTMENTS_MODE"
  , "BASEMENTAREA_MODE"
  , "YEARS_BEGINEXPLUATATION_MODE"
  , "YEARS_BUILD_MODE"
  , "COMMONAREA_MODE"
  , "ELEVATORS_MODE"
  , "ENTRANCES_MODE"
  , "FLOORSMAX_MODE"
  , "FLOORSMIN_MODE"
  , "LANDAREA_MODE"
  , "LIVINGAPARTMENTS_MODE"
  , "LIVINGAREA_MODE"
  , "NONLIVINGAPARTMENTS_MODE"
  , "NONLIVINGAREA_MODE"
  , "APARTMENTS_MEDI"
  , "BASEMENTAREA_MEDI"
  , "YEARS_BEGINEXPLUATATION_MEDI"
  , "YEARS_BUILD_MEDI"
  , "COMMONAREA_MEDI"
  , "ELEVATORS_MEDI"
  , "ENTRANCES_MEDI"
  , "FLOORSMAX_MEDI"
  , "FLOORSMIN_MEDI"
  , "LANDAREA_MEDI"
  , "LIVINGAPARTMENTS_MEDI"
  , "LIVINGAREA_MEDI"
  , "NONLIVINGAPARTMENTS_MEDI"
  , "NONLIVINGAREA_MEDI"
  , "TOTALAREA_MODE"
)

# Remove columns
df_all_removed <- df_all[, -which(colnames(df_all) %in% del_columns)]
dim(df_all_removed) # 307511 x 72

### 2. Train Test Split ###
# Set column names used for stratified sampling
stratified_columns <- c(
  "NAME_CONTRACT_TYPE"
  , "CODE_GENDER"
  , "FLAG_OWN_CAR"
  , "FLAG_OWN_REALTY"
  , "NAME_TYPE_SUITE"
  , "NAME_INCOME_TYPE"
  , "NAME_EDUCATION_TYPE"
  , "NAME_FAMILY_STATUS"
  , "NAME_HOUSING_TYPE"
  , "WEEKDAY_APPR_PROCESS_START"
  , "ORGANIZATION_TYPE"
  , "FLAG_MOBIL"
  , "FLAG_EMP_PHONE"
  , "FLAG_WORK_PHONE"
  , "FLAG_CONT_MOBILE"
  , "FLAG_PHONE"
  , "FLAG_EMAIL"
  , "REGION_RATING_CLIENT"
  , "REGION_RATING_CLIENT_W_CITY"
  , "REG_REGION_NOT_LIVE_REGION"
  , "REG_REGION_NOT_WORK_REGION"
  , "LIVE_REGION_NOT_WORK_REGION"
  , "REG_CITY_NOT_LIVE_CITY"
  , "REG_CITY_NOT_WORK_CITY"
  , "LIVE_CITY_NOT_WORK_CITY"
)

# Seprate data
# reference: https://www.rdocumentation.org/packages/fifer/versions/1.0/topics/stratified
# install.packages("splitstackshape")
library(splitstackshape)
set.seed(1)
test_id <- stratified(df_all_removed, stratified_columns, size = .3)$SK_ID_CURR
df_train <- df_all_removed[!(df_all_removed$SK_ID_CURR %in% test_id),]
df_test <- df_all_removed[df_all_removed$SK_ID_CURR %in% test_id,]
dim(df_train) # 251321 x 72
dim(df_test) # 56190 x 72

# Sanity check
table(df_train$TARGET)
table(df_test$TARGET) 

test_id[1] # 431068
df_train[df_train$SK_ID_CURR == test_id[1],] # 0 rows
df_test[df_test$SK_ID_CURR == test_id[1],] # 1 rows

### 3. Missing values ###
# Fill by a certain category (one by one)
sum(is.na(df_train$OCCUPATION_TYPE)) # 74456
sum(is.na(df_test$OCCUPATION_TYPE)) # 21935
df_train$OCCUPATION_TYPE[is.na(df_train$OCCUPATION_TYPE)] <- "Unknown"
df_test$OCCUPATION_TYPE[is.na(df_test$OCCUPATION_TYPE)] <- "Unknown"
sum(is.na(df_train$OCCUPATION_TYPE)) # 0
sum(is.na(df_test$OCCUPATION_TYPE)) # 0

df_train$NAME_TYPE_SUITE[is.na(df_train$NAME_TYPE_SUITE)] <- "Unknown"
df_test$NAME_TYPE_SUITE[is.na(df_test$NAME_TYPE_SUITE)] <- "Unknown"

# Fill by mean of training data (one by one)
mean_AMT_ANNUITY <- mean(df_train$AMT_ANNUITY[!(is.na(df_train$AMT_ANNUITY))])
mean_AMT_GOODS_PRICE <- mean(df_train$AMT_GOODS_PRICE[!(is.na(df_train$AMT_GOODS_PRICE))])
mean_EXT_SOURCE_2 <- mean(df_train$EXT_SOURCE_2[!(is.na(df_train$EXT_SOURCE_2))])

df_train$AMT_ANNUITY[is.na(df_train$AMT_ANNUITY)] <- mean_AMT_ANNUITY
df_train$AMT_GOODS_PRICE[is.na(df_train$AMT_GOODS_PRICE)] <- mean_AMT_GOODS_PRICE
df_train$EXT_SOURCE_2[is.na(df_train$EXT_SOURCE_2)] <- mean_EXT_SOURCE_2

df_test$AMT_ANNUITY[is.na(df_test$AMT_ANNUITY)] <- mean_AMT_ANNUITY
df_test$AMT_GOODS_PRICE[is.na(df_test$AMT_GOODS_PRICE)] <- mean_AMT_GOODS_PRICE
df_test$EXT_SOURCE_2[is.na(df_test$EXT_SOURCE_2)] <- mean_EXT_SOURCE_2

# Fill by median of training data (apply and loop)
median_columns <- c(
  "CNT_FAM_MEMBERS"
  , "OBS_30_CNT_SOCIAL_CIRCLE"
  , "DEF_30_CNT_SOCIAL_CIRCLE"
  , "OBS_60_CNT_SOCIAL_CIRCLE"
  , "DEF_60_CNT_SOCIAL_CIRCLE"
  , "DAYS_LAST_PHONE_CHANGE"
  , "AMT_REQ_CREDIT_BUREAU_HOUR"
  , "AMT_REQ_CREDIT_BUREAU_DAY"
  , "AMT_REQ_CREDIT_BUREAU_WEEK"
  , "AMT_REQ_CREDIT_BUREAU_MON"
  , "AMT_REQ_CREDIT_BUREAU_QRT"
  , "AMT_REQ_CREDIT_BUREAU_YEAR"
)

median_values <- apply(df_train[, median_columns], 2, median, na.rm=TRUE)

for (i in 1:length(median_columns)) {
  na_index_train <- is.na(df_train[, median_columns[i]])
  na_index_test <- is.na(df_test[, median_columns[i]])
  df_train[na_index_train, median_columns[i]] <- median_values[i]
  df_test[na_index_test, median_columns[i]] <- median_values[i]
}

# Check if all missing values are filled
sum(is.na(df_train)) # 0
sum(is.na(df_test)) # 0
sapply(df_train, function(y) sum(is.na(y))) 
sapply(df_test, function(y) sum(is.na(y))) 

### 4. Standardization ###
# Set column names to standardize
standard_columns <- c(
  "CNT_CHILDREN"
  , "DAYS_BIRTH"
  , "DAYS_EMPLOYED"
  , "DAYS_ID_PUBLISH"
  , "HOUR_APPR_PROCESS_START"
  , "AMT_INCOME_TOTAL"
  , "AMT_CREDIT"
  , "AMT_ANNUITY"
  , "AMT_GOODS_PRICE"
  , "REGION_POPULATION_RELATIVE"
  , "DAYS_REGISTRATION"
  , "CNT_FAM_MEMBERS"
  , "OBS_30_CNT_SOCIAL_CIRCLE"
  , "DEF_30_CNT_SOCIAL_CIRCLE"
  , "OBS_60_CNT_SOCIAL_CIRCLE"
  , "DEF_60_CNT_SOCIAL_CIRCLE"
  , "DAYS_LAST_PHONE_CHANGE"
  , "AMT_REQ_CREDIT_BUREAU_HOUR"
  , "AMT_REQ_CREDIT_BUREAU_DAY"
  , "AMT_REQ_CREDIT_BUREAU_WEEK"
  , "AMT_REQ_CREDIT_BUREAU_MON"
  , "AMT_REQ_CREDIT_BUREAU_QRT"
  , "AMT_REQ_CREDIT_BUREAU_YEAR"
)

standard_mean <- apply(df_train[, standard_columns], 2, mean)
standard_std <- apply(df_train[, standard_columns], 2, sd)

for (i in 1:length(standard_columns)) {
  df_train[, standard_columns[i]] <- sapply(df_train[, standard_columns[i]], function(x, mean=standard_mean[i], std=standard_std[i]) {
    return((x - mean) / std)})
  df_test[, standard_columns[i]] <- sapply(df_test[, standard_columns[i]], function(x, mean=standard_mean[i], std=standard_std[i]) {
    return((x - mean) / std)})
}

# Sanity check
mean(df_train[, "CNT_CHILDREN"]) # 0
sd(df_train[, "CNT_CHILDREN"]) # 1

### 5. One Hot Encoding ###
# <reference> https://www.rdocumentation.org/packages/makedummies/versions/1.2.1
# install.packages("makedummies")
library(makedummies)

# Set columns names for one hot encoding
encoding_columns <- c(
  "NAME_CONTRACT_TYPE"
  , "CODE_GENDER"
  , "FLAG_OWN_CAR"
  , "FLAG_OWN_REALTY"
  , "NAME_TYPE_SUITE"
  , "NAME_INCOME_TYPE"
  , "NAME_EDUCATION_TYPE"
  , "NAME_FAMILY_STATUS"
  , "NAME_HOUSING_TYPE"
  , "OCCUPATION_TYPE"
  , "WEEKDAY_APPR_PROCESS_START"
  , "ORGANIZATION_TYPE"
)

# Combine df_train and df_test for one hot encoding
num_train <- dim(df_train)[1]
num_test <- dim(df_test)[1]

df_tmp <- rbind(df_train, df_test)

for (i in 1:length(encoding_columns)) {
  dat <- data.frame(x = factor(df_tmp[, encoding_columns[i]]))
  dummies <- makedummies(dat, basal_level = TRUE)
  df_tmp <- cbind(df_tmp, dummies)
}
df_tmp <- df_tmp[, -which(colnames(df_tmp) %in% encoding_columns)]

df_train_fnl <- df_tmp[1:num_train,]
df_test_fnl <- df_tmp[num_train+1:num_test,]

# Remove unnecessary columns
df_train_fnl <- df_train_fnl[, -which(colnames(df_train_fnl) %in% c("SK_ID_CURR"))]
df_test_target <- df_test_fnl$TARGET
df_test_fnl <- df_test_fnl[, -which(colnames(df_test_fnl) %in% c("SK_ID_CURR", "TARGET"))]

### 7. Create models ###
attach(df_train_fnl)
# logistic regression
model_lr <- glm(TARGET~., data=df_train_fnl, family=binomial)
summary(model_lr)
pred_lr <- predict(model_lr, df_test_fnl, type="response")

### 8. Scores ###
# <reference> https://www.rdocumentation.org/packages/ROCR/versions/1.0-11
# install.packages(("ROCR"))
library(ROCR)
prep_lr <- prediction(pred_lr, df_test_target)
perf_lr <- performance(prep_lr, "tpr", "fpr")
plot(perf_lr)
auc_lr <- performance(prep_lr, "auc")
auc_lr <- as.numeric(auc_lr@y.values)
print(auc_lr) # score: 51.6
