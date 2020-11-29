# Delete all variables
rm(list=ls())

# Import raw data
df_all <- read.csv("./raw_data/application_train.csv", header=T, sep=",", na.strings=c('', 'NULL', '""')) # need na.strings to capture all missing values pattern
dim(df_all) # 307511 x 122

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

# Separate data
# reference: https://www.rdocumentation.org/packages/fifer/versions/1.0/topics/stratified
# install.packages("splitstackshape")
library(splitstackshape)
set.seed(1)
test_id <- stratified(df_all, stratified_columns, size = .3)$SK_ID_CURR
df_train <- df_all[!(df_all$SK_ID_CURR %in% test_id),]
df_test <- df_all[df_all$SK_ID_CURR %in% test_id,]
dim(df_train) # 251321 x 122
dim(df_test) # 56190 x 122

# Sanity check
table(df_train$TARGET)
table(df_test$TARGET) 

test_id[1] # 100002
df_train[df_train$SK_ID_CURR == test_id[1],] # 0 rows
df_test[df_test$SK_ID_CURR == test_id[1],] # 1 row

# Export
write.csv(df_train, "./Processed_Data/train_raw.csv", row.names=FALSE, na="")
write.csv(df_test, "./Processed_Data/test_raw.csv", row.names=FALSE, na="")