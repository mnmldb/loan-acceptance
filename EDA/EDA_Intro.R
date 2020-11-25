### preparation ###
# delete all variables
rm(list=ls())

# set working directory 
setwd("~/Documents/UCLA_Extension/Final_Project")

# import training data
app_train <- read.csv("./Data/application_train.csv", header=T, sep=",", na.strings=c('', 'NULL', '""')) # need na.strings to capture all missing values pattern
dim(app_train) # 307511 x 122
head(app_train)
names(app_train) # column names
summary(app_train)

### EDA ###
# distribution of the target
table(app_train$TARGET) # 0: 282686, 1: 24825
hist(app_train$TARGET) # imbalanced class problem

# missing values
col_train <- app_train[, colnames(app_train) != "TARGET"] # dataframe without target column
sum_na_train <- sapply(col_train, function(y) sum(is.na(y))) # vector with names (index)
length(sum_na_train[sum_na_train > 0]) # number of columns that have missing values: 67
sort(sum_na_train[sum_na_train > 0], decreasing=TRUE)[1:20] # top 20 columns with missing values
nas <- sort(sum_na_train[sum_na_train > 0], decreasing=TRUE)
write.table(nas, "missing_values.csv")

# column types
col_type <- sapply(col_train, class)
table(col_type) # character: 16, integer: 40, numeric: 65
write.table(col_type, "column_types.csv")

# number of classes in the character columns
category_columns <- names(col_type[col_type == "character"]) # names of 16 columns
col_train[category_columns]
cat <- c()
for (i in 1:length(category_columns)){
  cat <- append(cat, dim(table(col_train[category_columns[i]])))
}
num_class <- data.frame(category_columns, cat)
num_class # number of classes
write.table(num_class, "num_class.csv")

table(col_train$NAME_CONTRACT_TYPE)
table(col_train$CODE_GENDER)
table(col_train$FLAG_OWN_CAR)
table(col_train$FLAG_OWN_REALTY)
table(col_train$NAME_TYPE_SUITE)
table(col_train$NAME_INCOME_TYPE)
table(col_train$NAME_EDUCATION_TYPE)
table(col_train$NAME_FAMILY_STATUS)
table(col_train$NAME_HOUSING_TYPE)
write.table(table(col_train$OCCUPATION_TYPE), "occupations.csv")
table(col_train$WEEKDAY_APPR_PROCESS_START)
write.table(table(col_train$ORGANIZATION_TYPE), "organizations.csv")
table(col_train$FONDKAPREMONT_MODE)
table(col_train$HOUSETYPE_MODE)
table(col_train$WALLSMATERIAL_MODE)
table(col_train$EMERGENCYSTATE_MODE)

table(col_train$FLAG_EMP_PHONE)
table(col_train$REGION_RATING_CLIENT)
table(col_train$FLAG_DOCUMENT_2)
table(col_train$FLAG_DOCUMENT_10)

# anomalies, outliers
# TBD

# corelation
# TBD
