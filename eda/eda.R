### 1. Preparation ###
# Delete all variables
rm(list=ls())

# Import training and test data
df_train <- read.csv("./processed_data/train_raw.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE) # need na.strings to capture all missing values pattern
df_test <- read.csv("./processed_data/test_raw.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE) # need na.strings to capture all missing values pattern

### 2. Target Distribution ###
# Target is inbalanced
barplot(table(df_train$TARGET), main="Target Distribution", xlab="Target", ylab="Count")
table(df_train$TARGET) # 0: 230843, 1: 20478
table(df_train$TARGET)[2] / (table(df_train$TARGET)[1] + table(df_train$TARGET)[2]) * 100 # 1 accounts for approximately 8%

### 3. Variable Type ###
# Variable type
var_type <- sapply(df_train[, colnames(df_train) != "TARGET"], class) # exclude the target column
table(var_type) # character: 16, integer: 53, numeric: 52
col_char <- names(var_type[var_type == "character"])
col_int <- names(var_type[var_type == "integer"])
col_num <- names(var_type[var_type == "numeric"])

# Categorical variables: number of categories
cat <- c()
for (i in 1:length(col_char)){
  cat <- append(cat, dim(table(df_train[col_char[i]])))
}
num_categories <- data.frame(col_char, cat)
num_categories # number of categories

# Numerical variables
summary(df_train[, colnames(df_train) %in% col_int])
summary(df_train[, colnames(df_train) %in% col_num])



