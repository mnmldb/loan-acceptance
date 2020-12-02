### 1. Preparation ###
# Delete all variables
rm(list=ls())

# Import packages
library(dplyr)

# Import training and test data
df_train <- read.csv("./processed_data/train_raw.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE) # need na.strings to capture all missing values pattern
df_test <- read.csv("./processed_data/test_raw.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE) # need na.strings to capture all missing values pattern

n_train <- dim(df_train)[1] # 251321
n_test <- dim(df_test)[1] # 56190

### 2. Target Distribution ###
# Target is inbalanced
barplot(table(df_train$TARGET), main="Target Distribution", xlab="Target", ylab="Count")
table(df_train$TARGET) # 0: 230843, 1: 20478
table(df_train$TARGET)[2] / (table(df_train$TARGET)[1] + table(df_train$TARGET)[2]) * 100 # 1 accounts for approximately 8%

### 3. Variable Type ###
# Variable type
var_type <- sapply(df_train[, -which(colnames(df_train) %in% c("TARGET", "SK_ID_CURR"))], class) # exclude the customer id and target column
table(var_type) # character: 16, integer: 52, numeric: 52
col_cat <- names(var_type[var_type == "character"])
col_int <- names(var_type[var_type == "integer"])
col_num <- names(var_type[var_type == "numeric"])

df_train_cat <- df_train[, colnames(df_train) %in% col_cat]
df_train_int <- df_train[, colnames(df_train) %in% col_int]
df_train_num <- df_train[, colnames(df_train) %in% col_num]

# Split semi-categorical variables in integer and numeric types by counting unique
train_int_unique <- sapply(df_train_int, function(y) length(unique(y)))
train_num_unique <- sapply(df_train_num, function(y) length(unique(y)))
df_train_int_unique <- data.frame(names(train_int_unique), train_int_unique)
df_train_num_unique <- data.frame(names(train_num_unique), train_num_unique)
colnames(df_train_int_unique) <- c("Variable", "Unique")
colnames(df_train_num_unique) <- c("Variable", "Unique")
rownames(df_train_int_unique) <- NULL
rownames(df_train_num_unique) <- NULL

# Variables that has only a few unique numbers are suspicious; used 5 in this case
df_train_int_unique[df_train_int_unique$Unique <= 5, ] # 34 variables
df_train_num_unique[df_train_num_unique$Unique <= 5, ] # 0 variables

df_train_semicat_unique <- df_train_int_unique[df_train_int_unique$Unique <= 5, ]
df_train_int_unique <- df_train_int_unique[df_train_int_unique$Unique > 5, ] # delete semi-categorical variables

df_train_semicat <- df_train[, colnames(df_train) %in% df_train_semicat_unique$Variable]
df_train_int <- df_train[, colnames(df_train) %in% df_train_int_unique$Variable]
         
# Categorical variables: number of categories
cat <- c()
for (i in 1:length(col_cat)){ # use loop instead of sapply
  cat <- append(cat, dim(table(df_train_cat[col_cat[i]])))
}
df_train_cat_unique <- data.frame(col_char, cat)
colnames(df_train_cat_unique) <- c("Variable", "Categories")
df_train_cat_unique # number of categories

### 4. Missing Values ###
# Count missing values
na_cat <- sapply(df_train_cat, function(y) sum(is.na(y)))
na_semicat <- sapply(df_train_semicat, function(y) sum(is.na(y)))
na_int <- sapply(df_train_int, function(y) sum(is.na(y)))
na_num <- sapply(df_train_num, function(y) sum(is.na(y)))

# Calculate percentage
df_train_cat_na <- data.frame(na_cat, na_cat / n_train)
df_train_semicat_na <- data.frame(na_semicat, na_semicat / n_train)
df_train_int_na <- data.frame(na_int, na_int / n_train)
df_train_num_na <- data.frame(na_num, na_num / n_train)

# Create data frames
df_train_cat_na <- data.frame(rownames(df_train_cat_na), df_train_cat_na)
df_train_semicat_na <- data.frame(rownames(df_train_semicat_na), df_train_semicat_na)
df_train_int_na <- data.frame(rownames(df_train_int_na), df_train_int_na)
df_train_num_na <- data.frame(rownames(df_train_num_na), df_train_num_na)
colnames(df_train_cat_na) <- c("Variable", "Missing_Values", "Missing_Values_Percent")
colnames(df_train_semicat_na) <- c("Variable", "Missing_Values", "Missing_Values_Percent")
colnames(df_train_int_na) <- c("Variable", "Missing_Values", "Missing_Values_Percent")
colnames(df_train_num_na) <- c("Variable", "Missing_Values", "Missing_Values_Percent")
rownames(df_train_cat_na) <- NULL
rownames(df_train_semicat_na) <- NULL
rownames(df_train_int_na) <- NULL
rownames(df_train_num_na) <- NULL

# Result: number of missing values and percentage
df_train_cat_na
df_train_semicat_na
df_train_int_na
df_train_num_na

### 5. Variable Selection ###
# Converting summary to data frame
# https://stackoverflow.com/questions/30520350/convert-summary-to-data-frame
# as.data.frame(apply(df_train_int, 2, summary))

df_train_cat_summary <- inner_join(df_train_cat_unique, df_train_cat_na, by="Variable")
df_train_semicat_summary <- inner_join(df_train_semicat_unique, df_train_semicat_na, by="Variable")
df_train_int_summary <- inner_join(df_train_int_unique, df_train_int_na, by="Variable")
df_train_num_summary <- inner_join(df_train_num_unique, df_train_num_na, by="Variable")

