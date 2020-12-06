# ===================================================================================================
# 1. Preparation
# ===================================================================================================

# Delete all variables
rm(list=ls())

# Import packages
library(dplyr)
library(ggplot2)
library(ggsci)

# Import training and test data
df_train <- read.csv("./processed_data/train_raw.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE) # need na.strings to capture all missing values pattern
df_test <- read.csv("./processed_data/test_raw.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE) # need na.strings to capture all missing values pattern

n_train <- dim(df_train)[1] # 251321
n_test <- dim(df_test)[1] # 56190

# ===================================================================================================
# 2. Target Distribution
# ===================================================================================================
# Target is inbalanced
barplot(table(df_train$TARGET), main="Target Distribution", xlab="Target", ylab="Count")
table(df_train$TARGET) # 0: 230843, 1: 20478
table(df_train$TARGET)[2] / (table(df_train$TARGET)[1] + table(df_train$TARGET)[2]) * 100 # 1 accounts for approximately 8%

# ===================================================================================================
# 3. Variable Type
# ===================================================================================================
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
df_train_cat_unique <- data.frame(col_cat, cat)
colnames(df_train_cat_unique) <- c("Variable", "Categories")
df_train_cat_unique # number of categories

# ===================================================================================================
# 4. Missing Values
# ===================================================================================================
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

# ===================================================================================================
# 5. Variable Selection
# ===================================================================================================
# Converting summary to data frame
# https://stackoverflow.com/questions/30520350/convert-summary-to-data-frame
# as.data.frame(apply(df_train_int, 2, summary))

df_train_cat_summary <- inner_join(df_train_cat_unique, df_train_cat_na, by="Variable")
df_train_semicat_summary <- inner_join(df_train_semicat_unique, df_train_semicat_na, by="Variable")
df_train_int_summary <- inner_join(df_train_int_unique, df_train_int_na, by="Variable")
df_train_num_summary <- inner_join(df_train_num_unique, df_train_num_na, by="Variable")

# Majority of semi-categorical variables consist of "Flag" or "Rating"
# Exclude them for the time being

# Missing values cut-off: columns with more than 10% missing values are excluded (20% might be the best practice)
# Category numbers cut-off: columns with categories more than 6 are excluded
col_cat_use <- df_train_cat_summary %>%
  dplyr::filter(Missing_Values_Percent < .1) %>%
  dplyr::filter(Categories < 7) %>%
  dplyr::select(Variable) # 7

col_int_use <- df_train_int_summary %>%
  dplyr::filter(Missing_Values_Percent < .1) %>%
  dplyr::select(Variable) # 11

col_num_use <- df_train_num_summary %>%
  dplyr::filter(Missing_Values_Percent < .1) %>%
  dplyr::select(Variable) # 7

# Create the new data frame with necessary columns
df_train_processed <- df_train %>%
  dplyr::select(as.vector(unlist(rbind(col_cat_use, col_int_use, col_num_use)))) 

## Fill missing values
# As the number of missing values are small now, we fill them by median
col_cat_use_na <- df_train_cat_summary %>%
  dplyr::semi_join(col_cat_use, by="Variable") %>%
  dplyr::filter(Missing_Values != 0) %>%
  dplyr::select(Variable) # 0

col_int_use_na <- df_train_int_summary %>%
  dplyr::semi_join(col_int_use, by="Variable") %>%
  dplyr::filter(Missing_Values != 0) %>%
  dplyr::select(Variable) # 6

col_num_use_na <- df_train_num_summary %>%
  dplyr::semi_join(col_num_use, by="Variable") %>%
  dplyr::filter(Missing_Values != 0) %>%
  dplyr::select(Variable) # 3

# <reference> converting a data frame to a vector: https://stackoverflow.com/questions/2545228/convert-a-dataframe-to-a-vector-by-rows
# Calculate median by columns
na_median <- df_train %>%
  dplyr::select(as.vector(unlist(rbind(col_int_use_na, col_num_use_na)))) %>%
  sapply(median, na.rm=TRUE)

# Loop to fill missing values
for (i in 1:length(na_median)) {
  na_column <- names(na_median)[i]
  na_index_train <- is.na(df_train_processed[, na_column])
  df_train_processed[na_index_train, na_column] <- na_median[i]
} 

# Sanity check
df_train_processed %>%
  is.na %>%
  sum # 0

## One hot encoding
# <reference> https://www.rdocumentation.org/packages/makedummies/versions/1.2.1
# install.packages("makedummies")
library(makedummies)

for (i in 1:length(unlist(col_cat_use))) {
  col <- unlist(col_cat_use)[i]
  dat <- data.frame(factor(df_train_processed[, col]))
  colnames(dat) <- col
  dummies <- makedummies(dat, basal_level = TRUE)
  df_train_processed <- cbind(df_train_processed, dummies)
}

# Remove categorical variables and add target
df_train_processed <- df_train_processed[, -which(colnames(df_train_processed) %in% unlist(col_cat_use))]
df_train_processed <- cbind(df_train$TARGET, df_train_processed)
colnames(df_train_processed)[1] <- "TARGET"

# Calculate correlation with target
cor_train <- c()
for (i in 1:length(colnames(df_train_processed))) {
   cor_tmp <- df_train_processed[, colnames(df_train_processed)[i]] %>%
    cor(df_train_processed$TARGET)
   cor_train <- c(cor_train, cor_tmp)
}

df_cor_train <- data.frame(colnames(df_train_processed), cor_train)
colnames(df_cor_train) <- c("Variable", "Correlation")
df_cor_train <- df_cor_train %>%
  dplyr::filter(Variable != "TARGET") %>%
  dplyr::arrange(desc(Correlation))

# Visualize the correlation
ggplot(df_cor_train, aes(x = reorder(Variable, Correlation), y = Correlation)) +
  geom_bar(stat = "identity") +
  labs(x="Variables", y="Correlation", title="Correlation with Target")  +
  theme(axis.text=element_text(size=7), axis.title=element_text(size=9), plot.title=element_text(size=12, face="bold")) +
  coord_flip() 


  




