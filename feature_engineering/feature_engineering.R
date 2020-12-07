# ===================================================================================================
# 1. Preparation
# ===================================================================================================
# Import packages
library(dplyr)
library(ggplot2)
library(ggsci)

# Import training and test data
df_train_raw <- read.csv("./processed_data/train_raw.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE)
df_test_raw <- read.csv("./processed_data/test_raw.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE)

# Selected features from eda.R
col_cat_use <- c(
    "NAME_CONTRACT_TYPE"
  , "CODE_GENDER"
  , "FLAG_OWN_CAR"
  , "FLAG_OWN_REALTY"
  , "NAME_EDUCATION_TYPE"
  , "NAME_FAMILY_STATUS"
  , "NAME_HOUSING_TYPE"
)

col_int_use <- c(
    "CNT_CHILDREN"
  , "DAYS_BIRTH"
  , "DAYS_EMPLOYED"
  , "DAYS_ID_PUBLISH"
  , "CNT_FAM_MEMBERS"
  , "HOUR_APPR_PROCESS_START"
  , "OBS_30_CNT_SOCIAL_CIRCLE"
  , "DEF_30_CNT_SOCIAL_CIRCLE"
  , "OBS_60_CNT_SOCIAL_CIRCLE"
  , "DEF_60_CNT_SOCIAL_CIRCLE"
  , "DAYS_LAST_PHONE_CHANGE"
)

col_num_use <- c(
    "AMT_INCOME_TOTAL"
  , "AMT_CREDIT"
  , "AMT_ANNUITY"
  , "AMT_GOODS_PRICE"
  , "REGION_POPULATION_RELATIVE"
  , "DAYS_REGISTRATION"
  , "EXT_SOURCE_1" # added
  , "EXT_SOURCE_2"
  , "EXT_SOURCE_3" # added
)

col_int_use_na <- c(
   "CNT_FAM_MEMBERS"
  , "OBS_30_CNT_SOCIAL_CIRCLE"
  , "DEF_30_CNT_SOCIAL_CIRCLE"
  , "OBS_60_CNT_SOCIAL_CIRCLE"
  , "DEF_60_CNT_SOCIAL_CIRCLE"
  , "DAYS_LAST_PHONE_CHANGE"
)

col_num_use_na <- c(
    "AMT_ANNUITY"
  , "AMT_GOODS_PRICE"
  , "EXT_SOURCE_1" # added
  , "EXT_SOURCE_2"
  , "EXT_SOURCE_3" # added
)

# Create new data frames with the necessary features
df_train <- df_train_raw %>%
  dplyr::select(c(col_cat_use, col_int_use, col_num_use))

df_test <- df_test_raw %>%
  dplyr::select(c(col_cat_use, col_int_use, col_num_use))

# ===================================================================================================
# 2. Converting Negative Values
# ===================================================================================================
# For interpretability, we multiple -1 to the negative values 
# Features related to days are recorded relative to the current loan application
df_train$DAYS_BIRTH <- df_train$DAYS_BIRTH * (-1)
df_train$DAYS_EMPLOYED <- df_train$DAYS_EMPLOYED * (-1)
df_train$DAYS_ID_PUBLISH <- df_train$DAYS_ID_PUBLISH * (-1)
df_train$DAYS_LAST_PHONE_CHANGE <- df_train$DAYS_LAST_PHONE_CHANGE * (-1)
df_train$DAYS_REGISTRATION <- df_train$DAYS_REGISTRATION * (-1)

# ===================================================================================================
# 3. Missing Values
# ===================================================================================================
# Create the data frame to contain the filling value
df_fill_value <- data.frame(Feature=c(col_int_use_na, col_num_use_na),
                            Type=c(rep("integer", length(col_int_use_na)), rep("numeric", length(col_num_use_na))),
                            Method=c(rep(NA, length(col_int_use_na) + length(col_num_use_na))),
                            Value=c(rep(NA, length(col_int_use_na) + length(col_num_use_na)))
                              )

# Create the function to return the mode
calc_mode <- function(a) {
  return(names(which.max(table(a))))
}

# Create the function to plot the histogram
plot_hist <- function(f) {
  df_train %>%
    # ggplot(aes(x=df_train[, f], y = ..density..)) +
    ggplot(aes(x=df_train[, f])) +
    geom_histogram(position = "identity", alpha = 0.5) +
    # geom_density(aes(alpha = 0.2, color = "red")) +
    geom_vline(xintercept = mean(df_train[, f], na.rm = TRUE), linetype = "dashed", alpha = 0.5, color = "red") +
    geom_vline(xintercept = median(df_train[, f], na.rm = TRUE), linetype = "dashed", alpha = 0.5, color = "blue") +
    geom_vline(xintercept = as.numeric(calc_mode(df_train[, f])), linetype = "dashed", alpha = 0.5, color = "green") +
    labs(x="Value", y="Count")  +
    theme(axis.text=element_text(size=9), axis.title=element_text(size=11))
}

# Create the function to update df_fill_value
update_df_fill_value <- function(f, m, v) {
  df_fill_value %>%
    dplyr::mutate(Method = ifelse(Feature == f, m, Method)) %>%
    dplyr::mutate(Value = ifelse(Feature == f, v, Value))
}

# CNT_FAM_MEMBERS: 2 missing values: right skewed - filled by mode
plot_hist("CNT_FAM_MEMBERS")
summary(df_train$CNT_FAM_MEMBERS)
df_fill_value <- update_df_fill_value("CNT_FAM_MEMBERS", "mode", as.integer(names(which.max(table(df_train$CNT_FAM_MEMBERS)))))

# OBS_30_CNT_SOCIAL_CIRCLE: 887 missing values: right skewed - fill by mode
plot_hist("OBS_30_CNT_SOCIAL_CIRCLE")
summary(df_train$OBS_30_CNT_SOCIAL_CIRCLE)
df_fill_value <- update_df_fill_value("OBS_30_CNT_SOCIAL_CIRCLE", "mode", as.integer(calc_mode(df_train$OBS_30_CNT_SOCIAL_CIRCLE)))

# OBS_60_CNT_SOCIAL_CIRCLE: 887 missing values: right skewed - fill by mode
plot_hist("OBS_60_CNT_SOCIAL_CIRCLE")
summary(df_train$OBS_60_CNT_SOCIAL_CIRCLE)
df_fill_value <- update_df_fill_value("OBS_60_CNT_SOCIAL_CIRCLE", "mode", as.integer(calc_mode(df_train$OBS_60_CNT_SOCIAL_CIRCLE)))

# DEF_30_CNT_SOCIAL_CIRCLE: 887 missing values: right skewed - fill by mode
plot_hist("DEF_30_CNT_SOCIAL_CIRCLE")
summary(df_train$DEF_30_CNT_SOCIAL_CIRCLE)
df_fill_value <- update_df_fill_value("DEF_30_CNT_SOCIAL_CIRCLE", "mode", as.integer(calc_mode(df_train$DEF_30_CNT_SOCIAL_CIRCLE)))

# DEF_60_CNT_SOCIAL_CIRCLE: 887 missing values: right skewed - fill by mode
plot_hist("DEF_60_CNT_SOCIAL_CIRCLE")
summary(df_train$DEF_60_CNT_SOCIAL_CIRCLE)
df_fill_value <- update_df_fill_value("DEF_60_CNT_SOCIAL_CIRCLE", "mode", as.integer(calc_mode(df_train$DEF_60_CNT_SOCIAL_CIRCLE)))

# DAYS_LAST_PHONE_CHANGE: 1 missing values: left skewed - fill by mode
plot_hist("DAYS_LAST_PHONE_CHANGE")
summary(df_train$DAYS_LAST_PHONE_CHANGE)
df_fill_value <- update_df_fill_value("DAYS_LAST_PHONE_CHANGE", "mode", as.integer(calc_mode(df_train$DAYS_LAST_PHONE_CHANGE)))

# AMT_ANNUITY: 10 missing values: right skewed: high kurtosis - fill by median
plot_hist("AMT_ANNUITY")
summary(df_train$AMT_ANNUITY)
df_fill_value <- update_df_fill_value("AMT_ANNUITY", "median", median(df_train$AMT_ANNUITY, na.rm=TRUE))

# AMT_GOODS_PRICE: 275 missing values: right skewed: high kurtosis - fill by median
plot_hist("AMT_GOODS_PRICE")
summary(df_train$AMT_GOODS_PRICE)
df_fill_value <- update_df_fill_value("AMT_GOODS_PRICE", "median", median(df_train$AMT_GOODS_PRICE, na.rm=TRUE))

# EXT_SOURCE_1: 139170 missing values: not skewed: low kurtosis - fill by mean
plot_hist("EXT_SOURCE_1")
summary(df_train$EXT_SOURCE_1)
df_fill_value <- update_df_fill_value("EXT_SOURCE_1", "median", mean(df_train$EXT_SOURCE_1, na.rm=TRUE))

# EXT_SOURCE_2: 524 missing values: left skewed: high kurtosis - fill by median
plot_hist("EXT_SOURCE_2")
summary(df_train$EXT_SOURCE_2)
df_fill_value <- update_df_fill_value("EXT_SOURCE_2", "median", median(df_train$EXT_SOURCE_2, na.rm=TRUE))

# EXT_SOURCE_3: 49989 missing values: left skewed: high kurtosis - fill by median
plot_hist("EXT_SOURCE_3")
summary(df_train$EXT_SOURCE_3)
df_fill_value <- update_df_fill_value("EXT_SOURCE_3", "median", median(df_train$EXT_SOURCE_3, na.rm=TRUE))

# Replicate the dataframe
df_train_filled <- df_train
df_test_filled <- df_test

# Fill missing values
for (i in 1:length(df_fill_value$Feature)) {
  f <- df_fill_value$Feature[i]
  
  v <- df_fill_value %>%
    dplyr::filter(Feature == f) %>%
    dplyr::select(Value) %>%
    unlist()
  
  na_index_train <- is.na(df_train_filled[, f])
  na_index_test <- is.na(df_test_filled[, f])
  
  df_train_filled[na_index_train, f] <- v
  df_test_filled[na_index_test, f] <- v
}

# Sanity check
sum(is.na(df_train_filled)) # 0
sum(is.na(df_test_filled)) # 0

# ===================================================================================================
# 4. Anomalies
# ===================================================================================================
# Create the data frame to contain the necessary information
df_anom <- data.frame(Feature=c(col_int_use, col_num_use),
                            Type=c(rep("integer", length(col_int_use)), rep("numeric", length(col_num_use)))
)

# Calculate min
df_anom_min <- df_train_filled %>%
  dplyr::select(c(col_int_use, col_num_use)) %>%
  apply(2, min) %>%
  as.integer()

# Calculate max
df_anom_max <- df_train_filled %>%
  dplyr::select(c(col_int_use, col_num_use)) %>%
  apply(2, max) %>%
  as.integer()

# Combine
df_anom <- data.frame(df_anom, df_anom_min, df_anom_max)

# Create boxplot of DAYS_EMPLOYED
df_train_filled %>%
  dplyr::filter(DAYS_EMPLOYED != -365243) %>%
  ggplot(aes(x = DAYS_EMPLOYED)) +
  geom_boxplot()

# Create boxplot of AGE who has the anomaly of DAYS_EMPLOYED
df_train_filled %>%
  dplyr::filter(DAYS_EMPLOYED == -365243) %>%
  ggplot(aes(x = DAYS_BIRTH / 365)) +
  geom_boxplot()



# ===================================================================================================
# 5. Additional Features
# ===================================================================================================

# ===================================================================================================
# 6. Standardization
# ===================================================================================================

# ===================================================================================================
# 7. One Hot Encoding
# ===================================================================================================

