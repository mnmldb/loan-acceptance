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
# 2. Missing Values
# ===================================================================================================
# Create the data frame to contain the filling value
df_fill_value <- data.frame(Feature=c(col_int_use_na, col_num_use_na),
                            Type=c(rep("integer", length(col_int_use_na)), rep("numeric", length(col_num_use_na))),
                            Method=c(rep(NA, length(col_int_use_na) + length(col_num_use_na))),
                            Value=c(rep(NA, length(col_int_use_na) + length(col_num_use_na)))
                              )

# Create the function to plot the histogram
plot_hist <- function(f) {
  df_train %>%
    # ggplot(aes(x=df_train[, f], y = ..density..)) +
    ggplot(aes(x=df_train[, f])) +
    geom_histogram(position = "identity", alpha = 0.5) +
    # geom_density(aes(alpha = 0.2, color = "red")) +
    labs(x="Value", y="Count")  +
    theme(axis.text=element_text(size=9), axis.title=element_text(size=11))
}

# Create the function to update df_fill_value
update_df_fill_value <- function(f, m, v) {
  df_fill_value %>%
    dplyr::mutate(Method = ifelse(Feature == f, m, Method)) %>%
    dplyr::mutate(Value = ifelse(Feature == f, v, Value))
}

# Create the function to return the mode
calc_mode <- function(a) {
  return(names(which.max(table(a))))
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


# ===================================================================================================
# 3. Anomalies
# ===================================================================================================

# ===================================================================================================
# 4. Additional Features
# ===================================================================================================

# ===================================================================================================
# 5. Standardization
# ===================================================================================================

# ===================================================================================================
# 6. One Hot Encoding
# ===================================================================================================

