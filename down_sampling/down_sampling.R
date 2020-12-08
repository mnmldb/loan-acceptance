# ===================================================================================================
# 1. Preparation
# ===================================================================================================
# Import packages
library(dplyr)
library(ggplot2)
library(ggsci)

# Import training and test data
df_train_all <- read.csv("./processed_data/train_model.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE)
df_test_all <- read.csv("./processed_data/test_model.csv", header=T, sep=",", na.strings=c('', 'NULL', '""'), stringsAsFactors=FALSE)

# ===================================================================================================
# 2. Down Sampling
# ===================================================================================================
# Number of negative records in training set: 230843
n_train <- df_train_all %>%
  dplyr::filter(TARGET == 0) %>%
  count() %>%
  unlist()

# Number of positive records in training set: 20478
p_train <- df_train_all %>%
  dplyr::filter(TARGET == 1) %>%
  count() %>%
  unlist()

# Random sampling from training set with target 0
df_train_n <- df_train_all %>%
  dplyr::filter(TARGET == 0) %>%
  dplyr::sample_n(p_train)

# All records from training set with target 1
df_train_p <- df_train_all %>%
  dplyr::filter(TARGET == 1)

# Concat two data frames
df_train_ds <- rbind(df_train_n, df_train_p)

# ===================================================================================================
# 3. Export
# ===================================================================================================
write.csv(df_train_ds, "./processed_data/train_model_ds.csv", row.names=FALSE, na="")
