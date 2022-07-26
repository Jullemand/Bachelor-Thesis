

########

# Author: Julian Valdman
# Edited: 07JUN2022

# File Purpose
# Perform Hermite Series density estimation (MLE) on eBay data set and subsamples thereof

########

### PREAMBLE

rm(list=ls())

library(bbmle)
library(devtools)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source("R Library files/lib_data_functions.R")
source("R Library files/lib_hermite_series_functions.R")
source("R Library files/lib_hermite_series_estimations.R")

### Functions


subset_get_price <- function(dataset) {
  
  p <- dataset %>% select(price_dkk)
  p <- p[[1]]
  p <- IQR_outliers(p, 2)
  return(p)
  
}


subset_analysis <- function(subsample_data, k = 4) {
  
  prices = subset_get_price(subsample_data)
  fitted_coef = find_coef_estimates(prices, k)
  
  print(find_mean(fitted_coef))
  print(find_variance(fitted_coef) %>% sqrt())
  return(fitted_coef)
  
}



#### MAIN ####

ebay_data <- load_ebay_data(raw_data = T)


## Define subsamples (ss)

X = ebay_data
ss_A = ebay_data %>% filter((seller_feedback_negative > 0))
ss_B = ebay_data %>% filter(is.na(seller_feedback_pct) == F)
ss_C = ebay_data %>% filter(seller_feedback_positive + seller_feedback_neutral + seller_feedback_negative > 10)
ss_D = ebay_data %>% filter(num_bidders <= 10)
ss_E = ebay_data %>% filter(bids >= 30)
ss_F = ebay_data %>% filter(duration <= 4)

# Run Hermite Series analysis on subsamples and plot
# Print mean and std. deviation of estimated density

subset_analysis(X, k=4)
subset_analysis(ss_A, k=4)
subset_analysis(ss_B, k=4)
subset_analysis(ss_C, k=4)
subset_analysis(ss_D, k=4)
subset_analysis(ss_E, k=4)
subset_analysis(ss_F, k=4)















