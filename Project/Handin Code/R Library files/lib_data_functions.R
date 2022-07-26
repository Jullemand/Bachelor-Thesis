
########

# Author: Julian Valdman
# Edited: 07JUN2022

# File Purpose:
# Library of function related to data collection and preparation before estimations

########


### LIB

PATH = "C:/MyStuff/Own/KU/Bachelorprojekt/Valg af projekt/eBay/Code/Scraper/Sample code/auctions_data.xlsx"

load_ebay_data <- function(raw_data=F) {
  
  # Load eBay data collected using Web Scraper in Python
  
  ebay_data_path = PATH
  ebay_data = readxl::read_excel(ebay_data_path)
  
  if (raw_data) { return(ebay_data) }
  
  ebay_prices = ebay_data %>% select(price_dkk)
  ebay_prices = ebay_prices[[1]]
  
  assign("N", ebay_prices %>% length(), envir = .GlobalEnv)
  return(ebay_prices)
}


scale <- function(val, scaler, upper=T) {
  
  # Scale value given lower or upper bound
  
  if (!upper) {
    if (val >= 0) {
      val * (1 - scaler)
    } else {
      val * (1 + scaler)
    }
  } else {
    if (val >= 0) {
      val * (1 + scaler)
    } else {
      val * (1 - scaler)
    }
  }
}


IQR_outliers <- function(data, alpha=2) {
  
  # Removing outliers in data using Interquartile Ranges
  
  d_iqr = IQR(data)
  Q1 = quantile(data, prob=c(0.25))
  Q3 = quantile(data, prob=c(0.75))
  
  data <- data[data > (Q1 - alpha * d_iqr)]
  data <- data[data < (Q3 + alpha * d_iqr)]

  return(data)
  
}