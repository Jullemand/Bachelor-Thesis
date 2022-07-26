

########

# Author: Julian Valdman
# Edited: 07JUN2022

# File Purpose
# Perform Maximum Likelihood Estimation on eBay data using Normal Density

########

### PREAMBLE

library(maxLik)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source("R Library files/lib_TOEDIT.R")
source("R Library files/lib_data_functions.R")


### Functions

NLL <- function(pars, data) {
  
  mu= pars[1]
  sigma = pars[2]
  
  -sum(log(dnorm(x = data, mean = mu, sd = sigma))) / N
}



### MAIN

data = load_ebay_data(F)
data = IQR_outliers(data, 2)

N <- data %>% length()

MLE <- optim(par = c(mu = data %>% mean(), sigma=data %>% sd()),
            fn = NLL,
            data = data,
            hessian = T
)

coef       = MLE$par
std_errors = MLE$hessian %>% diag() %>% sqrt()


## Plot Empirical and estimated CDF  

seq.x = seq(min(data), max(data), len=100)

plot(seq.x, pnorm(seq.x, mean=coef["mu"], sd=coef["sigma"])*1, xlab="Auction Price, DKK", ylab="Cumulative Probability")
lines(ecdf(data))

