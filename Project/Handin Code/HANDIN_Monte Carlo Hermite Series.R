

########

# Author: Julian Valdman
# Edited: 07JUN2022

# File Purpose
# Performing Monte Carlo simulation on Hermite Series density estimation

########

### PREAMBLE

library("Rlab")  

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source("R Library files/lib_data_functions.R")
source("R Library files/lib_hermite_series_functions.R")
source("R Library files/lib_hermite_series_estimations.R")


#### Functions

simulate_data <- function() {
  
  all_V <- c()
  all_a <- c()
  all_b <- c()
  all_v <- c()
  all_o <- c()
  all_lnR <- c()
  all_lnV <- c()
  all_censored <- c()
  
  for (n in 1:N) {
    
    a1 = 2
    a2 = -3
    
    alpha = rnorm(n=1, mean=0, sd=1)
    beta = rexp(n=1, 1)
    v = rgamma(n=1, shape=9, rate=3)
    omega = rgamma(n=1, shape=9, rate=3) - 2
    
    V = exp(a1 * alpha + a2 * beta + v)
    lnV = a1 * alpha + a2 * beta + v
    lnR = a1 * alpha + a2 * beta + omega + a1 * (beta > 1) - a2 * (beta + 1) / alpha
    
    censored = as.integer(lnV < lnR)
    
    all_a <- append(all_a, alpha)
    all_b <- append(all_b, beta)
    all_v <- append(all_v, v)
    all_o <- append(all_o, omega)
    all_V <- append(all_V, V)
    all_lnV <- append(all_lnV, lnV)
    all_lnR <- append(all_lnR, lnR)
    all_censored <- append(all_censored, censored)
  }
  
  mat = matrix(c(all_a, all_b, all_v, all_o, all_V, all_lnV, all_lnR, all_censored), ncol=8)
  colnames(mat) <- c('alpha','beta', 'v', 'omega', 'V', 'lnV', 'lnR', 'b_Censored')
  
  return(mat)
}




### MAIN

S = 100     # Samples / iterations
N = 1000    # Sample size

## Export a simulation to feed ANN Monte Carlo

sim = simulate_data()
write.csv(sim, "data//bid_censored_simulated.csv")

all_mean = c() 
all_var = c() 

for (s in 1:S) {
  
  print(paste("S:" , s))
  tryCatch(
    {
      sim = simulate_data()
      lnV = sim[, "lnV"]
      
      fitted = find_coef_estimates(lnV, k=4)
      
      mean = find_mean(fitted)
      variance = find_variance(fitted)
      
      print(paste(mean, "--", variance))
      
      all_mean <- append(all_mean, mean)
      all_var <- append(all_var, variance)
    },
    error=function(cond) { print(cond) },
    finally={
      message("Finally")
    }
  )
}

# Present estimators

hist(all_mean, breaks=8, main="Mean Estimator", xlab="Mean", ylab="Samples")
hist(all_var,  breaks=8, main="Std. Dev. Estimator", xlab="Std. Dev.", ylab="Samples")





