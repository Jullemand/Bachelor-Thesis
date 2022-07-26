
########

# Author: Julian Valdman
# Edited: 07JUN2022

# File Purpose:
# Functions running the Hermite Estimation and function used after the estimation 

########


create_constraints <- function(k, data, ll = -Inf, uu = Inf) {
  
  # Setting coefficient boundaries
  min.std = 0.1
  max.std = 1e6
  min.mean = -1e6
  max.mean = 1e6
  min.params = -1e6
  max.params = 1e6
  
  constraints.min = eval(parse(text=create_constraints_lower(k, min.params, min.mean, min.std, "")))
  constraints.max = eval(parse(text=create_constraints_upper(k, max.params, max.mean, max.std, "")))
  starting.guess = eval(parse(text=create_coefficient_guess(k, mean(data), sd(data), "", "")))
  
  return(list(min=constraints.min,
              max=constraints.max,
              guess=starting.guess))
}


run_max_ll <- function(k, data, start_guess, constraint_min, constraint_max) {
  
  # Likelihood function
  NLL.stage.one = function(p) {
    
    # The integral denominator
    d = integrate(fhat_hermite_string_eval, min.x, max.x, p=p, k=k)$value
    - (1 / 1) * sum(log(v_hermite_pdf(data, p, k, d)))
    
  }
  
  parnames(NLL.stage.one) = names(start_guess)
  
  # Run Maximum Likelihood
  fit.stage.one = tryCatch(
    mle2(NLL.stage.one,start=start_guess, method='L-BFGS-B','optim',
         lower=constraint_min,
         upper=constraint_max,
         control=list(maxit=10000)),
    error=function(e) print(e)
  )
  
  return(fit.stage.one)
}


find_coef_estimates <- function(b, k=2) {
  
  min_b = b %>% min()
  max_b = b %>% max()
  
  eps = 0.02
  min.x = scale(min_b, eps, F)
  max.x = scale(max_b, eps, T)
  
  assign("min.x", min.x, envir = .GlobalEnv)
  assign("max.x", max.x, envir = .GlobalEnv)
  assign("k", k, envir = .GlobalEnv)
  
  con = create_constraints(k, b, ll = "", uu = "")
  
  # Fit parameters via MLL
  fitted <- run_max_ll(k, b, con$guess, con$min, con$max)
  
  # Plot empirical and fitted density
  plot_density(fitted, k, b)
  
  return(fitted)
}


### After fitting

plot_density <- function(fit.stage.one, k, data) {
  
  min.b = min(data)
  max.b = max(data)
  seq_b = seq(min.b,max.b,len=101)
  
  if(!inherits(fit.stage.one,'error')){
    # If successful estimation
    
    print(summary(fit.stage.one))
    params.stage.one = coef(fit.stage.one)
    d.stage.one = integrate(fhat_hermite_string_eval, min.x, max.x, p=params.stage.one,k=k)$value
    
    plot(seq_b, v_hermite_cdf(seq_b, params.stage.one, k, d.stage.one), ylim=c(0,1), xlab="Price, DKK", ylab="Cumulative Probability")
    lines(ecdf(data))
    
  } else {
    print(paste('Failed to run MLL', sep=''))
  } 
}


find_mean <- function(fitted) {
  
  params.stage.one = coef(fitted)
  d = integrate(fhat_hermite_string_eval, min.x, max.x, p=params.stage.one,k=k)$value
  
  mean = integrate(fhat_hermite_string_eval____mean, min.x, max.x, p=params.stage.one,k=k)$value / d
  return(mean)
  
}

find_variance <- function(fitted) {
  
  params.stage.one = coef(fitted)
  d = integrate(fhat_hermite_string_eval, min.x, max.x, p=params.stage.one,k=k)$value
  
  mean = find_mean(fitted)
  var = integrate(fhat_hermite_string_eval____variance, min.x, max.x, p=params.stage.one,k=k, mean=mean)$value / d
  return(var)
}

find_percentile <- function(fitted, percentile_measure = 0.5) {
  
  params.stage.one = coef(fitted)
  d = integrate(fhat_hermite_string_eval, min.x, max.x, p=params.stage.one,k=k)$value
  
  seq_b = seq(min.x, max.x, len=1000)
  
  median = 0 # Default value
  MARGIN = 0.005
  
  # Look where CDF value is 0.5
  for (i in seq_b) {
    percentile = hermite_cdf(i, params.stage.one, k, d)
    if (abs(percentile - percentile_measure) <= MARGIN) {
      median = i
      break
    }
  }
  return(median)
}






