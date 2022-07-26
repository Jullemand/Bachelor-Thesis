
########

# Author: Julian Valdman
# Edited: 12JUN2022

# File Purpose:
# Functions defining and enabling the Hermite Series Estimation

########


create_coefficient_guess = function(k,mean,std,min,max) {
  
  ### Setting the initial guess for the MLE
  
  u = 'c('
  if (k > 1) {
    for(i in seq(k-1)) {
      u = paste(u,'a',as.character(i),'=0,',sep='')
    }
  }
  u = paste(u,'a',as.character(k),'=0,m=',as.character(mean),',s=',as.character(std), ')',sep='')
  return(u)
}


create_constraints_lower = function(k,as,m,s,max.b) {
  
  ### Setting the lower boundary for the coefficient values 
  
  u = 'c('
  if (k > 1) {
    for(i in seq(k-1)) {
      u = paste(u,'a',as.character(i),'=',as.character(as),',',sep='')
    }
  }
  u = paste(u,'a',as.character(k),'=',as.character(as),',m=',as.character(m),',s=',as.character(s), ')',sep='')
  return(u)
}


create_constraints_upper = function(k,as,m,s,min.b) {
  
  ### Setting the upper boundary for the coefficient values 
  
  u = 'c('
  if (k > 1) {
    for(i in seq(k-1)) {
      u = paste(u,'a',as.character(i),'=',as.character(as),',',sep='')
    }
  }
  u = paste(u,'a',as.character(k),'=',as.character(as),',m=',as.character(m),',s=',as.character(s), ')',sep='')
  return(u)
}



#### CORE DENSITY ESTIMATION FUNCTIONS ####


# fhat0.hermite.stage.one = function(k) {
fhat_hermite_string = function(k) {
  
  ### Creating the Hermite Series polynomial of degree k as a string
  
  u = '1'
  for(i in seq(k)) {
    u = paste(u,'+a',as.character(i),'*((x-m)/s)^',as.character(i),sep='')
  }
  u = paste('(',u,')^2*dnorm(x,m,s)/(pnorm(max.x,m,s)-pnorm(min.x,m,s))',sep='')
  return(u)
}


# fhat0.stage.one = function(x,p,k) {
fhat_hermite_string_eval = function(x,p,k) {
  
  ### Evaluate the Hermite Series string given x-interval 
  ### and set of Hermite coefficients
  
  for(i in seq(k)) {
    eval(parse(text=paste('a',as.character(i),' = p[[\'a',as.character(i),'\']]',sep='')))
  }
  m = p[['m']]
  s = p[['s']]
  
  eval(parse(text=fhat_hermite_string(k)))
}

fhat_hermite_string_eval____mean = function(x,p,k) {
  
  ### Find the mean of a given estimated Hermite Series given 
  ### coefficients vector p
  
  for(i in seq(k)) {
    eval(parse(text=paste('a',as.character(i),' = p[[\'a',as.character(i),'\']]',sep='')))
  }
  m = p[['m']]
  s = p[['s']]
  
  t = fhat_hermite_string(k)
  
  # Add term to find density mean
  t = paste(t, " * x", sep="")
  
  eval(parse(text=t))
}

fhat_hermite_string_eval____variance = function(x,p,k,mean) {
  
  ### Find the variance of a given estimated Hermite Series given 
  ### coefficients vector p and mean
  
  for(i in seq(k)) {
    eval(parse(text=paste('a',as.character(i),' = p[[\'a',as.character(i),'\']]',sep='')))
  }
  m = p[['m']]
  s = p[['s']]
  
  t = fhat_hermite_string(k)
  
  # Add term to find density variance
  t = paste(t, " * (x - mean)^2", sep="")
  
  eval(parse(text=t))
}



# fhat.stage.one = function(x,p,k,d) {
hermite_pdf = function(x,p,k,d) {
  
  ### Find point density for x given coefficients p and 
  ### denominator d
  
  if(x<unname(min.x-1e-7)) {
    return(0)
  } else if(x>max.x+1e-7) {
    return(0)
  } else {
    return(fhat_hermite_string_eval(x,p,k)/d)
  }
}

### Running PDF for a vector x
v_hermite_pdf = Vectorize(hermite_pdf,'x')

# Fhat.stage.one = function(x,p,k,d) {
hermite_cdf = function(x,p,k,d) {
  
  ### Find cumulative probability for x given coefficients p and 
  ### denominator d
  
  if(x<min.x) {
    return(0)
  } else if(x>max.x) {
    return(1)
  } else {
    return(integrate(v_hermite_pdf, min.x, x,p=p,k=k,d=d)$value)
  }
}

### Running CDF for a vector x
v_hermite_cdf = Vectorize(hermite_cdf, 'x')






