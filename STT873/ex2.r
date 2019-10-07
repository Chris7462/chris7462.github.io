# Load the data
load('HW2_dat.Rdata')
x <- dat$X
y <- dat$Y

# define kernels
Gaussian <- function(X,Y){
  return(exp(-(X-Y)^2/0.25))
}
Laplacian <- function(X,Y){
  return(exp(-abs(X-Y)))
}
Polynomial <- function(X,Y){
  return((X*Y+1)^2)
}

RKHSReg <- function(x, y, lambda, kernel){
  kernels <- c('Gaussian', 'Laplacian', 'Polynomial')
  if( !(kernel %in% kernels) ){
    stop("Invalid kernel. Use: 'Gaussian', 'Laplacian' or 'Polynomial'.")
  }
  if ( lambda <= 0 ){
    stop("lambda must be greater than 0.")
  }
  Ker <- get(kernel)
  n <- length(x)
  X <- matrix(rep(x,n),n)
  K <- Ker(X,t(X))
  # fitted f.hat
  fitted <- K%*%solve(K+lambda*diag(n))%*%y
  return(fitted)
}
  
# run the RKHS regression with specified kernel and lambda
gauss.5 <- RKHSReg(x, y, lambda = 0.5, kernel = 'Gaussian')
gauss.01 <- RKHSReg(x, y, lambda = 0.01, kernel = 'Gaussian')
laplacian.5 <- RKHSReg(x, y, lambda = 0.5, kernel = 'Laplacian')
laplacian.01 <- RKHSReg(x, y, lambda = 0.01, kernel = 'Laplacian')
poly.5 <- RKHSReg(x, y, lambda = 0.5, kernel = 'Polynomial')
poly.2 <- RKHSReg(x, y, lambda = 0.2, kernel = 'Polynomial')

# plot the data and fitted values together
library(ggplot2)
library(reshape2)

# creates a dataframe of predicted values
RKHS.pred <- data.frame(x = x, "Gaussian_0.5" = gauss.5, 
	"Gaussian_0.01" = gauss.01, "Laplacian_0.5" = laplacian.5, 
	"Laplacian_0.01" = laplacian.01, "Polynomial_0.5" = poly.5, 
	"Polynomial_0.2" = poly.2)

# melt the data into the plotting format
plot.data <- melt(RKHS.pred, id ="x")
names(plot.data) <- c("x","Method", "Fitted")

# Create Plots
ggplot()+geom_point(data=as.data.frame(dat), mapping=aes(x=X,y=Y))+geom_line(data=plot.data, mapping=aes(x=x, y=Fitted, color=Method),size=1.3)+theme(axis.text=element_text(size=14), axis.title=element_text(size=16), legend.title=element_text(size=16), legend.text=element_text(size=14))
