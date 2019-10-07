library(ElemStatLearn)
library(pls)
library(dplyr)
library(ggplot2)
library(ridge)
library(lasso2)
library(glmnet)
library(xtable)
library(cowplot)
library(reshape2)
library(elasticnet)
library(selectiveInference)

# Question 1
# data processing
data("prostate")
prostate.std <- data.frame(scale(prostate[!colnames(prostate) %in% c("lpsa", "train")]))
prostate.std$lpsa <- prostate$lpsa
prostate.train <- prostate.std[prostate$train, ]
prostate.test <- prostate.std[!prostate$train, ]
n.train <- dim(prostate.train)[1]
n.test <- dim(prostate.test)[1]

# linear regression
lm.train <- lm(lpsa ~ ., data = prostate.train)
lm.test.pred <- predict(lm.train, newdata = prostate.test)
lm.test.err <- mean((lm.test.pred - prostate.test$lpsa)^2)
lm.test.se <- sd((lm.test.pred - prostate.test$lpsa)^2)/sqrt(n.test)
LS <- round(c(coef(lm.train), lm.test.err, lm.test.se),3)
coef.table <- data.frame(LS, row.names = c(names(coef(lm.train)), "Test Error", "Std Error"))

# best subset
p <- 8	# number of variables 
# list all possible subset models
train.formulas <- vector()
vars <- names(prostate.train)[1:p]
subset.size <- vector()
for ( i in 1:2^p ){
	vars.index <- as.logical(intToBits(i-1)[1:p])
	subset.size[i] <- sum(vars.index)
  sub.vars <- vars[vars.index]
  train.formulas[i] <- paste("lpsa~", paste(c(1,sub.vars), collapse = "+"), sep = "")
}

# best subset cross validation
set.seed(1)
k <- 10
folds <- cvsegments(n.train, k)
subset.errors <- matrix(NA, nrow = 10, ncol = 2^p)
for (i in 1:k){
  testIndexes <- unlist(folds[i])
  cv.test <- prostate.train[testIndexes,]
  cv.train <- prostate.train[-testIndexes,]
  test.errors <- vector()
  for (j in 1:length(train.formulas)){
    train.lm <- lm(formula(train.formulas[j]), data = cv.train)
    test.pred <- predict(train.lm, newdata = cv.test)
    test.errors <- c(test.errors, mean((test.pred - cv.test$lpsa)^2))
  }
	subset.errors[i,] <- test.errors
}

# finds the best model
mean.errors <- apply(subset.errors, 2, mean)
sd.errors <- apply(subset.errors, 2, sd)/sqrt(k)
subset.df <- data.frame(train.formulas, "subset.size" = as.numeric(subset.size), 
                        "mean.errors" = as.numeric(mean.errors), 
                        "sd.errors" = as.numeric(sd.errors), 
                        stringsAsFactors = FALSE)
subset.df <- subset.df %>% group_by(subset.size) %>% mutate(error = min(mean.errors)) %>% distinct(subset.size, .keep_all = TRUE)
b <- which.min(subset.df$mean.errors)
best.mod <- which(subset.df$mean.errors < subset.df$mean.errors[b] + subset.df$sd.errors[b])[1]

# best subset selection plot
d <- 1:p
best.x <- c(0, d)[best.mod]
best.y <- subset.df$mean.errors[best.mod]
best.plot.data <- data.frame(cbind(c(0, d),subset.df$mean.errors, subset.df$sd.errors))
subset.plot <- ggplot(data = best.plot.data, aes(x = X1, y = X2)) + 
  geom_line() + geom_errorbar(aes(ymin = X2-X3, ymax = X2+X3), width = 0.25, colour = "grey50") +
  geom_point()+
  labs(title = "All Subsets", 
       x = "Subset Size", y = "CV Error") +
  theme(axis.text = element_text(size = 8), 
        axis.title = element_text(size = 8)) +
  scale_y_continuous(limits = c(0.3, 1.8), 
                     breaks = seq(0.4, 1.8, 0.2)) +
  geom_vline(xintercept = best.x, linetype = 2, 
             colour = "blue") + 
  geom_hline(yintercept = best.y, linetype = 2, 
             colour = "blue")

# testing error
bs.train <- lm(formula(subset.df$train.formulas[best.mod]), data = prostate.train)
bs.test.pred <- predict(bs.train, newdata = prostate.test)
bs.test.err <- mean((bs.test.pred - prostate.test$lpsa)^2)
bs.test.se <- sd((bs.test.pred - prostate.test$lpsa)^2)/sqrt(n.test)

# best subset selection column for table
bs.coef <- round(coef(bs.train),3)
BS <- rep("", p+1)
bs.index <- which(names(bs.coef) %in% rownames(coef.table))
BS[bs.index] <- bs.coef
BS <- c(BS, round(bs.test.err,3), round(bs.test.se,3))
coef.table <- cbind(coef.table, BS)


# ridge regression
ridge.errors <- data.frame()
for (i in 1:k){
  testIndexes <- unlist(folds[i])
  cv.test <- prostate.train[testIndexes, ]
  cv.train <- prostate.train[-testIndexes, ]
  null.model <- lm(lpsa ~ 1, data = cv.train)
  null.pred <- predict(null.model, newdata = cv.test)
  null.resid <- null.pred - cv.test$lpsa
  mean.errors <- mean(null.resid^2)
  test.errors <- c(mean.errors)
  for (j in 1:length(d)){
    df <- d[j]
    p <- NCOL(cv.train)
    svd.x <- svd(cbind(cv.train, 1), nu = p, nv = p)
    dd <- svd.x$d
    fun <- function(df, lambda) df - sum(dd^2/(dd^2 + lambda))
    lambda <- sapply(df, FUN = function(df) uniroot(f = function(lambda) fun(df, lambda), 
                                                    lower = -1e-06, upper = 1000, 
                                                    maxiter = 10000)$root)
    ridge.mod <- linearRidge(lpsa ~ . , data = cv.train, 
                             lambda = lambda, scaling = "scale")
    ridge.pred <- predict(ridge.mod, newdata = cv.test)
    ridge.resid <- cv.test[, 9] - ridge.pred
    test.errors <- c(test.errors, mean(ridge.resid^2))
  }
  ridge.errors <- rbind(ridge.errors, test.errors)
}

# finds the best model
mean.errors <- apply(ridge.errors, 2, mean)
sd.errors <- apply(ridge.errors, 2, sd)/sqrt(10)
b <- which.min(mean.errors)
best.mod <- which(mean.errors < mean.errors[b] + sd.errors[b])[1]

# ridge regression plot
best.x <- c(0, d)[best.mod]
best.y <- mean.errors[best.mod]
ridge.plot.data <- data.frame(cbind(c(0, d), mean.errors, sd.errors))
ridge.plot <- ggplot(data = ridge.plot.data, aes(x = V1, y = mean.errors)) + 
  geom_line() +
  geom_errorbar(aes(ymin = mean.errors-sd.errors, 
                    ymax = mean.errors+sd.errors), 
                width = 0.25, colour = "grey50") +
  geom_point()+
  labs(title = "Ridge Regression", 
       x = "Degrees of Freedom", y = "CV Error") +
  theme(axis.text = element_text(size = 8), 
        axis.title = element_text(size = 8)) +
  scale_y_continuous(limits = c(0.3, 1.8), 
                     breaks = seq(0.4, 1.8, 0.2)) +
  geom_vline(xintercept = best.x, 
             linetype = 2, colour = "blue") + 
  geom_hline(yintercept = best.y, 
             linetype = 2, colour = "blue")

# testing error
df <- best.x
p <- NCOL(prostate.train[, 1:8])
svd.x <- svd(prostate.train[, 1:8], nu = p, nv = p)
dd <- svd.x$d
fun <- function(df, lambda) df - sum(dd^2/(dd^2 + lambda))
lambda <- sapply(df, FUN = function(df) uniroot(f = function(lambda) fun(df, lambda), 
                                                lower = -1e-06, upper = 1000, 
                                                maxiter = 10000)$root)
ridge.mod <- linearRidge(lpsa ~., data = prostate.train, 
                         lambda = lambda, scaling = "scale")
ridge.pred <- predict(ridge.mod, newdata = prostate.test)
ridge.error <- mean((prostate.test$lpsa - ridge.pred)^2)

# ridge regression column for table
ridge.stde <- sd((prostate.test$lpsa-ridge.pred)^2)/sqrt(n.test)
ridge.coef <- coef(ridge.mod)
Ridge <- round(c(ridge.coef, ridge.error, ridge.stde),3)
coef.table <- cbind(coef.table, Ridge)


# Lasso
s <- seq(0.0001, 1, 0.1249)
# lasso cross validation
lasso.errors <- data.frame()
for (i in 1:k){
  testIndexes <- unlist(folds[i])
  cv.test <- prostate.train[testIndexes, ]
  cv.train <- prostate.train[-testIndexes, ]
  test.errors <- vector()
  for (j in 1:length(s)){
    lasso.mod <- l1ce(lpsa ~ ., data = cv.train, 
                      bound = s[j], sweep.out = ~ 1)
    lasso.pred <- predict(lasso.mod, newdata = cv.test)
    lasso.resid <- cv.test[, 9] - lasso.pred
    test.errors[j] <- mean(lasso.resid^2)
  }
  lasso.errors <- rbind(lasso.errors, test.errors)
}

# finds the best model
mean.errors <- apply(lasso.errors, 2, mean)
sd.errors <- apply(lasso.errors, 2, sd)/sqrt(10)
b <- which.min(mean.errors)
best.mod <- which(mean.errors < mean.errors[b] + sd.errors[b])[1]

# lasso plot
best.x <- s[best.mod]
best.y <- mean.errors[best.mod]
lasso.plot.data <- data.frame(cbind(s, mean.errors, sd.errors))
lasso.plot <- ggplot(data = lasso.plot.data, aes(x = s, y = mean.errors)) + 
  geom_line() +
  geom_errorbar(aes(ymin = mean.errors-sd.errors, 
                    ymax = mean.errors+sd.errors), 
                width = .05, colour = "grey50") +
  geom_point()+
  labs(title = "Lasso", 
       x = "Shrinkage Factor s", y = "CV Error") +
  theme(axis.text = element_text(size = 8), 
        axis.title = element_text(size = 8)) +
  scale_y_continuous(limits = c(0.3, 1.8), 
                     breaks = seq(0.4, 1.8, 0.2)) +
  scale_x_continuous(breaks = seq(0, 1, 0.2)) +
  geom_vline(xintercept = best.x, 
             linetype = 2, colour = "blue") + 
  geom_hline(yintercept = best.y, 
             linetype = 2, colour = "blue")

# testing error
s <- best.x
lasso.mod <- l1ce(lpsa ~ ., data = prostate.train, 
                  bound = s, sweep.out = ~ 1)
lasso.pred <- predict(lasso.mod, newdata = prostate.test)
lasso.error <- mean((prostate.test$lpsa - lasso.pred)^2)
lasso.stde <- sd((prostate.test$lpsa - lasso.pred)^2)/sqrt(n.test)

# lasso column for table
lasso.coef <- round(lasso.mod$coefficients, 3)
lasso.coef <- ifelse(lasso.coef ==  0, "", lasso.coef)
Lasso <- c(lasso.coef, round(lasso.error, 3), round(lasso.stde, 3))
coef.table <- cbind(coef.table, Lasso)


# PCR
pcr.train <- pcr(lpsa ~ ., data = prostate.train, 
                 method = pls.options()$pcralg, 
                 validation = "CV", segments = folds)
pcr.cv <- pcr.train$validation
pcr.resid <- matrix(pcr.train$residuals^2 , ncol = 8)

# PCR cross validation
pcr.errors <- data.frame()
for (i in 1:k){
  pred.index <- unlist(pcr.cv$segments[i])
  cv.resid <- pcr.resid[pred.index, ]
  null.model <- lm(lpsa ~ 1, 
                   data = prostate.train[-pred.index, ])
  null.pred <- predict(null.model, 
                       newdata = prostate.train[pred.index, ])
  null.resid <- null.pred - prostate.train$lpsa[pred.index]
  mean.errors <- apply(cbind(null.resid^2, cv.resid), 2, mean)
  pcr.errors <- rbind(pcr.errors, mean.errors)
}

# finds the best model
mean.errors <- apply(pcr.errors, 2, mean)
sd.errors <- apply(pcr.errors, 2, sd)/sqrt(k)
b <- which.min(mean.errors)
best.mod <- which(mean.errors < mean.errors[b] + sd.errors[b])[1]

# PCR plot
best.x <- best.mod-1
best.y <- mean.errors[best.mod]
pcr.plot.data <- data.frame(cbind(c(0:8), mean.errors, sd.errors))
pcr.plot <- ggplot(data = pcr.plot.data, aes(x = V1, 
                                             y = unlist(mean.errors))) + 
  geom_line() +
  geom_errorbar(aes(ymin = mean.errors-sd.errors, 
                    ymax = mean.errors+sd.errors), 
                width = .5, colour = "grey50") +
  geom_point()+
  labs(title = "Principal Components Regression", 
       x = "Number of Directions", y = "CV Error") +
  theme(axis.text = element_text(size = 8), 
        axis.title = element_text(size = 8)) +
  scale_y_continuous(limits = c(0.3, 1.8), 
                     breaks = seq(0.4, 1.8, 0.2)) +
  geom_vline(xintercept = best.x, linetype = 2, 
             colour = "blue") + 
  geom_hline(yintercept = best.y, linetype = 2, 
             colour = "blue")

# testing error
pcr.mod <- pcr(lpsa ~ ., data = prostate.train, 
               method = pls.options()$pcralg, ncomp = best.mod-1)
pcr.pred <- predict(pcr.mod, newdata = prostate.test)
pcr.error <- mean((pcr.pred - prostate.test$lpsa)^2)
pcr.stde <- sd((pcr.pred - prostate.test$lpsa)^2)/sqrt(n.test)

# PCR column for table
pcr.betas <- coef(pcr.mod, intercept = TRUE)
PCR <- round(c(pcr.betas, pcr.error, pcr.stde),3)
coef.table <- cbind(coef.table, PCR)


# PLSR
plsr.train <- plsr(lpsa ~ . , data = prostate.train, 
                   method = pls.options()$plsralg, 
                   validation = "CV", segments = folds)
plsr.cv <- plsr.train$validation
plsr.resid <- matrix(plsr.train$residuals^2 , ncol = 8)

# PLSR cross validation
plsr.errors <- data.frame()
for (i in 1:k){
  pred.index <- unlist(plsr.cv$segments[i])
  cv.resid <- plsr.resid[pred.index, ]
  null.model <- lm(lpsa ~ 1, 
                   data = prostate.train[-pred.index, ])
  null.pred <- predict(null.model, 
                       newdata = prostate.train[pred.index, ])
  null.resid <- null.pred - prostate.train$lpsa[pred.index]
  mean.errors <- apply(cbind(null.resid^2, cv.resid), 2, mean)
  plsr.errors <- rbind(plsr.errors, mean.errors)
}

# finds the best model
mean.errors <- apply(plsr.errors, 2, mean)
sd.errors <- apply(plsr.errors, 2, sd)/sqrt(k)
b <- which.min(mean.errors)
best.mod <- which(mean.errors < mean.errors[b] + sd.errors[b])[1]

# PLSR plot
best.x <- best.mod-1
best.y <- mean.errors[best.mod]
plsr.plot.data <- data.frame(cbind(c(0:8), mean.errors, sd.errors))
plsr.plot <- ggplot(data = plsr.plot.data, aes(x = V1, 
                                               y = unlist(mean.errors))) + 
  geom_line() +
  geom_errorbar(aes(ymin = mean.errors-sd.errors, 
                    ymax = mean.errors+sd.errors), 
                width = .5, colour = "grey50") +
  geom_point()+
  labs(title = "Partial Least Squares", 
       x = "Number of Directions", y = "CV Error") +
  theme(axis.text = element_text(size = 8), 
        axis.title = element_text(size = 8)) +
  scale_y_continuous(limits = c(0.3, 1.8), 
                     breaks = seq(0.4, 1.8, 0.2)) +
  geom_vline(xintercept = best.x, linetype = 2, 
             colour = "blue") + 
  geom_hline(yintercept = best.y, linetype = 2, 
             colour = "blue")

# testing error
plsr.mod <- plsr(lpsa ~ ., data = prostate.train, 
                 method = pls.options()$plsralg, ncomp = best.mod-1)
plsr.betas <- coef(plsr.mod, intercept = TRUE)
plsr.pred <- predict(plsr.mod, newdata = prostate.test)
plsr.error <- mean((plsr.pred - prostate.test$lpsa)^2)
plsr.stde <- sd((plsr.pred - prostate.test$lpsa)^2)/sqrt(n.test)

# PLSR column for table
plsr.betas <- coef(plsr.mod, intercept = TRUE)
PLSR <- round(c(plsr.betas, plsr.error, plsr.stde),3)
coef.table <- cbind(coef.table, PLSR)

coef.xtable <- xtable(coef.table, digits = 3,
                      caption = "Estimated coefficients and test error results,
                      for different subset and shrinkage methods applied to the prostate data.
                      The blank entries correspond to variables omitted.")
print.xtable(coef.xtable,
             hline.after = c(0, -1, 9, nrow(coef.xtable)),
             caption.placement = "top")

plot_grid(subset.plot, ridge.plot, lasso.plot, pcr.plot, plsr.plot, nrow = 3)


# Question 2
load("./Lasso1.Rdat")
N <- 10000
K <- min(N-1, ncol(dat1$X)) + 1
lambda.k <- vector()
n.lambda <- 10000

# step 1
dat1$X <- scale(dat1$X)
r.k <- matrix(ncol = K, nrow = nrow(dat1$X))
b.k <- matrix(ncol = K, nrow = ncol(dat1$X))
r.k[, 1] <- dat1$y - mean(dat1$y)
b.k[, 1] <- rep(0, ncol(dat1$X))

# step 2
A <- which.max(abs(r.k[, 1] %*% dat1$X))
lambda.k[1] <- abs(r.k[, 1] %*% dat1$X[, A])
lambdas <- seq(0.001, lambda.k[1], length.out = n.lambda)
X.A <- dat1$X[, A]

# step 3
for (k in 2:K){
  crosses.zero <- TRUE
  count <- 0
  while (crosses.zero ==  TRUE){
    count <- count + 1
    # (a)
    X.A <- dat1$X[, A]
    delta <- 1/lambda.k[k-1] * solve(t(X.A) %*% X.A) %*% t(X.A) %*% r.k[, k-1]
    Delta <- matrix(rep(0, ncol(dat1$X)), ncol = 1)
    Delta[A] <- delta
    # (b)
    b.l <- matrix(nrow = ncol(dat1$X), ncol = n.lambda)
    r.l <- matrix(nrow = nrow(dat1$X), ncol = n.lambda)
    for (i in 1:n.lambda){
      b.l[, i] <- b.k[, k-1] + (lambda.k[k-1] - lambdas[i]) * Delta
      r.l[, i] <- dat1$y - dat1$X %*% b.l[, i]
    }
    # (c)
    l <- 0
    for (i in n.lambda:1){
      if (l ==  0 && lambdas[i] < lambda.k[k-1]){
        inner.prod <- abs(r.l[, i] %*% dat1$X)
        l.temp <- which(inner.prod > min(inner.prod[A]))
        l.temp <- l.temp[!l.temp %in% A]
        l <- ifelse(length(l.temp)>0, l.temp, l)
        lambda.index <- i
      }
    }
    a <- length(A)
    sign.diff <- ifelse(b.k[A[-a], k-1] >=  0, 1, 0) ==  sign(b.l[A[-a], lambda.index])
    if (sum(sign.diff) !=  length(A[-a])){
      A <- A[-which(sign.diff ==  F)[1]]
    } else{
      crosses.zero <- FALSE
    }
  }
  
  lambda.k[k] <- lambdas[lambda.index]
  
  # (d)
  A <- c(A, l)
  X.A <- dat1$X[, A]
  b.k[, k] <- b.l[, lambda.index]
  r.k[, k] <- dat1$y - dat1$X %*% b.k[, k]
}

# lasso solution path plot
plot.data <- data.frame(cbind(lambda.k, t(b.k)))
plot.data <- plot.data[, colSums(plot.data !=  0) > 0]
plot.data <- melt(plot.data, id = "lambda.k")
solution.path <- ggplot(plot.data, aes(x = log(lambda.k),
                                       y = value, colour = variable)) +
  geom_line(aes(color = variable)) +
  scale_x_reverse() +
  theme(axis.text = element_text(size = 8), 
        axis.title = element_text(size = 8)) +
  theme(legend.position="none")
zoom <- ggplot(plot.data, aes(x = log(lambda.k),
                              y = value, colour = variable)) +
  geom_line(aes(color = variable)) +
  scale_x_reverse(lim = c(3.15, 2.5)) +
  scale_y_continuous(lim = c(-0.01, 0.1)) +
  theme(axis.text = element_text(size = 8), 
        axis.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.title = element_text(size = 8))
plot_grid(solution.path, zoom, nrow = 1, labels = "auto", label_size = 8)

# (2)
load("./Lasso2.Rdat")

# lasso solution path
lasso.mod <- enet(dat2$X, dat2$y, lambda = 0)

# elastic net solution path
enet.mod <- enet(dat2$X, dat2$y, lambda = 0.5)

# plots the solution paths
par(mfrow = c(1, 2), mar = c(3.9,4.3,1.4,0.3), mgp = c(2.6,1,0))
plot(lasso.mod, main = "Lasso")
plot(enet.mod, main = "Elastic Net")

# (3)
# bootstrap
load("./Lasso2.Rdat")
n <- nrow(dat2$X)
k <- 10
train.X <- data.frame(scale(dat2$X[sample(n), ]))
train.y <- data.frame(dat2$y[sample(n)])
names(train.y) <- "y"
random.train <- cbind(train.y, train.X)
folds <- cut(seq(1, n), breaks = k, labels = FALSE)
s <- seq(0.01, 1, 0.01)

# lasso cross validation for bootstrap
lasso.errors <- data.frame()
for (i in 1:k){
  testIndexes <- which(folds ==  i)
  cv.test <- random.train[testIndexes, ]
  cv.train <- random.train[-testIndexes, ]
  test.errors <- vector()
  for (j in 1:length(s)){
    lasso.mod <- l1ce(y ~ ., data = cv.train, 
                      bound = s[j], standardize = FALSE)
    lasso.pred <- predict(lasso.mod, newdata = cv.test)
    lasso.resid <- cv.test$y - lasso.pred
    test.errors[j] <- mean(lasso.resid^2)
  }
  lasso.errors <- rbind(lasso.errors, test.errors)
}

# finds the best model
mean.errors <- apply(lasso.errors, 2, mean)
sd.errors <- apply(lasso.errors, 2, sd)/sqrt(10)
b <- which.min(mean.errors)
best.mod <- which(mean.errors < mean.errors[b] + sd.errors[b])[1]
best.x <- s[best.mod]
best.y <- mean.errors[best.mod]

# finds lambda_cv
lasso.mod <- cv.glmnet(dat2$X, dat2$y, alpha = 1)
lasso.coef <- coef(lasso.mod, lasso.mod$lambda.1se)
lambda.cv <- lasso.mod$lambda.1se

# bootstrapping for coefficients
N <- 1000
bootstrap.coef <- data.frame(paste("Beta", c(0:200), sep = ""))
lambda.N <- vector()
for (i in 1:N){
  lasso.betas <- rep(0, 201)
  
  # repeated random samples with replacement
  samp <- sample(1:nrow(dat2$X), nrow(dat2$X), replace = TRUE)
  X <- dat2$X[samp, ]
  y <- dat2$y[samp]
  
  # finds the lasso coefficients
  lasso.cv <- cv.glmnet(X, y, alpha = 1)
  lasso.coef <- coef(lasso.cv, lasso.cv$lambda.1se)
  lambda.N[i] <- lasso.cv$lambda.1se
  lasso.mod <- glmnet(dat2$X, dat2$y, 
                      alpha = 1, lambda = lambda.N[i])
  lasso.coef <- predict(lasso.mod, type = "coefficients")@x
  lasso.coef.index <- c(1, lasso.mod$beta@i + 2)
  lasso.betas[lasso.coef.index] <- lasso.coef
  
  bootstrap.coef <- cbind(bootstrap.coef, lasso.betas)
}

# finds all nonzero coefficients
names(bootstrap.coef) <- c("coef", paste("it.", c(1:N), sep = ""))
non.zero <- which(apply(bootstrap.coef[, -1], 1, sum) !=  0)
non.zero.coef <- bootstrap.coef[non.zero, ]

# bootstrap boxplots
axis.labels <- vector()
for (i in 1:length(non.zero)){
  axis.labels[i] <- parse(text = paste("beta[", non.zero[i] - 1, "]", sep = ""))
}
ggplot(melt(non.zero.coef, id = "coef"), aes(x = coef, y = value)) +
  geom_vline(xintercept = seq(1, length(non.zero)),color="gray85") +
  geom_boxplot(outlier.size = 2,
               fill = "darkslategray3", outlier.shape = 124) +
  scale_x_discrete(labels = axis.labels) +
  geom_hline(yintercept = 0, colour = "red", 
             linetype = "dashed", size = 0.4) +
  labs(title = "1000 Bootstrap Realizations for Lasso", y = "Coefficients", x = "") +
  theme(axis.text = element_text(size = 8), 
        axis.title = element_text(size = 8)) +
  coord_flip() 

# (4)
# post selection inference for lasso
# data import and initializing
load("./Lasso2.Rdat")
X <- scale(dat2$X)
y <- dat2$y
n <- nrow(X)
lambda <- 0.1

# runs lasso 
lasso.mod <- glmnet(X, y, alpha = 1, standardize = F, 
                    maxit = 1e9, thresh = 1e-20)

# lasso coefficients
lasso.betas <- coef(lasso.mod, s = lambda/n, 
                    exact = TRUE, x = X, y = y)[-1]

# estimated lasso sigma
sig <- estimateSigma(X, y, standardize = F)$sigmahat

# post selection inference for lasso
lasso.inf <- fixedLassoInf(X, y, beta = lasso.betas, 
                           lambda = lambda, alpha = 0.05, 
                           sigma = sig, tol.kkt = 0.1, 
                           tol.beta = 1e-5, bits = 400)

# post selection inference plot
lasso.ci <- lasso.inf$ci
lasso.ci[, 1] <- ifelse(lasso.ci[, 1] ==  Inf, 
                        2500, lasso.ci[, 1])
lasso.ci[, 2] <- ifelse(lasso.ci[, 2] ==  Inf, 
                        2500, lasso.ci[, 2])
lasso.ci[, 1] <- ifelse(lasso.ci[, 1] ==  -Inf, 
                        -2500, lasso.ci[, 1])
lasso.ci[, 2] <- ifelse(lasso.ci[, 2] ==  -Inf, 
                        -2500, lasso.ci[, 2])
ols.X <- X[, lasso.inf$vars]
ols <- lm(y ~ 0 + ols.X)
ols.ci <- confint(ols, level = 0.95)
plot.data <- data.frame(rep(lasso.inf$vars, 2))
plot.data$lower <- c(lasso.ci[, 1], ols.ci[, 1])
plot.data$upper <- c(lasso.ci[, 2], ols.ci[, 2])
plot.data$Method <- c(rep("Lasso", 78), rep("OLS", 78))
names(plot.data)[1] <- "vars"
ggplot(plot.data, aes(x = vars, ymax = upper, 
                      ymin = lower, colour = Method)) +
  geom_linerange(size = 0.75, position = position_dodge(1)) + 
  scale_color_manual(values = c("darkred", "blue")) + 
  theme(axis.text = element_text(size = 8), 
        axis.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.title = element_text(size = 8)) +
  labs(title = "Confidence Intervals for Lasso and OLS", 
       x = "X", y = "Coefficient") +
  scale_y_continuous(breaks = c(-2500, -2000, -1000, 0, 
                                1000, 2000, 2500), 
                     labels = c("-2500" = "-Inf", 
                                "-2000" = "-2000", 
                                "-1000" = "-1000", 
                                "0" = "0", 
                                "1000" = "1000", 
                                "2000" = "2000", 
                                "2500" = "Inf"))
