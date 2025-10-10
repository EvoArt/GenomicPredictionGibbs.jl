library(Rcpp)
library(RcppEigen)

sourceCpp("src/bayes_b.cpp")

X <- as.matrix(read.csv("R/X.csv", header = TRUE)) # CSVs not tracked due to size
y <- as.vector(read.csv("R/y.csv", header = TRUE))

burn_in <- 50
beta_mean <- rowMeans(result$beta[, (burn_in + 1):ncol(result$beta)])
which(abs(beta_mean) > 0.01)
