#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>
#include <cmath>

using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::depends(RcppEigen)]]


// rng
std::random_device rd;
std::mt19937 gen(rd());

// helper funcs for distributions
double rnorm(double mean, double sd) {
  std::normal_distribution<> d(mean, sd);
  return d(gen);
}

bool rbernoulli(double p) {
  std::bernoulli_distribution d(p);
  return d(gen);
}

double rinvgamma(double shape, double scale) {
  std::gamma_distribution<> d(shape, 1.0 / scale);
  return 1.0 / d(gen);
}

double rbeta(double alpha, double beta) {
  std::gamma_distribution<> d1(alpha, 1.0);
  std::gamma_distribution<> d2(beta, 1.0);
  double x = d1(gen);
  double y = d2(gen);
  return x / (x + y);
}

double logpdf_norm(double x, double mean, double sd) {
  double z = (x - mean) / sd;
  return -0.5 * log(2 * M_PI) - log(sd) - 0.5 * z * z;
}

// [[Rcpp::export]]
List Bayes_B_Gibbs_step(
    const Eigen::Map<Eigen::MatrixXd>& X,
    const Eigen::Map<Eigen::VectorXd>& y,
    Eigen::VectorXd& beta,
    double& sigma2_b,
    double& sigma2_e,
    double& pi,
    double b = 1.0,
    double c = 1.0,
    double S2_b = -1.0,
    double S2_e = -1.0,
    double nu_b = 3.0,
    double nu_e = 3.0
) {
  int n = X.rows();
  int p = X.cols();
  
  // handle defaults
  if (S2_b < 0) {
    double y_mean = y.mean();
    S2_b = (y.array() - y_mean).square().sum() / (n - 1) * 0.1;
  }
  if (S2_e < 0) {
    double y_mean = y.mean();
    S2_e = (y.array() - y_mean).square().sum() / (n - 1);
  }
  
  // X * beta and predicted vals
  Eigen::MatrixXd X_beta(n, p);
  for (int j = 0; j < p; j++) {
    X_beta.col(j) = beta(j) * X.col(j);
  }
  Eigen::VectorXd predicted = X_beta.rowwise().sum();
  
  int k = 0;  
  int n_nan = 0;
  
  // update beta
  for (int j = 0; j < p; j++) {
    double beta_prev = beta(j);
    Eigen::VectorXd x_j = X.col(j);
    double x_j2 = x_j.squaredNorm();
    
    Eigen::VectorXd X_minus_j_beta = predicted - X_beta.col(j);
    
    double r_j = x_j.dot(y - X_minus_j_beta);
    
    double beta_hat = r_j / x_j2;
    double sigma2_0 = sigma2_e / x_j2;
    double sigma2_1 = sigma2_b + sigma2_0;
    
    // calculate non-zero (log)probs 
    double p_0 = log(pi) + logpdf_norm(beta_hat, 0.0, sqrt(sigma2_0));
    double p_1 = log(1.0 - pi) + logpdf_norm(beta_hat, 0.0, sqrt(sigma2_1));
    
    // logsumexp
    double m = std::max(p_0, p_1);
    double prob = exp(p_1 - m) / (exp(p_0 - m) + exp(p_1 - m));
    
    prob = std::max(1e-12, std::min(prob, 1.0 - 1e-12));
    
    if (std::isnan(prob)) {
      prob = 0.0;
      n_nan++;
    }
    
    bool delta = rbernoulli(prob);
    
    if (delta) {
      // draw non-zero coeffs
      k++;
      double mu_j = (sigma2_b / sigma2_e) * r_j / (1.0 + sigma2_b / sigma2_0);
      double sigma2_j = sigma2_b * sigma2_e / (sigma2_e + sigma2_b * x_j2);
      beta(j) = rnorm(mu_j, sqrt(sigma2_j));
    } else {
      beta(j) = 0.0;
    }
    
    // update predicted
    if (beta(j) != beta_prev) {
      X_beta.col(j) = beta(j) * x_j;
      predicted += (X_beta.col(j) - beta_prev * x_j);
    }
  }
  
  // draw var and prob-zero params
  Eigen::VectorXd e = y - predicted;
  double a_b = (nu_b + k) / 2.0;
  double b_b = (nu_b * S2_b + beta.squaredNorm()) / 2.0;
  double a_e = (n + nu_e) / 2.0;
  double b_e = (nu_e * S2_e + e.squaredNorm()) / 2.0;
  
  sigma2_b = rinvgamma(a_b, b_b);
  sigma2_e = rinvgamma(a_e, b_e);
  pi = rbeta(b + (p - k), c + k);
  
  if (n_nan > 0) {
    Rcpp::warning("Warning: " + std::to_string(n_nan) + 
                  " numerical errors during SNP inclusion probability calculations.");
  }
  
  return List::create(
    Named("beta") = beta,
    Named("sigma2_b") = sigma2_b,
    Named("sigma2_e") = sigma2_e,
    Named("pi") = pi
  );
}

// [[Rcpp::export]]
List Bayes_B_Gibbs_run(
    const Eigen::Map<Eigen::MatrixXd>& X,
    const Eigen::Map<Eigen::VectorXd>& y,
    int n_iter,
    double b = 1.0,
    double c = 1.0,
    double S2_b = -1.0,
    double S2_e = -1.0,
    double nu_b = 3.0,
    double nu_e = 3.0
) {
  int n = X.rows();
  int p = X.cols();
  
  // handle defaults
  if (S2_b < 0) {
    double y_mean = y.mean();
    S2_b = (y.array() - y_mean).square().sum() / (n - 1) * 0.1;
  }
  if (S2_e < 0) {
    double y_mean = y.mean();
    S2_e = (y.array() - y_mean).square().sum() / (n - 1);
  }
  
  // init store
  Eigen::MatrixXd beta_res(p, n_iter);
  Eigen::VectorXd sigma2_b_res(n_iter);
  Eigen::VectorXd sigma2_e_res(n_iter);
  Eigen::VectorXd pi_res(n_iter);
  
  // init pars
  Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
  double sigma2_b = rinvgamma(nu_b, nu_b * S2_b);
  double sigma2_e = rinvgamma(nu_e, nu_e * S2_e);
  double pi = rbeta(b, c);
  
  // run
  for (int iter = 0; iter < n_iter; iter++) {
    List result = Bayes_B_Gibbs_step(X, y, beta, sigma2_b, sigma2_e, pi,
                                      b, c, S2_b, S2_e, nu_b, nu_e);
    
    beta = as<Eigen::VectorXd>(result["beta"]);
    sigma2_b = result["sigma2_b"];
    sigma2_e = result["sigma2_e"];
    pi = result["pi"];
    
    beta_res.col(iter) = beta;
    sigma2_b_res(iter) = sigma2_b;
    sigma2_e_res(iter) = sigma2_e;
    pi_res(iter) = pi;
  }
  
  return List::create(
    Named("beta") = beta_res,
    Named("sigma2_b") = sigma2_b_res,
    Named("sigma2_e") = sigma2_e_res,
    Named("pi") = pi_res
  );
}