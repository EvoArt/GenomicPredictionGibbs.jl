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
  std::gamma_distribution<> d(shape, 1.0/scale);
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
