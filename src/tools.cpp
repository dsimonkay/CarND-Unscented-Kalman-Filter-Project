#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // checking the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  // ... your code here
  if (estimations.size() == 0) {
    cout << "[Tools::CalculateRMSE] Error: empty input received." << endl;
    return rmse;

  } else if ( estimations.size() != ground_truth.size() ) {
    cout << "[Tools::CalculateRMSE] Error: estimation vector size doesn't match that of the ground truth vector." << endl;
    return rmse;
  }

  VectorXd residual(4);
  VectorXd residual_sq(4);

  // accumulating squared residuals
  for (int i = 0; i < estimations.size(); ++i) {

    residual = estimations[i] - ground_truth[i];
    residual_sq = residual.array() * residual.array();
    rmse += residual_sq;
  }

  // calculating the mean
  rmse = rmse / estimations.size();

  // calculating the squared root
  rmse = rmse.array().sqrt();

  // returning the result
  return rmse;
}
