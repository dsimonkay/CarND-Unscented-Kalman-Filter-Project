#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;



class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;


  // custom extensions to make my life easier

  ///* Number of sigma points
  int n_sigma_;

  ///* Debug mode
  bool debug_mode_;

  ///* NIS files name prefix and suffix
  const std::string nis_filename_prefix_ = "../data/";
  const std::string nis_filename_suffix_ = "_P0.8";


  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix. This is a wrapper function.
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);



private:
  /**
   * Initializing the Kalman Filter properly.
   * @param meas_package The latest measurement data of either radar or laser
   */
  void Init(MeasurementPackage meas_package);

  /**
   * First operation in the prediction step:
   *  - creating augmented mean state
   *  - creating augmented covariance matrix
   *  - creating augmented sigma points
   */
  void GenerateSigmaPoints(MatrixXd &Xsig_aug);

  /**
   * Second operation in the prediction step:
   *  - predicting sigma points
   */
  void PredictSigmaPoints(double delta_t, const MatrixXd &Xsig_aug);

  /**
   * Third operation in the prediction step:
   *  - predicting state mean
   *  - predicting covariance matrix
   */
  void PredictMeanAndCovariance();

  /**
   * First operation in the measurement update step for a radar measurement:
   *  - transforming sigma points into measurement space
   *  - calculating mean predicted measurement
   *  - calculate innovation covariance matrix S
   */
  void PredictRadarMeasurement(VectorXd &z_pred, MatrixXd &Zsig, MatrixXd &S);

  /**
   * First operation in the measurement update step for a laser measurement:
   *  - transforming sigma points into measurement space
   *  - calculating mean predicted measurement
   *  - calculate innovation covariance matrix S
   */
  void PredictLidarMeasurement(VectorXd &z_pred, MatrixXd &Zsig, MatrixXd &S);

  /**
   * Second operation in the measurement update step for a radar or lidar measurement:
   *  - calculating cross correlation matrix
   *  - calculating Kalman gain K;
   *  - updating state mean and covariance matrix
   */
  void UpdateState(const MeasurementPackage &measurement, const VectorXd &z_pred, const MatrixXd &Zsig, const MatrixXd &S);

  /**
   * Normalizes the given angle between [-PI, +PI].
   * @param angle The angle to normalize
   */
  void NormalizeAngle(double &angle);

  /**
   * A helper method to calculate NIS (and eventually dumpt it into a corresponding file as well).
   */
  void CalculateNIS(const MeasurementPackage &measurement, const VectorXd &z_pred, const MatrixXd &S);

  /**
   * A helper method to display a log/debug message.
   */
  void Log(std::string message, bool print_newline = true);

};


#endif /* UKF_H */
