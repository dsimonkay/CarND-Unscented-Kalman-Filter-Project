#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.275;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  /* Completing the initialization. */
  is_initialized_ = false;
  time_us_ = 0;

  // number of things
  n_x_ = 5;
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;
  n_sigma_ = 2 * n_aug_ + 1;

  // initializing (helper) vectors and matrices
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);
  Xsig_pred_.fill(0.0);

  x_.fill(0.0);
  P_.fill(0.0);

  // initializing weights
  weights_ = VectorXd(n_sigma_);
  weights_.fill(0.0);

  const double w_0 = lambda_ / (lambda_ + n_aug_);
  const double w_i = 0.5 / (lambda_ + n_aug_);

  weights_(0) = w_0;
  for ( int i = 1; i < n_sigma_; i++) {
    weights_(i) = w_i;
  }

  // controlling debug mode
  debug_mode_ = false;
}


UKF::~UKF() {}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_) {
    // ...then initialize the filter (which shouldn't be allowed in case the measurement originates from
    // a sensor type that's currently disabled, but... well, whatever. :-( )
    Init(meas_package);
    return;
  }

  // do we have to consider this measurement package at all?
  if ( meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_ ||
       meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_ ) {
    Log(" *** Skipping data package as the measurements from this sensor type will be ignored in this run.");
    return;
  }

  // computing the time elapsed between the current and previous measurements (in seconds)
  const double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // prediction step
  Prediction(dt);

  // measurement update step
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);

  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
}


/**
 * Initializing the Kalman Filter properly.
 * @param meas_package The latest measurement data of either radar or laser
 */
void UKF::Init(MeasurementPackage meas_package) {

  // first measurement
  Log("[UKF] initializing...", false);

  // state vector elements. "assuming" a kind of hibernated object
  double px, py;
  double v = 0, yaw = 0, yaw_d = 0.1;

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

    // extracting values
    float ro = meas_package.raw_measurements_[0];
    yaw = meas_package.raw_measurements_[1];

    // Converting radar from polar to cartesian coordinates
    px = cos(yaw) * ro;
    py = sin(yaw) * ro;

  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

    // initializing the state with the given position and zero velocity
    px = meas_package.raw_measurements_[0];
    py = meas_package.raw_measurements_[1];
  }

  // state vector initialization
  x_ << px, py, v, yaw, yaw_d;

  // we don't pretend to be pretty sure about the world out there
  P_ << 0.8, 0, 0, 0, 0,
        0, 0.8, 0, 0, 0,
        0, 0, 0.8, 0, 0,
        0, 0, 0, 0.8, 0,
        0, 0, 0, 0, 0.8;

  // storing the timestamp, too
  time_us_ = meas_package.timestamp_;

  // initialization is done
  is_initialized_ = true;

  Log(" done.");
}



/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // defining matrix for holding the augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);
  Xsig_aug.fill(0.0);

  // 1: generate sigma points
  GenerateSigmaPoints(Xsig_aug);

  // 2: predict sigma points
  PredictSigmaPoints(delta_t, Xsig_aug);

  // 3: predict mean and covariance
  PredictMeanAndCovariance();
}


/**
 * First operation in the prediction step: generating sigma points
 */
void UKF::GenerateSigmaPoints(MatrixXd &Xsig_aug) {

  // augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  // initializing augmented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_,   n_x_)   = std_a_ * std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_;

  // creating square root matrix
  const MatrixXd A = P_aug.llt().matrixL();

  // copying x_aug to the first column  
  Xsig_aug.col(0) = x_aug;

  // precomputing the commonly used matrix
  const MatrixXd A_helper = sqrt(n_aug_ + lambda_) * A;

  // performing column copying
  for( int i = 0; i < n_aug_; ++i ) {

    Xsig_aug.col(1+i) = x_aug + A_helper.col(i);
    Xsig_aug.col(1+i+n_aug_) = x_aug - A_helper.col(i);
  }
}


/**
 * Second operation in the prediction step: predicting sigma points
 */
void UKF::PredictSigmaPoints(double delta_t, const MatrixXd &Xsig_aug) {

  VectorXd x_aug = VectorXd(n_aug_);
  VectorXd x_k;
  VectorXd model;
  VectorXd noise;

  // helper variables for readability
  double psi, psi_d, v;
  double nu_a, nu_psi_dd;
  double cos_psi;
  double sin_psi;
  double c1, c2;
  const double c3 = delta_t * delta_t / 2;

  for( int i = 0;  i < n_sigma_; i++ ) {

    // extracting the augmented state vector
    x_aug = Xsig_aug.col(i);

    // this is just for even more readability
    x_k = x_aug.head(n_x_);

    // initializing the helper variables (or extracting state vector variables)
    v = x_aug(2);
    psi = x_aug(3);
    psi_d = x_aug(4);
    nu_a = x_aug(5);
    nu_psi_dd = x_aug(6);

    // precomputing trigonometrical values
    sin_psi = sin(psi);
    cos_psi = cos(psi);

    // the noise will be the same in either case
    noise = VectorXd(n_x_);
    noise <<  c3 * cos_psi * nu_a,
              c3 * sin_psi * nu_a,
              delta_t * nu_a,
              c3 * nu_psi_dd,
              delta_t * nu_psi_dd;

    // ...but not the model
    model = VectorXd(n_x_);

    if ( abs(psi_d) < 0.0001 ) {
      // handling the case where psi_d is near-zero
      model <<  v * cos_psi * delta_t,
                v * sin_psi * delta_t,
                0,
                0,
                0;
    } else {
      // 'normal' case
      c1 = v / psi_d;
      c2 = psi + psi_d * delta_t;

      model <<  c1 * (sin(c2) - sin_psi),
                c1 * (-cos(c2) + cos_psi),
                0,
                psi_d * delta_t,
                0;
    }

    Xsig_pred_.col(i) = x_k + model + noise;
  }
}


/**
 * Third operation in the prediction step: predicting state mean and covariance matrix
 */
void UKF::PredictMeanAndCovariance() {

  // helper variable
  VectorXd x_diff = VectorXd(n_x_);

  // initializing
  P_.fill(0.0);

  // predicting state mean
  x_.fill(0.0);
  for( int i = 0;  i < n_sigma_; i++ ) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  for ( int i = 0;  i < n_sigma_; i++ ) {

    // state difference with normalized angle
    x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}




/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // measurement dimension. lidar can measure px and py
  const int n_z = 2;

  VectorXd z_pred = VectorXd(n_z);              // predicted measurement
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_);      // sigma points transformed to measurement space
  MatrixXd S = MatrixXd(n_z, n_z);              // measurement covariance matrix

  // initializing in/out variables
  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);

  // 1: measurement prediction step
  PredictLidarMeasurement(z_pred, Zsig, S);

  // 2: update step
  UpdateState(meas_package, z_pred, Zsig, S);

  // +1: calculating NIS
  CalculateNIS(meas_package, z_pred, S);
}



/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // measurement dimension. radar can measure r, phi, and r_dot
  const int n_z = 3;

  VectorXd z_pred = VectorXd(n_z);              // predicted measurement
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_);      // sigma points transformed to measurement space
  MatrixXd S = MatrixXd(n_z, n_z);              // measurement covariance matrix

  // initializing in/out variables
  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);

  // 1: measurement prediction step
  PredictRadarMeasurement(z_pred, Zsig, S);

  // 2: update step
  UpdateState(meas_package, z_pred, Zsig, S);

  // +1: calculating NIS
  CalculateNIS(meas_package, z_pred, S);
}


/**
 * First operation in the measurement update step for a radar measurement:
 *  - transforming sigma points into measurement space
 *  - calculating mean predicted measurement
 *  - calculate covariance matrix S
 */
void UKF::PredictRadarMeasurement(VectorXd &z_pred, MatrixXd &Zsig, MatrixXd &S) {

  // measurement dimension. radar can measure r, phi, and r_dot
  const int n_z = 3;

  // defining helper variables
  double px, py, v, psi, r, phi, r_d;
  VectorXd radar_measurement = VectorXd(n_z);
  VectorXd z_diff = VectorXd(n_z);

  // transforming sigma points into measurement space
  for( int i = 0;  i < n_sigma_; i++ ) {

    // extracting values
    px = Xsig_pred_(0, i);
    py = Xsig_pred_(1, i);
    v = Xsig_pred_(2, i);
    psi = Xsig_pred_(3, i);

    r = sqrt(px*px + py*py);
    phi = atan2(py, px);
    r_d = v * (px*cos(psi) + py*sin(psi)) / r;

    radar_measurement << r, phi, r_d;

    // filling Zsig and z_pred in the same loop
    Zsig.col(i) = radar_measurement;
    z_pred += weights_(i) * radar_measurement;
  }


  // calculating covariance matrix S
  for ( int i = 0;  i < n_sigma_; i++ ) {

    // calculating residual with angle normalization
    z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // measurement noise covariance matrix S
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;

  // adding measurement noise to the covariance matrix
  S += R;
}


/**
 * First operation in the measurement update step for a lidar measurement:
 *  - transforming sigma points into measurement space
 *  - calculating mean predicted measurement
 *  - calculate covariance matrix S
 */
void UKF::PredictLidarMeasurement(VectorXd &z_pred, MatrixXd &Zsig, MatrixXd &S) {

  // measurement dimension. lidar can measure px and py
  const int n_z = 2;

  // defining helper variables
  double px, py;
  VectorXd lidar_measurement = VectorXd(n_z);
  VectorXd z_diff = VectorXd(n_z);

  // transforming sigma points into measurement space
  for( int i = 0;  i < n_sigma_; i++ ) {

    // extracting values
    px = Xsig_pred_(0, i);
    py = Xsig_pred_(1, i);

    lidar_measurement << px, py;

    // filling Zsig and z_pred in the same loop
    Zsig.col(i) = lidar_measurement;
    z_pred += weights_(i) * lidar_measurement;
  }


  // calculating covariance matrix S
  for ( int i = 0;  i < n_sigma_; i++ ) {

    // calculating residual
    z_diff = Zsig.col(i) - z_pred;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // measurement noise covariance matrix S
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;

  // adding measurement noise to the covariance matrix
  S += R;
}


/**
 * Second operation in the measurement update step for a radar or lidar measurement:
 *  - calculating cross correlation matrix
 *  - calculating Kalman gain K;
 *  - updating state mean and covariance matrix
 */
void UKF::UpdateState(const MeasurementPackage &measurement, const VectorXd &z_pred, const MatrixXd &Zsig, const MatrixXd &S) {

  const bool is_radar_measurement = measurement.sensor_type_ == MeasurementPackage::RADAR;
  const int n_z = is_radar_measurement ? 3 : 2;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  // common helper variables
  VectorXd x_diff;
  VectorXd z_diff;

  // calculating cross correlation matrix
  for( int i = 0;  i < n_sigma_; i++ ) {

    // state difference (with angle normalization for radar measurements)
    x_diff = Xsig_pred_.col(i) - x_;
    if ( is_radar_measurement ) {
      NormalizeAngle(x_diff(3));
    }

    // residual (with angle normalization for radar measurements)
    z_diff = Zsig.col(i) - z_pred;
    if ( is_radar_measurement ) {
      NormalizeAngle(z_diff(1));
    }

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculating Kalman gain
  MatrixXd K = Tc * S.inverse();

  // residual (with angle normalization for radar measurements)
  z_diff = measurement.raw_measurements_ - z_pred;
  if ( is_radar_measurement ) {
    NormalizeAngle(z_diff(1));
  }

  // updating state mean and covariance matrix
  x_ += K * z_diff;
  P_ += -K * S * K.transpose();
}


/**
 * Normalizing an angle into [-pi, pi].
 * @param {double} angle
 */
void UKF::NormalizeAngle(double &angle) {

    while (angle >  M_PI) { angle -= 2.0 * M_PI; }
    while (angle < -M_PI) { angle += 2.0 * M_PI; }
}



/**
 * A helper method to calculate NIS (and eventually dump it into a corresponding file as well).
 */
void UKF::CalculateNIS(const MeasurementPackage &measurement, const VectorXd &z_pred, const MatrixXd &S) {

  // actual calculation
  VectorXd residual = measurement.raw_measurements_ - z_pred;
  float e = residual.transpose() * S.inverse() * residual;

  // putting the result into a corresponding file + logging
  if ( debug_mode_ ) {

    const std::string sensor_type = ( measurement.sensor_type_ == MeasurementPackage::RADAR ? "Radar" : "Lidar" );

    // helper variables for string manipulation
    char buffer[10];
    std::stringstream content;
    std::stringstream filename;

    // assembling filename. It will look something like this: "../data/Lidar_NIS_values_2.1_0.4.txt"
    filename << nis_filename_prefix_ << sensor_type << "_NIS_values_";
    sprintf(buffer, "%.1f", std_a_);
    filename << std::string(buffer) << "_";
    sprintf(buffer, "%.2f", std_yawdd_);
    filename << std::string(buffer) << nis_filename_suffix_ << ".txt";

    // preparing content
    content << e;

    // dumping the NIS value into the file
    ofstream nis_dump;
    nis_dump.open(filename.str().c_str(), ios_base::out | ios_base::app);
    nis_dump << content.str() << endl;
    nis_dump.close();

    // displaying info in the console
    std::stringstream message;
    message << sensor_type << " NIS: " << e;
    Log(message.str());
  }
}



/**
 * A helper method to display a log/debug message.
 */
void UKF::Log(std::string message, bool print_newline) {

  if ( debug_mode_ ) {

    // hello, Kitty
    std::cout << message;

    if ( print_newline ) {
      std::cout << endl;
    }
  }
}
