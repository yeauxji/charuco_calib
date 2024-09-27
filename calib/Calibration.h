#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "Board.h"
#include "Camera.h"
#include "BaseType.h"
#include "CameraGraph.h"


struct CalibrationOptions
{
	// corner refine can help for finding more corners
	cv::aruco::CornerRefineMethod corner_refine = cv::aruco::CORNER_REFINE_NONE;

};

/**
 * Detect pattern features from a image.
 *
 * \param cam the source camera of the input image
 * \param img input image to be detected
 * \param img_id numerical id of the input image
 * \param boards a collection of all calibration boards and the
 * related pattern dictionaries
 * \param calib_params a pointer to aruco detector parameters
 * \param debug_mode save the img with detected features if true
 */
void detectPatterns(Camera& cam, const cv::Mat& img, const size_t img_id, const ChArUcoBoardBundle& boards,
	cv::Ptr<cv::aruco::DetectorParameters> = cv::aruco::DetectorParameters::create(), const bool debug_mode = false, const std::string output_path = ".");


/**
 * Computre the intrinsic matrix, distortion coefficients, and
 * rotations/translations matrices by the collected features.
 *
 * \param cam a camera with feature data
 * \param boards all calibration boards
 * \param calibratio_flag charuco calibration mode flag
 * \return
 */
bool calibrateIntrinsic(Camera& cam, const ChArUcoBoardBundle& boards, int calibratio_flag = 0);

bool solveCameraRigExtrinsic(Camera& cam, const ChArUcoBoardBundle& boards);


struct ExtrinsicOptimizerOptions
{
	bool with_intrinsic = false;
	int max_iter = 200;
	double stop_threshold = 2e-6;
	double init_step_size = 1e-7;
	double step_improvement_multiplier = 0.1;
	double alpha_smooth = 0.01;
};


/**
 * Optimize all the extrinsic matrice to correct the numerical error.
 * 
 * \param graph the camera graph to be optimized, each camera has a 
 * transformation matrix to the reference camera
 */
class ExtrinsicOptimizer
{
public:
	ExtrinsicOptimizer(CameraGraph& cam_graph, const ExtrinsicOptimizerOptions& opt_options, CameraType type = CameraType::Pinhole);

	double optimize(std::vector<Camera>& cameras);

private:
	Sp<CameraGraph> graph;
	ExtrinsicOptimizerOptions options;
	CameraType cam_type;

	double solveNonLinearOpimization(cv::Mat& JtJ_inv, cv::Mat& JtE);

	double computeProjectError(cv::Mat& params);

	bool computePhotoCameraJacobian(const cv::Mat& rvec_feat, const cv::Mat& tvec_feat, const cv::Mat& rvec_cam, const cv::Mat& tvec_cam,
		cv::Mat& rvec_trans, cv::Mat& tvec_trans, const cv::Mat& obj_points, const cv::Mat& img_points, const cv::Mat& intrinsic,
		const cv::Mat& distortion, const double xi, cv::Mat& J_feat, cv::Mat& J_cam, cv::Mat& error, const bool is_omni = false);

	// Set an unreachable large number as the start value of 
	// the error for the optimization.
	const double error_max = 1e4;

	// The camera parameters matrix (vector)
	cv::Mat X;
	// step size for the Jacobian matrix
	double step_size;

};




