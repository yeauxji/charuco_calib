#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "Board.h"
#include "Camera.h"
#include "BaseType.h"
#include "CameraGraph.h"
#include "Calibration.h"

/**
 * Optimize all the extrinsic matrice to correct the numerical error.
 *
 * \param graph the camera graph to be optimized, each camera has a
 * transformation matrix to the reference camera
 */
class RigExtrinsicOptimizer
{
public:
	RigExtrinsicOptimizer(CameraGraph& cam_graph, const ExtrinsicOptimizerOptions& opt_options, CameraType type = CameraType::Pinhole);

	double optimize(std::vector<Camera>& cameras, std::vector<Camera>& rig_cameras);

private:
	Sp<CameraGraph> graph;
	ExtrinsicOptimizerOptions options;
	CameraType cam_type;

	double solveNonLinearOpimization(cv::Mat& JtJ_inv, cv::Mat& JtE, std::vector<Camera>& rig_cameras);

	double computeProjectError(cv::Mat& params, std::vector<Camera>& rig_cameras);

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




