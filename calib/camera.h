#pragma once

#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>

#include <string>
#include <vector>

#include "BaseType.h"
#include "Board.h"


class Camera;

/**
 * A structure that contains all feature information collected 
 * from one calibration board.
 */
class Features {

public:
	// the related image id, different images with same image 
	// id are collected by different cameras at the same time
	size_t image_id = -1;
	// Pointer to the calibration board
	ChArUcoBoardPtr board_ptr = nullptr;
	// all detected charuco corner indices of the related 
	// calibration board
	std::vector<int> feature_ids;
	// all detected charuco corner 2D coordinates on the image
	std::vector<cv::Vec2f> image_features;
	// all detected charuco corner 3D coordinates in the real space
	std::vector<cv::Vec3d> obj_features;
	// objects project to the image space
	std::vector<cv::Vec2d> reprojected_features;
	// a parameter used for constructing camera graph. The higher
	// the weight, the features are more reliable 
	double weight = 0.0;
	// the rotation vector from the board to the camera
	cv::Mat rotation;
	// the translation vector from the board to the camera
	cv::Mat translation;
	// true if this dataset is capable for extrinsic estimation
	bool is_valid = false;

	void clear();
};

/**
 * An enum class of camera types.
 */
enum class CameraType
{
	Pinhole,
	OmniDirectional,
	Unknown = -1
};

/**
 * An struct to save camera distortion parameters.
 */
struct CameraDistortion
{
	double k_1 = 0.0;
	double k_2 = 0.0;
	double k_3 = 0.0;

	double p_1 = 0.0;
	double p_2 = 0.0;

	// return the 1x5 distortion vector of the conventional 
	// order:
	// k1, k2, p1, p2, k3
	cv::Mat getDistortionVector(int size = 5) const {
		if (size == 5) {
			cv::Mat distortion_vector(1, 5, CV_64F);
			distortion_vector.at<double>(0, 0) = k_1;
			distortion_vector.at<double>(0, 1) = k_2;
			distortion_vector.at<double>(0, 2) = p_1;
			distortion_vector.at<double>(0, 3) = p_2;
			distortion_vector.at<double>(0, 4) = k_3;
			return distortion_vector;
		}
		else if (size == 4) {
			cv::Mat distortion_vector(1, 4, CV_64F);
			distortion_vector.at<double>(0, 0) = k_1;
			distortion_vector.at<double>(0, 1) = k_2;
			distortion_vector.at<double>(0, 2) = p_1;
			distortion_vector.at<double>(0, 3) = p_2;
			return distortion_vector;
		}
	}

	// set the distortion coefficient 
	void setDistortion(const cv::Mat& dist) {
		k_1 = dist.at<double>(0, 0);
		k_2 = dist.at<double>(0, 1);
		p_1 = dist.at<double>(0, 2);
		p_2 = dist.at<double>(0, 3);
		k_3 = dist.at<double>(0, 4);
	}

	void setDistortion(double k1, double k2, double p1, double p2, double k3) {
		k_1 = k1;
		k_2 = k2;
		p_1 = p1;
		p_2 = p2;
		k_3 = k3;
	}
};


class Camera
{
public:
	Camera() = default;

	// camera index
	// int idx = 0;
	// camera name
	std::string name = "";
	// width of images from this camera
	int width = 0;
	// height of images from this camera
	int height = 0;
	// the camera type
	CameraType type = CameraType::Unknown;

	// intrinsic matrix, structed as [[fx, 0, cx],[0,fy,cy],[0,0,1]]
	cv::Mat intrinsic = cv::Mat::eye(3, 3, CV_64F);
	// extrinsic matrix to the referrence camera, structed as [[rot (3x3), trans (1x3)], [0,0,0,1]]
	cv::Mat extrinsic = cv::Mat::eye(4, 4, CV_64F);
	// camera distortion, the order of the vector is [k1, k2, p1, p2, k3]
	CameraDistortion distortion;
	// xi Output parameter xi for CMei's model
	double xi = 0;
	// the projection error of the current intrinsic and distortion values
	double reproj_error = 0.0;

	// collection of calibration features from all boards in all images
	std::vector<Features> all_features; 
};

/**
 * Return the index of a camera in a camera array by its name.
 * 
 * \param cameras a camera array
 * \param camera_name search pattern
 * \return the camera index in the array, -1 if not exist
 */
size_t searchCameraByName(const std::vector<Camera>& cameras, const std::string& camera_name);

/**
 * Reshape the Rodrigues rotation matrix (1x3) and the traslation
 *  matrix (1x3) to a transform matrix (4x4).
 * 
 * \param rotation the rotation matrix (1x3) as input
 * \param translation the translation matrix (1x3) as input
 * \pamam transform the transform matrix (4x4) as output
 */
void rotTransToTransform(const cv::Mat& rotation, const cv::Mat& translation, cv::Mat& transform, int data_type = CV_32F);

/**
 * Reshape the transform matrix (4x4) to the Rodrigues rotation 
 * matrix (1x3) and the traslation matrix (1*3).
 *
 * \pamam transform the transform matrix (4x4) as input
 * \param rotation the rotation matrix (1x3) as output
 * \param translation the translation matrix (1x3) as output
 */
void transformToRotTrans(const cv::Mat& transform, cv::Mat& rotation, cv::Mat& translation, int data_type = CV_32F);

