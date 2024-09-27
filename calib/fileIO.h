#pragma once

#include <opencv2/highgui.hpp>

#include "Camera.h"
#include "Board.h"

bool readCameraParameters(std::string filename, cv::Mat& camMatrix, cv::Mat& distCoeffs);

bool saveCameraParams(const std::string& filename, cv::Size imageSize, float aspectRatio, int flags,
    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, double totalAvgErr);