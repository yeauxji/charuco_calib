#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <regex>
#include <atomic>
#include <chrono>
#include <thread>
#include <omp.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Calibration.h"
#include "Board.h"
#include "BaseType.h"
#include "RigCalibration.h"


namespace fs = std::filesystem;

void displayProgressBar(std::atomic<int>& progress, int total) {
	const int barWidth = 70;
	std::cout << "Progress: [";
	std::cout.flush();
	while (progress < total) {
		int pos = barWidth * progress / total;
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos) std::cout << "=";
			else if (i == pos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "] " << int(progress * 100 / total) << "%\r";
		std::cout.flush();
		std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Adjust according to your needs
	}
	std::cout << std::endl;
}

cv::Mat getRotation(const std::vector<double>& Q)
{
	cv::Mat R(3, 3, CV_64F);
	double q0 = Q[0];
	double q1 = Q[1];
	double q2 = Q[2];
	double q3 = Q[3];

	R.at<double>(0, 0) = 1 - 2 * q2 * q2 - 2 * q3 * q3;
	R.at<double>(0, 1) = 2 * q1 * q2 - 2 * q0 * q3;
	R.at<double>(0, 2) = 2 * q1 * q3 + 2 * q0 * q2;

	R.at<double>(1, 0) = 2 * q1 * q2 + 2 * q0 * q3;
	R.at<double>(1, 1) = 1 - 2 * q1 * q1 - 2 * q3 * q3;
	R.at<double>(1, 2) = 2 * q2 * q3 - 2 * q0 * q1;

	R.at<double>(2, 0) = 2 * q1 * q3 - 2 * q0 * q2;
	R.at<double>(2, 1) = 2 * q2 * q3 + 2 * q0 * q1;
	R.at<double>(2, 2) = 1 - 2 * q1 * q1 - 2 * q2 * q2;

	return R;
}

void demosaicImage(cv::Mat& img, cv::ColorConversionCodes conv = cv::COLOR_BayerBG2BGR) {
	cv::demosaicing(img, img, conv);
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
}


std::vector<Camera> loadCameraRigParams(const std::string& calib_filepath) {
	std::vector<Camera> cameras;
	std::ifstream file(calib_filepath);
	if (!file.is_open()) {
		std::cerr << "Error: Could not open file " << calib_filepath << std::endl;
		return cameras;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);

		if (iss.peek() == '#') {
			continue;
		}

		double fu, u0, v0, aspect_ratio, s;
		double k1, k2, p1, p2, k3;
		double quat[4];
		double transl[3];
		int width, height;

		if (!(iss >> fu >> u0 >> v0 >> aspect_ratio >> s >> 
			k1 >> k2 >> p1 >> p2 >> k3 >> 
			quat[0] >> quat[1] >> quat[2] >> quat[3] >> 
			transl[0] >> transl[1] >> transl[2] >> 
			width >> height)) {
			std::cerr << "Error: Could not parse line " << line << std::endl;
			continue;
		}

		Camera camera;
		camera.width = width;
		camera.height = height;

		cv::Mat intrinsic = cv::Mat::eye(3, 3, CV_64F);
		intrinsic.at<double>(0, 0) = fu;
		intrinsic.at<double>(1, 1) = fu / aspect_ratio;
		intrinsic.at<double>(0, 2) = u0;
		intrinsic.at<double>(1, 2) = v0;
		intrinsic.at<double>(0, 1) = s;
		camera.intrinsic = intrinsic;

		cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
		distortion.at<double>(0, 0) = k1;
		distortion.at<double>(0, 1) = k2;
		distortion.at<double>(0, 2) = p1;
		distortion.at<double>(0, 3) = p2;
		distortion.at<double>(0, 4) = k3;
		camera.distortion = CameraDistortion();
		camera.distortion.setDistortion(distortion);

		cv::Mat rotation = getRotation(std::vector<double>(quat, quat + 4));
		cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
		translation.at<double>(0, 0) = transl[0];
		translation.at<double>(1, 0) = transl[1];
		translation.at<double>(2, 0) = transl[2];
		cv::Mat transform = cv::Mat::eye(4, 4, CV_64F);
		//std::cout << "Rotation: " << rotation << std::endl;
		//std::cout << "Translation: " << translation << std::endl;

		rotation.copyTo(transform.rowRange(0, 3).colRange(0, 3));
		translation.copyTo(transform.rowRange(0, 3).col(3));
		camera.extrinsic = transform;

		cameras.push_back(camera);
	}
	return cameras;
}

size_t getCameraRigRootCameraId(const std::vector<Camera>& cameras) {
	for (size_t i = 0; i < cameras.size(); ++i) {
		if (!cv::norm(cameras[i].extrinsic, cv::Mat::eye(4, 4, CV_64F), cv::NORM_L1)) {
			return i;
		}
	}
	return -1;
}

std::array<int, 2> getRigIdAndCamIdFromFilepath(const std::string& filepath, std::regex regex_pattern) {
	std::smatch pieces_match;
	std::regex_match(filepath, pieces_match, regex_pattern);

	if (pieces_match.size() != 3) {
		std::cout << "Err: pattern not matched!" << std::endl;
		return { -1, -1 };
	}

	int rig_id = std::stoi(pieces_match[1].str());
	int cam_id = std::stoi(pieces_match[2].str());

	std::cout << "rig number is " << rig_id << ", cam number is " << cam_id << "." << std::endl;
	return { rig_id, cam_id };
}

std::vector<Camera> loadCameraRigImages(const std::string& images_path, const std::vector<Camera>& rig_cameras, std::regex regex_pattern, ChArUcoBoardBundle boards, size_t num_rigs, size_t num_cams,
	bool debug_mode = false, bool demosaic_image = true, std::string output_path = "", size_t root_camera_idx = 0, size_t start_cam_id = 0) {

	if (!fs::is_directory(output_path)) {
		fs::create_directories(output_path);
	}

	std::cout << "root cam id:" << root_camera_idx << " start_cam_id: " << start_cam_id << std::endl;

	// Iterate the image files in images_path, load each image and detect patterns
	std::vector<Camera> cameras = std::vector<Camera>(num_rigs * num_cams);

	omp_set_num_threads(24);

	std::vector<std::string> img_entires;
	for (const auto& entry : fs::recursive_directory_iterator(images_path)) {
		img_entires.push_back(entry.path().string());
	}

	std::cout << "Find " << img_entires.size() << " images, start detecting features..." << std::endl;

	std::atomic<int> progress(0);
	int total = img_entires.size(); // Total progress steps
	std::thread t_progress_bar(displayProgressBar, std::ref(progress), total);

#pragma omp parallel for
	for (int i = 0; i < img_entires.size(); ++i) {
		std::string filepath = img_entires[i];
		auto [rig_id, cam_id] = getRigIdAndCamIdFromFilepath(filepath, regex_pattern);
		if (rig_id == -1 || cam_id == -1) {
			//std::cerr << filepath << " is not a vaild image filepath." << std::endl;
			progress++;
			continue;
		}

		if (cam_id < start_cam_id) {
			progress++;
			continue;
		}

		cam_id -= start_cam_id;

		Camera& cam = cameras[rig_id * num_cams + cam_id];
		std::cout << "load image filename: " << filepath << std::endl;
		cv::Mat img = cv::imread(filepath, cv::IMREAD_ANYCOLOR);
		if (demosaic_image) {
			demosaicImage(img, cv::COLOR_BayerBG2BGR);
		}

		// initialize
		if (cam.width == 0 && cam.height == 0) {
			cam.width = img.cols;
			cam.height = img.rows;
			cam.name = "rig_" + std::to_string(rig_id) + "_cam_" + std::to_string(cam_id);
			cam.all_features.resize(boards.bundle_size);
			cam.type = CameraType::Pinhole;
			cam.intrinsic = rig_cameras[cam_id].intrinsic;
			cam.distortion = rig_cameras[cam_id].distortion;
		} 
		else {
			std::cerr << "Camera has already been initialized! Rig id: " << rig_id << ", cam id: " << cam_id << std::endl;
		}

		size_t img_id = 0;
		cv::Ptr<cv::aruco::DetectorParameters> detector_params = cv::aruco::DetectorParameters::create();
		detector_params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;
		//detector_params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
		//detector_params->adaptiveThreshWinSizeMax = 73;
		//detector_params->minMarkerPerimeterRate = 0.01;
		//detector_params->minCornerDistanceRate = 0.02;
		//detector_params->cornerRefinementMaxIterations = 100;
		//detector_params->cornerRefinementWinSize = 9;
		detectPatterns(cam, img, img_id, boards, detector_params, debug_mode, output_path);

		//std::cout << "Extracting features from " << filepath << " finished!" << std::endl;
		progress++;
	}
	t_progress_bar.join();

	std::cout << "Feature extraction finished!" << std::endl;

	for (size_t i = 0; i < cameras.size(); ++i) {
		// calibrate each camera
		solveCameraRigExtrinsic(cameras[i], boards);
	}

	return cameras;
}

void solveCameraRigExtrinsic(std::vector<Camera>& cameras, std::vector<Camera>& rig_cameras, size_t root_camera_idx, 
	const std::string& output_path, size_t rig_ba_max_iter) 
{
	// Build the camera graph
	CameraGraph graph(cameras);
	graph.updateReferrenceCameraID(root_camera_idx);
	graph.calculateTransforms();

	std::string cam_param_filename = output_path + "//no_ba";
	if (cam_param_filename.length() > 0) {
		graph.saveCameraParamsToFile(cam_param_filename + ".txt");
		graph.saveFeaturesToFile(cam_param_filename + "_features.txt");
	}

	// TODO img id must start from 0000
	// Print camera parameters
	int i = 0;
	//for (Camera& cam : cameras) {
	//	std::cout << "Camera " << cam.name << " parameters:" << std::endl;
	//	std::cout << "# patterns:" << cam.all_features.size() << std::endl;
	//	std::cout << "reprojection error:" << cam.reproj_error << std::endl;
	//	std::cout << "intrinsic:" << std::endl << cam.intrinsic << std::endl;
	//	std::cout << "distortion:" << std::endl << cam.distortion.getDistortionVector() << std::endl;
	//	std::cout << "extrinsic:" << std::endl << graph.getCameraVertices()[i].transform << std::endl;
	//	i++;
	//}

	std::cout << "Extrinsic calibration finished, start bundle adjustment..." << std::endl;

	//executeSBA(cam_param_filename + ".txt", cam_param_filename + "_features.txt", output_path + "//sba_fix2_fix3.txt", 2, 3);
	//executeSBA(cam_param_filename + ".txt", cam_param_filename + "_features.txt", output_path + "//sba_fix_all.txt", 5, 5);

	//Bundle adjustment
	ExtrinsicOptimizerOptions optimize_options;
	optimize_options.stop_threshold = 1e-5;
	ExtrinsicOptimizer optimizer(graph, optimize_options);
	double err = optimizer.optimize(cameras);
	graph.updateCameraExtrinsics(cameras);


	std::string refined_cam_param_filename = output_path + "//opencv_ba";
	graph.saveCameraParamsToFile(refined_cam_param_filename + ".txt", true, true, err);
	graph.saveFeaturesToFile(refined_cam_param_filename + "_features.txt");

	//for (Camera& cam : cameras) {
	//	std::cout << "new extrinsic:" << std::endl << cam.extrinsic << std::endl;
	//}

	//Rig Bundle adjustment
	std::cout << "Start rig bundle adjustment..." << std::endl;
	optimize_options.max_iter = rig_ba_max_iter;
	RigExtrinsicOptimizer rig_optimizer(graph, optimize_options);
	err = rig_optimizer.optimize(cameras, rig_cameras);

	std::string refined_rig_cam_param_filename = output_path + "//opencv_ba_rig";
	graph.saveCameraParamsToFile(refined_rig_cam_param_filename + ".txt", true, true, err, true, rig_cameras.size());
	graph.saveFeaturesToFile(refined_rig_cam_param_filename + "_features.txt");
}


int main(int argc, char** argv) {
	if (argc < 2) {
		std::cout << "Usage: rig_calib.exe [images_path] [output_path] [rig_calib_filepath] [num_rigs] [start_camera_id] [debug_mode] [demosaic_image] [rig_ba_max_iter]";
		exit(-1);
	}


	// parse parameters
	std::string images_path = argv[1];
	std::string output_path = argv[2];
	std::string calib_filepath = argv[3];
	std::regex regex_pattern(".*rig_([0-9]{3})\\\\cam_([0-9]{3})\\\\frame000003.*$");

	size_t num_rigs = std::stoi(argv[4]);
	size_t start_cam_id = std::stoi(argv[5]);
	bool debug_mode = std::stoi(argv[6]) != 0;
	bool demosaic_images = std::stoi(argv[7]) != 0;
	size_t rig_ba_max_iter = std::stoi(argv[8]);

	///////////////////////////////////////////////////////////////////////////////////////////
	// define calibration board information (no need to change if the charuco board is fixed)
	ChArUcoParams board_params;
	// board_param.loadFromFile(board_filepath);
	board_params.predifined_dict = cv::aruco::DICT_4X4_1000;
	board_params.rows = 6;
	board_params.cols = 6;
	board_params.split_offset = 18;
	board_params.marker_size = 12.0;
	board_params.square_size = 16.0;
	board_params.max_board_num = 25;
	ChArUcoBoardBundle boards(board_params);
	///////////////////////////////////////////////////////////////////////////////////////////




	std::vector<Camera> rig_cameras = loadCameraRigParams(calib_filepath);
	size_t root_camera_idx = getCameraRigRootCameraId(rig_cameras); // root_camera_idx = 0;
	size_t num_cams = rig_cameras.size() - start_cam_id; // start_cam_id = 2, num_cams = 4
	std::cout << "number of cameras:" << num_cams << std::endl;;
	root_camera_idx -= start_cam_id; // root_camera_idx = -2?
	
	// skip the lower cameras (charuco board maybe not visible)
	rig_cameras = std::vector<Camera>(rig_cameras.begin() + start_cam_id, rig_cameras.end());
	std::cout << "Root camera index: " << root_camera_idx << std::endl;
	std::vector<Camera> cameras = loadCameraRigImages(images_path, rig_cameras, regex_pattern, boards, num_rigs, 
		num_cams, debug_mode, demosaic_images, output_path, root_camera_idx, start_cam_id);

	for (auto cam : rig_cameras) {
		std::cout << "Rig cam extrinsic: " << cam.extrinsic << std::endl;
	}

	solveCameraRigExtrinsic(cameras, rig_cameras, root_camera_idx, output_path, rig_ba_max_iter);
}