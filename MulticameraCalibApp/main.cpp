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

void demosaicImage(cv::Mat &img, cv::ColorConversionCodes conv = cv::COLOR_BayerBG2BGR) {
    cv::demosaicing(img, img, conv);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
}

void executeSBA(std::string input_cam_filename, std::string input_feature_filename,
    std::string output_cam_filename, int fixed_intrinsic, int fixed_dist, bool print_info = false) {
    // Defined in the project property pages
    // Configuration Properties -> C/C++ -> Preprocessor -> Preprocessor Definitions
    // SBA_EXECUTABLE=R"($(ProjectDir)..\bin\sba.exe)"
    std::string sba_executable = SBA_EXECUTABLE;
    //std::string sba_executable = "sba.exe";

    char cmd[1024];

    sprintf_s(cmd, "%s %s %s %s %d %d", sba_executable.c_str(),
        input_cam_filename.c_str(), input_feature_filename.c_str(), output_cam_filename.c_str(),
        fixed_intrinsic, fixed_dist);

    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    if (print_info)
        std::cout << result << std::endl;
}

void printChArUcoBoards(std::string save_path, ChArUcoBoardBundle& boards) {
    auto board_images = boards.drawAllBoards();

    for (int i = 0; i < board_images.size(); ++i) {
        std::string save_filepath = save_path + "/" + std::to_string(i) + ".png";
        cv::imwrite(save_filepath, board_images[i]);
    }
}

void calibFromImagesPath(std::string images_path, std::regex regex_pattern, ChArUcoBoardBundle boards, size_t num_cameras,
    size_t num_frames, bool debug_mode = false, std::string output_path = "", size_t root_camera_idx = 0) {
    clock_t start, end, mid;
    start = clock();

    cv::Ptr<cv::aruco::DetectorParameters> detector_params = cv::aruco::DetectorParameters::create();
    //detector_params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;
    detector_params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    //detector_params->adaptiveThreshWinSizeMax = 73;
    //detector_params->minMarkerPerimeterRate = 0.01;
    //detector_params->minCornerDistanceRate = 0.02;
    //detector_params->cornerRefinementMaxIterations = 100;
    //detector_params->cornerRefinementWinSize = 9;

    if (!fs::is_directory(output_path)) {
        fs::create_directories(output_path);
    }

    // Iterate the image files in images_path, load each image and detect patterns
    std::vector<Camera> cameras;
    cameras.resize(num_cameras);

    omp_set_num_threads(24);

    std::vector<std::string> img_entires;
    for (const auto& entry : fs::directory_iterator(images_path)) {
        img_entires.push_back(entry.path().string());
    }

    std::cout << "Find " << img_entires.size() << " images, start detecting features..." << std::endl;

    std::atomic<int> progress(0);
    int total = img_entires.size(); // Total progress steps
    std::thread t_progress_bar(displayProgressBar, std::ref(progress), total);

#pragma omp parallel for
    for (int i = 0; i < img_entires.size(); ++i) {
        std::string filepath = img_entires[i];

        std::smatch pieces_match;
        std::regex_match(filepath, pieces_match, regex_pattern);

        if (pieces_match.size() != 4) {
            std::cerr << filepath << " is not a vaild image filepath." << std::endl;
            progress++;
            continue;
        }

        size_t img_id = std::stoi(pieces_match[1].str());
        std::string cam_name = pieces_match[2].str();
        //size_t cam_id = searchCameraByName(cameras, cam_name);
        size_t cam_id = std::stoi(cam_name);

        cv::Mat img = cv::imread(filepath,cv::IMREAD_ANYDEPTH);
        int channels = img.channels();
        std::cout << "image is " << channels << " channels" << std::endl;
        if (channels == 1)
        {
            std::cout << "image is mono or raw data" << std::endl;

        }
        else if (channels == 3)
        {
            std::cout << "image is color" << std::endl;
        }

        Camera& cam = cameras[cam_id];

        // initialize
        if (cam.width == 0 && cam.height == 0) {
            cam.width = img.cols;
            cam.height = img.rows;
            cam.name = cam_name;
            cam.type = CameraType::Pinhole;
            cam.all_features.resize(num_frames * boards.bundle_size);
        }

        //auto demosaic_pattern = cv::COLOR_BayerBG2BGR;
        //demosaicImage(img, cv::COLOR_BayerBG2BGR);
        detectPatterns(cam, img, img_id, boards, detector_params, debug_mode, output_path);

        //std::cout << "Extracting features from " << filepath << " finished!" << std::endl;
        progress++;
    }
    t_progress_bar.join();

    std::cout << "Features detection finished, start calibrating cameras..."  << std::endl;
    
    mid = clock();

    // Calibrate intrinsic parameters for each camera
#pragma omp parallel for
    for (int cam_id = 0; cam_id < cameras.size(); ++cam_id) {
        Camera& cam = cameras[cam_id];
        calibrateIntrinsic(cam, boards);
        std::cout << "Compute intrinsic parameters for camera " << cam_id << " finished!" << std::endl;
    }

    std::cout << "Intrinsic calibration all finished, start calibrating extrinsic..." << std::endl;

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
    for (Camera& cam : cameras) {
        std::cout << "Camera " << cam.name << " parameters:" << std::endl;
        std::cout << "# patterns:" << cam.all_features.size() << std::endl;
        std::cout << "reprojection error:" << cam.reproj_error << std::endl;
        std::cout << "intrinsic:" << std::endl << cam.intrinsic << std::endl;
        std::cout << "distortion:" << std::endl << cam.distortion.getDistortionVector() << std::endl;
        std::cout << "extrinsic:" << std::endl << graph.getCameraVertices()[i].transform << std::endl;
        i++;
    }

    std::cout << "Extrinsic calibration finished, start bundle adjustment..." << std::endl;

    executeSBA(cam_param_filename + ".txt", cam_param_filename + "_features.txt", output_path + "//sba_fix2_fix3.txt", 2, 3);
    executeSBA(cam_param_filename + ".txt", cam_param_filename + "_features.txt", output_path + "//sba_fix_all.txt", 5, 5);
    
    // Bundle adjustment
    ExtrinsicOptimizerOptions optimize_options;
    ExtrinsicOptimizer optimizer(graph, optimize_options);
    double err = optimizer.optimize(cameras);

    graph.updateCameraExtrinsics(cameras);
    std::string refined_cam_param_filename = output_path + "//opencv_ba";
    graph.saveCameraParamsToFile(refined_cam_param_filename + ".txt", true, true, err);

    for (Camera& cam : cameras) {
        std::cout << "new extrinsic:" << std::endl << cam.extrinsic << std::endl;
    }

    end = clock();

    printf("Time for reading images: %0.8f sec\n",
        ((float)mid - start) / CLOCKS_PER_SEC);

    printf("Time for calibration: %0.8f sec\n",
        ((float)end - mid) / CLOCKS_PER_SEC);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: run_test.exe [images_path] [output_path] [num_cameras] [num_frames] [root_camera_id] [debug_mode]";
        exit(-1);
    }

    ChArUcoParams board_params;
    // board_param.loadFromFile(board_filepath);
    board_params.predifined_dict = cv::aruco::DICT_4X4_1000;
    board_params.rows = 6;
    board_params.cols = 6;
    int square_number = board_params.rows * board_params.cols;
    if (square_number % 2)
    {
        board_params.split_offset = (square_number-1)/2;
        board_params.board_shift = 1;
    }
    else
    {
        board_params.split_offset = square_number / 2;
        board_params.board_shift = 0;
    }
    
    board_params.marker_size = 7.5;
    board_params.square_size = 10.0;
    board_params.max_board_num = 4;
    ChArUcoBoardBundle boards(board_params);
    auto board_imgs = boards.drawAllBoards();

    //board_params.predifined_dict = cv::aruco::DICT_5X5_1000;
    //board_params.rows = 5;
    //board_params.cols = 5;
    //board_params.split_offset = 12;
    //board_params.marker_size = 16.0;
    //board_params.square_size = 20.0;
    //board_params.max_board_num = 20;
    //ChArUcoBoardBundle boards(board_params);

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    // Format: frameXXXXXX_camYYY.ZZZ
    // XXXX: 6 digits image id
    // YY: 3 digits camera id
    // ZZZ: file suffix
    //std::regex regex_pattern(".*img\.([0-9]{6})\_(cam[0-9]{3})\.(.*)$");
    std::regex regex_pattern(".*frame([0-9]{6})\_cam([0-9]{3})\.(.*)$");
    size_t num_cameras = std::stoi(argv[3]);  // 42
    size_t num_frames = std::stoi(argv[4]);  // 40
    size_t root_camera_idx = std::stoi(argv[5]);  // 32
    bool debug_mode = std::stoi(argv[6]);  // 0 - false, 1 - true;

    // printChArUcoBoards(path, boards);

    calibFromImagesPath(input_path, regex_pattern, boards, num_cameras, num_frames, debug_mode, output_path, root_camera_idx);

    return 0;
}