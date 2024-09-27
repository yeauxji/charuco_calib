#pragma once
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/opencv.hpp"

#include <vector>

using DictTemplate = cv::aruco::PREDEFINED_DICTIONARY_NAME;
using ArUcoDictionaryPtr = cv::Ptr<cv::aruco::Dictionary>;
using ChArUcoBoardPtr = cv::Ptr<cv::aruco::CharucoBoard>;

struct ChArUcoParams {
    // number of rows on the board
    int rows = 9;
    // number of columns on the board
    int cols = 9;
    // size of a checkerboard square in mm
    float square_size = 10.0;
    // size of a checkerboard marker in mm
    float marker_size = 7.5;
    // the opencv code for the aruco dictionary
    DictTemplate predifined_dict = cv::aruco::DICT_4X4_100;
    // to split the predefined arcu dictionary
    int split_offset = 40;
    // 
    int max_board_num = 15;

    ///for drawing boards
    // size of a checkerboard square in pixel
    int square_size_pixel = 800;
    // width of the margin in pixel
    int margin_width_pixel = 0;
    // number of border pixels of a marker, it is a 
    // relative size to the marker, and will be resized
    // accordingly while drawing boards
    int marker_border_bit_size = 1;

    // board odd even shifting
    int board_shift = 0;
};

/**
 * A collection of calibration board
 * 
 * We need to split a pre-defined pattern dictionary to generate
 * multiple calibration boards respect to the splitted dicts. Doing
 * this allows us to capture features of different boards from one
 * image. 
 */
struct ChArUcoBoardBundle
{   
    size_t bundle_size;
    std::vector<ArUcoDictionaryPtr> dicts;
    std::vector<ChArUcoBoardPtr> boards;
    ChArUcoParams params;

    explicit ChArUcoBoardBundle(const ChArUcoParams& board_params);

    std::vector<cv::Mat> drawAllBoards();
};