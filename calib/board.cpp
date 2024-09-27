#include "Board.h"

#include <iostream>

using std::vector;
using namespace cv;

namespace {

    /**
     * Spilt a pre-defined ArUco pattern dictionary into smaller ones 
     * according to the offset size.
     * 
     * \param dict the dictionary to be splitted
     * \param offset the offset size of the splitted dicts
     * \return an array of all splitted dictionaries with smaller sizes
     */
    vector<ArUcoDictionaryPtr> splitDictionary(ArUcoDictionaryPtr dict, int offset, int board_shift, int max_size = 1024) {
        vector<ArUcoDictionaryPtr> splits;
        int count = 0;
 
        while (count < max_size && dict->bytesList.rows >= offset) {
            if (count % 2 && board_shift )
            {
                Mat splitBytesList;
                Rect current(0, 0, dict->bytesList.cols, offset + 1);
                Rect remaining(0, offset + 1, dict->bytesList.cols, dict->bytesList.rows - (offset + 1));
                dict->bytesList(current).copyTo(splitBytesList);
                dict->bytesList(remaining).copyTo(dict->bytesList);

                splits.push_back(std::make_shared<aruco::Dictionary>(splitBytesList, dict->markerSize, dict->maxCorrectionBits));
                ++count;
            }
            else
            {
                Mat splitBytesList;
                Rect current(0, 0, dict->bytesList.cols, offset);
                Rect remaining(0, offset, dict->bytesList.cols, dict->bytesList.rows - offset);
                dict->bytesList(current).copyTo(splitBytesList);
                dict->bytesList(remaining).copyTo(dict->bytesList);

                splits.push_back(std::make_shared<aruco::Dictionary>(splitBytesList, dict->markerSize, dict->maxCorrectionBits));
                ++count;
            }


            
        }

        return splits;
    }
}

/**
 * Constructor with charuco parameters.
 * 
 * \param board_params 
 */
ChArUcoBoardBundle::ChArUcoBoardBundle(const ChArUcoParams& _params) :
    params(_params)
{
    ArUcoDictionaryPtr pred = aruco::getPredefinedDictionary(params.predifined_dict);
    int board_shift = params.board_shift;
    dicts = splitDictionary(pred, params.split_offset, board_shift, params.max_board_num);

    for (ArUcoDictionaryPtr dict : dicts) {
        auto board = aruco::CharucoBoard::create(params.cols, params.rows, params.square_size, params.marker_size, dict);
        boards.push_back(board);
    }

    bundle_size = dicts.size();
}


/**
 * Draw all boards into opencv images.
 * 
 * \return 
 */
vector<Mat> ChArUcoBoardBundle::drawAllBoards()
{
    vector<Mat> board_images;

    int i = 0;
    for (ChArUcoBoardPtr board : boards) {
        Mat img;
        auto img_size = Size(params.cols * params.square_size_pixel, params.rows * params.square_size_pixel);
        board->draw(img_size, img, params.margin_width_pixel, params.marker_border_bit_size);
        //cv::imwrite(std::to_string(params.rows) + "x" + std::to_string(params.cols)  + "_" + std::to_string(i) + ".png", img);
        board_images.push_back(std::move(img));
        
        i++;
    }
    
    return board_images;
}
