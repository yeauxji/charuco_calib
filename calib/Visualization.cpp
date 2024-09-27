#include "Visualization.h"

cv::Point _topLeftCorner;
cv::Point _bottomRightCorner;
int _stage = 0;

void onClickDrawRectangle(int event, int x, int y, int flag, void* userdata)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
    {
        if (_stage == 0) {
            _topLeftCorner = cv::Point(x, y);
            _stage++;
        }
        else if (_stage == 1) {
            _bottomRightCorner = cv::Point(x, y);
            _stage++;

            cv::Mat img = *(cv::Mat*)userdata;
            cv::Rect bbox(_topLeftCorner, _bottomRightCorner);
            cv::rectangle(img, bbox, cv::Scalar(0, 255, 0), 3);
        }
    }
    }
}
