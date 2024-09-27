#include "Camera.h"

using namespace cv;
using namespace std;

size_t searchCameraByName(const vector<Camera>& cameras, const string& camera_name)
{
	for (size_t i = 0; i < cameras.size(); ++i) {
		if (cameras[i].name == camera_name)
			return i;
	}

	return -1;
}

void Features::clear()
{
	image_id = -1;
	feature_ids.clear();
	image_features.clear();
	rotation.release();
	translation.release();
}


void rotTransToTransform(const Mat& rotation, const Mat& translation, Mat& transform, int data_type)
{
	transform = Mat::eye(4, 4, data_type);
	Mat R, T;
	Rodrigues(rotation, R);
	T = translation.reshape(1, 3);
	R.convertTo(R, data_type);
	T.convertTo(T, data_type);

	R.copyTo(transform.rowRange(0, 3).colRange(0, 3));
	T.copyTo(transform.rowRange(0, 3).col(3));
}

void transformToRotTrans(const Mat& transform, Mat& rotation, Mat& translation, int data_type)
{
	rotation.release();
	translation.release();
	Rodrigues(transform.rowRange(0, 3).colRange(0, 3), rotation);
	transform.rowRange(0, 3).col(3).copyTo(translation);
	rotation.convertTo(rotation, data_type);
	translation.convertTo(translation, data_type);
}

