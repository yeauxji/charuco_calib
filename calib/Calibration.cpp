#define _USE_MATH_DEFINES

#include "Calibration.h"

#include <iostream>
#include <Windows.h>
#include <queue>

#include <opencv2/ccalib/omnidir.hpp>

using namespace std;
using namespace cv;

namespace {
    /**
     * This function generates the theoretical coordinate in real space of
     * a feature corner
     *
     * @param idx the index of a feature corner.
     * @param width number of columns on the ChArUco board, usually using
     * ChArUcoParams.cols
     * @param scale the scale we use to map the ChArUco board features
     * to the real space, usually using ChArUcoParams.square_size
     * @return theoretical real world coordinate of the given feature
     *
     * @deprecated the obj coordinates can be found from board->chessboardCorners
     */
    Vec3d charucoIdToImageCoordinates(int idx, int width, double scale) {
        auto x = idx % (width - 1);
        auto y = idx / (width - 1);
        return Vec3d(scale * x, scale * y, 0.0);
    };

    /**
     * Check if the detected features lie in one line and barely form as
     * distinguishable triangles.
     *
     * @param markerCorners all detected feature corners in one calibration
     * board
     * @return true if the features are degenerate, false elsewise
     */
    bool checkDegenration(vector<Vec2f>& markerCorners) {
        size_t corner_size = markerCorners.size();

        for (size_t i = 0; i < corner_size; i++) {
            for (size_t j = 0; j < corner_size; j++) {
                for (size_t k = 0; k < corner_size; k++) {
                    if (j != i && k != i && k != j) {
                        Vec2f ab = markerCorners[j] - markerCorners[i];
                        Vec2f ac = markerCorners[k] - markerCorners[i];
                        ab /= norm(ab);
                        ac /= norm(ac);

                        double angle = 180.0 * acos(ab[0] * ac[0] + ab[1] * ac[1]) / M_PI;

                        if (std::fabs(angle) > 45 && std::fabs(angle) < 135) {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    /**
     * Filter out the false projections for better calibration quality for features
     * on one board.
     *
     * @param ids an array of feature indices
     * @param image_features an array of feature points
     * @param object_features an output array of projected feature points in 3D
     * @param refined_ids an output array of corrected feature ids
     * @param refined_corners an output array of corrected feature points
     * @param params the ChArUco parameters
     * @param threshold the feature points will be filtered out if the norm of
     * the projection error is greater than the threshold.
     */
    void refineOneFrame(Features& feat, vector<int>& refined_ids,
        vector<Vec2f>& refined_corners, const ChArUcoParams& params, double threshold) {

        const vector<int>& ids = feat.feature_ids;
        const vector<Vec2f>& image_features = feat.image_features;

        if (ids.size() < 9)
            return;

        vector<Vec2d> pattern_corners;

        for (size_t i = 0; i < ids.size(); i++) {
            const Vec3d& pattern_corner = feat.obj_features[i];
            pattern_corners.push_back(Vec2d(pattern_corner[0], pattern_corner[1]));
        }

        Mat H = findHomography(image_features, pattern_corners, RANSAC);
        H.convertTo(H, CV_64F);

        if (H.rows == 0)
            return;

        for (size_t i = 0; i < image_features.size(); i++) {
            Mat projected = H * Mat(Vec3d(image_features[i][0], image_features[i][1], 1.0));
            // calculate the projection error
            Vec2d diff_2d = Vec2d(projected.at<double>(0, 0) / projected.at<double>(2, 0),
                projected.at<double>(1, 0) / projected.at<double>(2, 0)) - pattern_corners[i];

            double nm = cv::norm(diff_2d);

            if (nm < threshold) {
                refined_corners.push_back(image_features[i]);
                refined_ids.push_back(ids[i]);
            }
        }
    }

    /**
     * Filter out the false projections for better calibration quality for all features.
     *
     * \param all_ids an input array of all feature ids, each element is an array of feature
     * indices collected from one board
     * \param all_corners an input array of all feature points on the picture, each element
     * is an array of feature points from one board
     * \param all_object_features an output array that saves all feature points projected to
     * the real space
     * \param refined_all_ids an output array that saves all feature ids after correction
     * \param refined_all_corners an output array that saves all feature pints after correction
     * \param params a referrence to the ChArUco parameters
     */
    vector<size_t> refineFalseProjection(vector<Features>& features, vector2D<int>& refined_all_ids,
        vector2D<Vec2f>& refined_all_corners, const ChArUcoParams& params, double threshold = 0.5) {

        size_t bundle_size = features.size();
        vector<size_t> valid_ids;

        for (size_t i = 0; i < bundle_size; ++i) {
            Features& feat = features[i];

            if (!feat.feature_ids.empty()) {
                vector<int> refined_ids;
                vector<Vec2f> refined_corners;
                vector<Vec3d> object_features;

                refineOneFrame(feat, refined_ids, refined_corners, params, threshold);

                if (refined_ids.size() >= 9 && !checkDegenration(refined_corners)) {
                    refined_all_ids.push_back(refined_ids);
                    refined_all_corners.push_back(refined_corners);
                    valid_ids.push_back(i);
                }
            }
        }

        return valid_ids;
    }

    /**
     * Concatenate the rotation Rodrigues vector (1x3) and the translation
     * vector (1x3) to a transformation matrix (4x4) of the shape:
     * | R T |
     * | 0 1 |
     *
     * \param rot a rotation Rodrigues vector
     * \param trans a translation vector
     * \return
     */
    Matx44d concatenateRotTrans(const Mat& rot, const Mat& trans) {
        assert(rot.rows == 3 && rot.cols == 1);
        assert(trans.rows == 3 && rot.cols == 1);

        Mat transform = Mat::eye(4, 4, CV_64F);
        Mat R, T;
        Rodrigues(rot, R);
        T = trans.reshape(1, 3);
        R.copyTo(transform.rowRange(0, 3).colRange(0, 3));
        T.copyTo(transform.rowRange(0, 3).col(3));

        return Matx44d((double*)transform.ptr());
    }

    void composeMotion(InputArray _om1, InputArray _T1, InputArray _om2, InputArray _T2, Mat& om3, Mat& T3,
        Mat& dom3dom1, Mat& dom3dT1, Mat& dom3dom2, Mat& dom3dT2, Mat& dT3dom1, Mat& dT3dT1, Mat& dT3dom2, Mat& dT3dT2)
    {
        Mat om1, om2, T1, T2;
        _om1.getMat().convertTo(om1, CV_64F);
        _om2.getMat().convertTo(om2, CV_64F);
        _T1.getMat().reshape(1, 3).convertTo(T1, CV_64F);
        _T2.getMat().reshape(1, 3).convertTo(T2, CV_64F);

        // Rotations:
        Mat R1, R2, R3;
        Mat dR1dom1, dR2dom2;
        cv::Rodrigues(om1, R1, dR1dom1);
        cv::Rodrigues(om2, R2, dR2dom2);
        dR1dom1 = dR1dom1.t();
        dR2dom2 = dR2dom2.t();
        R3 = R2 * R1;

        Mat dR3dR2, dR3dR1;
        //dAB(R2, R1, dR3dR2, dR3dR1);
        matMulDeriv(R2, R1, dR3dR2, dR3dR1);
        Mat dom3dR3;
        cv::Rodrigues(R3, om3, dom3dR3);
        dom3dR3 = dom3dR3.t();

        dom3dom1 = dom3dR3 * dR3dR1 * dR1dom1;
        dom3dom2 = dom3dR3 * dR3dR2 * dR2dom2;
        dom3dT1 = Mat::zeros(3, 3, CV_64FC1);
        dom3dT2 = Mat::zeros(3, 3, CV_64FC1);

        //% Translations:
        Mat T3t = R2 * T1;
        Mat dT3tdR2, dT3tdT1;
        //dAB(R2, T1, dT3tdR2, dT3tdT1);
        matMulDeriv(R2, T1, dT3tdR2, dT3tdT1);

        Mat dT3tdom2 = dT3tdR2 * dR2dom2;
        T3 = T3t + T2;
        dT3dT1 = dT3tdT1;
        dT3dT2 = Mat::eye(3, 3, CV_64FC1);
        dT3dom2 = dT3tdom2;
        dT3dom1 = Mat::zeros(3, 3, CV_64FC1);
    }

    void vector2parameters(const Mat& parameters, const size_t num_verts, std::vector<Vec3f>& rvecVertex, std::vector<Vec3f>& tvecVertexs)
    {
        CV_Assert((int)parameters.channels() == 1 && (int)parameters.total() == 6 * (num_verts - 1));
        CV_Assert(parameters.depth() == CV_32F);
        parameters.reshape(1, 1);

        rvecVertex.reserve(0);
        tvecVertexs.resize(0);

        for (int i = 0; i < num_verts - 1; ++i)
        {
            rvecVertex.push_back(Vec3f(parameters.colRange(i * 6, i * 6 + 3)));
            tvecVertexs.push_back(Vec3f(parameters.colRange(i * 6 + 3, i * 6 + 6)));
        }
    }

    std::vector<Vec3f> findRefinedObjectPoints(const ChArUcoBoardPtr& board, const std::vector<int>& refined_ids)
    {
        std::vector<Vec3f> refined_object_points;
        const std::vector<Point3f>& chessboard_corner_map = board->chessboardCorners;
        for (auto fid : refined_ids)
        {
            const Point3f& obj_point = chessboard_corner_map[fid];
            refined_object_points.push_back(Vec3f(obj_point.x, obj_point.y, obj_point.z));
        }
        return refined_object_points;
    }
}

void detectPatterns(Camera& cam, const Mat& img, size_t img_id, const ChArUcoBoardBundle& boards,
    Ptr<aruco::DetectorParameters> calib_params, const bool debug_mode, const std::string output_path) {

    assert(cam.height == img.rows && cam.width == img.cols);

    Mat img_copy;
    if (debug_mode) {
        img.copyTo(img_copy);
    }

    for (size_t i = 0; i < boards.bundle_size; ++i) {
        ArUcoDictionaryPtr dict = boards.dicts[i];
        ChArUcoBoardPtr board = boards.boards[i];

        vector<int> marker_ids;
        vector2D<Point2f> marker_corners;

        // Find all aruco markers in the current dictionary from the input image 
        aruco::detectMarkers(img, dict, marker_corners, marker_ids, calib_params);

        Features feat;
        feat.image_id = img_id;
        feat.board_ptr = board;

        // Use the aruco markers to predict charuco corners.
        // For each marker, its four corners are provided
        if (!marker_ids.empty()) {
            aruco::interpolateCornersCharuco(marker_corners, marker_ids, img, board,
                feat.image_features, feat.feature_ids);

            for (auto fid : feat.feature_ids) {
                Point3f corner = board->chessboardCorners[fid];
                feat.obj_features.push_back(Vec3d((double)corner.x, (double)corner.y, (double)corner.z));
            }
        }

        if (debug_mode) {
            //aruco::drawDetectedMarkers(img_copy, marker_corners, marker_ids);
            aruco::drawDetectedCornersCharuco(img_copy, feat.image_features, feat.feature_ids, cv::Scalar(255, 255, 0));
        }

        cam.all_features[img_id * boards.bundle_size + i] = feat;
    }

    if (debug_mode) {
        string output_filename = output_path + "\\" + "frame" + to_string(img_id) + "_cam" + cam.name + ".png";
        imwrite(output_filename, img_copy);
    }
}


bool calibrateIntrinsic(Camera& cam, const ChArUcoBoardBundle& boards, int calibratio_flag) {
    vector2D<int> all_refined_ids;
    vector2D<Vec2f> all_refined_features;

    // we remove all zero/degenerate features from the collected data for camera calibration
    vector<size_t> valid_ids = refineFalseProjection(cam.all_features, all_refined_ids, all_refined_features, boards.params, 0.5);

    if (valid_ids.size() < 4) {
        cerr << "Not enough corners for calibration" << endl;
        return false;
    }

    Mat intrinsic, dist;
    vector<Mat> rvecs, tvecs;
    Size image_size = Size(cam.width, cam.height);
    int flags = 0;
    cv::TermCriteria critia(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON);

    if (cam.type == CameraType::Pinhole) {
        double repError =
            aruco::calibrateCameraCharuco(all_refined_features, all_refined_ids, boards.boards[0], image_size,
                intrinsic, dist, rvecs, tvecs, calibratio_flag);

        cam.intrinsic = intrinsic;
        cam.distortion.setDistortion(dist);
        cam.reproj_error = repError;

        // Find the correspondence between rvecs/tvecs and the features in camera
        for (size_t i = 0; i < valid_ids.size(); ++i) {
            Features& feat = cam.all_features[valid_ids[i]];
            projectPoints(feat.obj_features, rvecs[i], tvecs[i], intrinsic, dist, feat.reprojected_features);
            feat.rotation = rvecs[i];
            feat.translation = tvecs[i];
            feat.is_valid = true;
        }
    }
    else if (cam.type == CameraType::OmniDirectional) {
        vector2D<Vec3f> all_refined_object_points;
        const std::vector<Point3f>& chessboard_corner_map = boards.boards[0]->chessboardCorners;
        for (auto& refined_ids : all_refined_ids) {
            std::vector<Vec3f> refined_object_points;
            for (auto fid : refined_ids) {
                const Point3f& obj_point = chessboard_corner_map[fid];
                refined_object_points.push_back(Vec3f(obj_point.x, obj_point.y, obj_point.z));
            }
            all_refined_object_points.push_back(refined_object_points);
        }

        vector<int> ids;
        Mat xi_vec;
        double repError = cv::omnidir::calibrate(all_refined_object_points, all_refined_features, image_size, intrinsic, xi_vec, dist, rvecs, tvecs, flags, critia, ids);
        cam.intrinsic = intrinsic;
        cam.distortion.setDistortion(dist);
        cam.reproj_error = repError;
        cam.xi = xi_vec.at<double>(0);

        // Find the correspondence between rvecs/tvecs and the features in camera
        for (size_t i = 0; i < ids.size(); ++i) {
            Features& feat = cam.all_features[valid_ids[ids[i]]];
            omnidir::projectPoints(feat.obj_features, feat.reprojected_features, rvecs[i], tvecs[i], intrinsic, cam.xi, dist);
            feat.rotation = rvecs[i];
            feat.translation = tvecs[i];
            feat.is_valid = true;
        }
    }

    return true;
}


bool solveCameraRigExtrinsic(Camera& cam, const ChArUcoBoardBundle& boards) {
    vector2D<int> all_refined_ids;
    vector2D<Vec2f> all_refined_features;

    // we remove all zero/degenerate features from the collected data for camera calibration
    vector<size_t> valid_ids = refineFalseProjection(cam.all_features, all_refined_ids, all_refined_features, boards.params, 0.5);

    // Find the correspondence between rvecs/tvecs and the features in camera
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        std::vector<int>& refined_ids = all_refined_ids[i];
        std::vector<Vec2f>& refined_corners = all_refined_features[i];
        std::vector<Vec3f> object_points = findRefinedObjectPoints(boards.boards[0], refined_ids);

        Mat rvecs, tvecs;
        solvePnP(object_points, refined_corners, cam.intrinsic, cam.distortion.getDistortionVector(), rvecs, tvecs, false, cv::SOLVEPNP_SQPNP);

        Features& feat = cam.all_features[valid_ids[i]];
        projectPoints(feat.obj_features, rvecs, tvecs, cam.intrinsic, cam.distortion.getDistortionVector(), feat.reprojected_features);
        feat.rotation = rvecs;
        feat.translation = tvecs;
        feat.is_valid = true;
    }

    return true;
}


ExtrinsicOptimizer::ExtrinsicOptimizer(CameraGraph& cam_graph, const ExtrinsicOptimizerOptions& opt_options, CameraType type) :
    graph(make_shared<CameraGraph>(cam_graph)), options(opt_options), step_size(opt_options.init_step_size), cam_type(type)
{
}


double ExtrinsicOptimizer::optimize(vector<Camera>& cameras)
{
    X = graph->buildExtrinsicParams();
    //double opt_error = error_max;
    //double last_error = opt_error;
    int iter = 0;
    double change = 1.0;

    double opt_error = computeProjectError(X);
    std::cout << "The final project error before bundle adjustment is : " << opt_error << std::endl;

    cout << "Start optimization..." << endl;
    while (iter < options.max_iter && change >= options.stop_threshold) {
        Mat JtJ_inv, Jt_error;
        solveNonLinearOpimization(JtJ_inv, Jt_error);

        double smooth = 1 - std::pow(1 - options.alpha_smooth, (double)iter + 1.0);
        Mat G = smooth * JtJ_inv * Jt_error;

        if (G.depth() == CV_64F) {
            G.convertTo(G, CV_32F);
        }

        G = G.reshape(1, 1);
        X = X + G;
        change = norm(G) / norm(X);

        opt_error = computeProjectError(X);
        cout << "Iter " << iter << ", current error: " << opt_error << ", change: " << change << endl;
        iter++;
    }

    opt_error = computeProjectError(X);
    std::cout << "The final project error after bundle adjustment is : " << opt_error << std::endl;


    for (size_t cam_idx = 0; cam_idx < graph->numCameras(); ++cam_idx) {
        if (cam_idx == graph->getReferrenceCameraID()) {
            continue;
        }

        size_t cam_offset = (cam_idx < graph->getReferrenceCameraID() ? cam_idx : cam_idx - 1) * 6;
        Mat transform = Mat::eye(4, 4, CV_64F);
        Mat om = X.colRange(cam_offset, cam_offset + 3);
        Mat T = X.colRange(cam_offset + 3, cam_offset + 6);
        rotTransToTransform(om, T, transform);

        Camera& cam = cameras[cam_idx];
        transform.convertTo(cam.extrinsic, CV_32F);
    }

    for (size_t feat_idx = 0; feat_idx < graph->numFeatures(); ++feat_idx) {
        size_t feat_offset = (feat_idx + graph->numCameras() - 1) * 6;
        Mat transform = Mat::eye(4, 4, CV_64F);
        Mat om = X.colRange(feat_offset, feat_offset + 3);
        Mat T = X.colRange(feat_offset + 3, feat_offset + 6);
        rotTransToTransform(om, T, transform);

        size_t feat_real_idx = graph->getFeatureIndex(feat_idx);
        auto& feat_vert = graph->getFeatureVertex(feat_real_idx);
        transform.convertTo(feat_vert.transform, CV_32F);
    }

    return opt_error;
}


double ExtrinsicOptimizer::solveNonLinearOpimization(Mat& JtJ_inv, Mat& JtE)
{
    const auto cameras = graph->getCameras();
    size_t nParam = X.total();
    size_t nEdge = graph->numEdges();
    std::vector<int> pointsLocation(nEdge + 1, 0);
    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        const auto& edge = graph->getEdge(edgeIdx);
        const size_t cam_id = edge.camera_vert;
        const size_t feat_id = graph->getFeatureIndex(edge.feature_vert);

        size_t num_features = cameras->at(cam_id).all_features[feat_id].feature_ids.size();
        pointsLocation[edgeIdx + 1] = pointsLocation[edgeIdx] + 2 * num_features;
    }

    JtJ_inv = Mat(nParam, nParam, CV_64F);
    JtE = Mat(nParam, 1, CV_64F);

    // construct the Jacobian and error matrices
    Mat J = Mat::zeros(pointsLocation.back(), X.total(), CV_64F);
    Mat E = Mat::zeros(pointsLocation.back(), 1, CV_64F);

    // traverse all the edges for optimization
#pragma omp parallel for
    for (int edge_idx = 0; edge_idx < nEdge; ++edge_idx) {
        const auto& edge = graph->getEdges().at(edge_idx);
        size_t cam_idx = edge.camera_vert;
        size_t feat_idx = edge.feature_vert;
        size_t feat_real_idx = graph->getFeatureIndex(feat_idx);

        const Features& feat = (*cameras)[cam_idx].all_features[feat_real_idx];
        Mat obj_points, img_points;
        Mat(feat.obj_features).convertTo(obj_points, CV_64FC3);
        Mat(feat.image_features).convertTo(img_points, CV_64FC2);

        Mat rvec_trans, tvec_trans;
        transformToRotTrans(edge.transform, rvec_trans, tvec_trans);

        size_t feat_offset = (feat_idx + graph->numCameras() - 1) * 6;
        Mat rvec_feat = X.colRange(feat_offset, feat_offset + 3);
        Mat tvec_feat = X.colRange(feat_offset + 3, feat_offset + 6);

        Mat rvec_cam, tvec_cam;
        size_t cam_offset = 0;
        if (cam_idx == graph->getReferrenceCameraID()) {
            rvec_cam = Mat::zeros(3, 1, CV_32F);
            tvec_cam = Mat::zeros(3, 1, CV_32F);
        }
        else {
            cam_offset = (cam_idx < graph->getReferrenceCameraID() ? cam_idx : cam_idx - 1) * 6;
            rvec_cam = X.colRange(cam_offset, cam_offset + 3);
            tvec_cam = X.colRange(cam_offset + 3, cam_offset + 6);
        }

        Mat J_feat, J_cam, error;
        Mat intrinsic = (*cameras)[cam_idx].intrinsic;
        Mat distortion = (*cameras)[cam_idx].distortion.getDistortionVector();
        double xi = (*cameras)[cam_idx].xi;
        bool is_omni = (*cameras)[cam_idx].type == CameraType::OmniDirectional;
        if (is_omni) {
            distortion = (*cameras)[cam_idx].distortion.getDistortionVector(4);
        }

        computePhotoCameraJacobian(rvec_feat, tvec_feat, rvec_cam, tvec_cam, rvec_trans, tvec_trans,
            obj_points, img_points, intrinsic, distortion, xi, J_feat, J_cam, error, is_omni);

        if (cam_idx != graph->getReferrenceCameraID()) {
            J_cam.copyTo(J.rowRange(pointsLocation[edge_idx], pointsLocation[edge_idx + 1]).
                colRange(cam_offset, cam_offset + 6));
        }

        J_feat.copyTo(J.rowRange(pointsLocation[edge_idx], pointsLocation[edge_idx + 1]).
            colRange(feat_offset, feat_offset + 6));
        error.copyTo(E.rowRange(pointsLocation[edge_idx], pointsLocation[edge_idx + 1]));
    }

    JtJ_inv = (J.t() * J + 1e-10).inv();
    JtE = J.t() * E;

    return 0.0;
}

double ExtrinsicOptimizer::computeProjectError(Mat& parameters)
{
    size_t nVertex = graph->numCameras() + graph->numFeatures();
    size_t nEdge = graph->numEdges();
    CV_Assert((int)parameters.total() == (nVertex - 1) * 6 && parameters.depth() == CV_32F);


    // recompute the transform between photos and cameras

    auto edgeList = graph->getEdges();
    std::vector<Vec3f> rvecVertex, tvecVertex;
    vector2parameters(parameters, nVertex, rvecVertex, tvecVertex);

    float totalError = 0;
    int totalNPoints = 0;
    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        Mat transform, photoTransform, cameraTransform;
        int cameraVertex = edgeList[edgeIdx].camera_vert;
        int photoVertex = edgeList[edgeIdx].feature_vert;
        int photoIndex = graph->getFeatureIndex(photoVertex);

        int realPhotoVertex = photoVertex + graph->numCameras() - 1;
        rotTransToTransform(Mat(rvecVertex[realPhotoVertex]), Mat(tvecVertex[realPhotoVertex]), photoTransform);

        cameraTransform = Mat::eye(4, 4, CV_32F);
        if (cameraVertex != graph->getReferrenceCameraID())
        {
            int realCameraVertex = cameraVertex < graph->getReferrenceCameraID() ? cameraVertex : cameraVertex - 1;
            rotTransToTransform(Mat(rvecVertex[realCameraVertex]), Mat(tvecVertex[realCameraVertex]), cameraTransform);
        }
        transform = cameraTransform * photoTransform;

        //transform.copyTo(edgeList[edgeIdx].transform);
        Mat rvec, tvec;
        transformToRotTrans(transform, rvec, tvec);

        auto objectPoints = graph->getCameras()->at(cameraVertex).all_features[photoIndex].obj_features;
        auto imagePoints = graph->getCameras()->at(cameraVertex).all_features[photoIndex].image_features;
        auto intrinsic = graph->getCameras()->at(cameraVertex).intrinsic;

        vector<cv::Vec2d> proImagePoints;
        if (cam_type == CameraType::Pinhole) {
            auto distortion = graph->getCameras()->at(cameraVertex).distortion.getDistortionVector();
            cv::projectPoints(objectPoints, rvec, tvec, intrinsic, distortion, proImagePoints);
        }
        else if (cam_type == CameraType::OmniDirectional) {
            auto distortion = graph->getCameras()->at(cameraVertex).distortion.getDistortionVector(4);
            double xi = graph->getCameras()->at(cameraVertex).xi;
            cv::omnidir::projectPoints(objectPoints, proImagePoints, rvec, tvec, intrinsic, xi, distortion);
        }

        for (size_t i = 0; i < proImagePoints.size(); ++i) {
            cv::Vec2d err = cv::Vec2d(imagePoints[i][0] - proImagePoints[i][0], imagePoints[i][1] - proImagePoints[i][1]);
            totalError += sqrt(err[0] * err[0] + err[1] * err[1]);
            totalNPoints++;
        }
    }
    double meanReProjError = totalError / totalNPoints;
    return meanReProjError;
}

bool ExtrinsicOptimizer::computePhotoCameraJacobian(const Mat& rvec_feat, const Mat& tvec_feat, const Mat& rvec_cam,
    const Mat& tvec_cam, Mat& rvec_trans, Mat& tvec_trans, const Mat& obj_points, const Mat& img_points,
    const Mat& intrinsic, const Mat& distortion, const double xi, Mat& J_feat, Mat& J_cam, Mat& error, const bool is_omni)
{
    Mat drtrans_drfeat, drtrans_dtfeat,
        drtrans_drcam, drtrans_dtcam,
        dttrans_drfeat, dttrans_dtfeat,
        dttrans_drcam, dttrans_dtcam;

    composeMotion(rvec_feat, tvec_feat, rvec_cam, tvec_cam, rvec_trans, tvec_trans, drtrans_drfeat, drtrans_dtfeat,
        drtrans_drcam, drtrans_dtcam, dttrans_drfeat, dttrans_dtfeat, dttrans_drcam, dttrans_dtcam);

    if (rvec_trans.depth() == CV_64F) {
        rvec_trans.convertTo(rvec_trans, CV_32F);
    }
    if (tvec_trans.depth() == CV_64F) {
        tvec_trans.convertTo(tvec_trans, CV_32F);
    }

    Mat img_points_2, jacobian, dx_drvecCamera, dx_dtvecCamera, dx_drvecPhoto, dx_dtvecPhoto;

    if (is_omni) {
        cv::omnidir::projectPoints(obj_points, img_points_2, rvec_trans, tvec_trans, intrinsic, xi, distortion);
    }
    else {
        projectPoints(obj_points, rvec_trans, tvec_trans, intrinsic, distortion, img_points_2, jacobian);
    }

    if (obj_points.depth() == CV_32F) {
        Mat(img_points - img_points_2).convertTo(error, CV_64FC2);
    }
    else {
        error = img_points - img_points_2;
    }
    error = error.reshape(1, (int)img_points.total() * 2);

    dx_drvecCamera = jacobian.colRange(0, 3) * drtrans_drcam + jacobian.colRange(3, 6) * dttrans_drcam;
    dx_dtvecCamera = jacobian.colRange(0, 3) * drtrans_dtcam + jacobian.colRange(3, 6) * dttrans_dtcam;
    dx_drvecPhoto = jacobian.colRange(0, 3) * drtrans_drfeat + jacobian.colRange(3, 6) * dttrans_drfeat;
    dx_dtvecPhoto = jacobian.colRange(0, 3) * drtrans_dtfeat + jacobian.colRange(3, 6) * dttrans_dtfeat;

    J_cam = Mat(dx_drvecCamera.rows, 6, CV_64F);
    J_feat = Mat(dx_drvecPhoto.rows, 6, CV_64F);

    dx_drvecCamera.copyTo(J_cam.colRange(0, 3));
    dx_dtvecCamera.copyTo(J_cam.colRange(3, 6));
    dx_drvecPhoto.copyTo(J_feat.colRange(0, 3));
    dx_dtvecPhoto.copyTo(J_feat.colRange(3, 6));

    return true;
}

