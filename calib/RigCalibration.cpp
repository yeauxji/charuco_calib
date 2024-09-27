#define _USE_MATH_DEFINES

#include "RigCalibration.h"

#include <iostream>
#include <Windows.h>
#include <queue>

#include <opencv2/ccalib/omnidir.hpp>

using namespace std;
using namespace cv;

namespace {
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
}


RigExtrinsicOptimizer::RigExtrinsicOptimizer(CameraGraph& cam_graph, const ExtrinsicOptimizerOptions& opt_options, CameraType type) :
    graph(make_shared<CameraGraph>(cam_graph)), options(opt_options), step_size(opt_options.init_step_size), cam_type(type)
{
}


double RigExtrinsicOptimizer::optimize(std::vector<Camera>& cameras, std::vector<Camera>& rig_cameras)
{
    //double opt_error = error_max;
    //double last_error = opt_error;
    int iter = 0;
    double change = 1.0;

    X = graph->buildRigExtrinsicParams(rig_cameras.size());

    double opt_error = computeProjectError(X, rig_cameras);
    std::cout << "The final project error before bundle adjustment is : " << opt_error << std::endl;

    cout << "Start optimization..." << endl;
    double prev_error = -1e5;
    while (iter < options.max_iter && change >= options.stop_threshold) {

        Mat JtJ_inv, Jt_error;
        solveNonLinearOpimization(JtJ_inv, Jt_error, rig_cameras);

        double smooth = 1 - std::pow(1 - options.alpha_smooth, (double)iter + 1.0);
        Mat G = smooth * JtJ_inv * Jt_error;

        if (G.depth() == CV_64F) {
            G.convertTo(G, CV_32F);
        }

        G = G.reshape(1, 1);
        X = X + G;
        change = norm(G) / norm(X);

        opt_error = computeProjectError(X, rig_cameras);

        if (opt_error > prev_error + options.stop_threshold) {
            X = X - G;
            break;
		}
	    
        prev_error = opt_error;

        cout << "Iter " << iter << ", current error: " << opt_error << ", change: " << change << endl;
        iter++;
    }

    opt_error = computeProjectError(X, rig_cameras);
    std::cout << "The final project error after bundle adjustment is : " << opt_error << std::endl;


    size_t nRig = graph->numCameras() / rig_cameras.size();
    for (size_t rig_idx = 0; rig_idx < nRig; ++rig_idx) {
        Mat rig_transform = Mat::eye(4, 4, CV_64F);

        if (rig_idx != 0) {
            Mat R, T, om;
            size_t rig_offset = (rig_idx - 1) * 6;
            om = X.colRange(rig_offset, rig_offset + 3);
            Rodrigues(om, R);
            R.copyTo(rig_transform.colRange(0, 3).rowRange(0, 3));
            T = X.colRange(rig_offset + 3, rig_offset + 6);
            T.reshape(1, 3).copyTo(rig_transform.rowRange(0, 3).col(3));
        }

        for (size_t rig_cam_idx = 0; rig_cam_idx < rig_cameras.size(); ++rig_cam_idx) {
            Mat& cam_transform = rig_cameras[rig_cam_idx].extrinsic;
            Mat transform = cam_transform * rig_transform;

            size_t cam_idx = rig_idx * rig_cameras.size() + rig_cam_idx;
            Camera& cam = cameras[cam_idx];
            transform.convertTo(cam.extrinsic, CV_32F);
        }
    }

    for (size_t feat_idx = 0; feat_idx < graph->numFeatures(); ++feat_idx) {
        size_t feat_offset = (feat_idx + nRig - 1) * 6;
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

double RigExtrinsicOptimizer::solveNonLinearOpimization(Mat& JtJ_inv, Mat& JtE, std::vector<Camera>& rig_cameras)
{
    const auto cameras = graph->getCameras();
    size_t nParam = X.total();
    size_t nEdge = graph->numEdges();
    size_t nRig = graph->numCameras() / rig_cameras.size();
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
        int rigIdx = cam_idx / rig_cameras.size();
        int rigCameraIdx = cam_idx % rig_cameras.size();

        const Features& feat = (*cameras)[cam_idx].all_features[feat_real_idx];
        Mat obj_points, img_points;
        Mat(feat.obj_features).convertTo(obj_points, CV_64FC3);
        Mat(feat.image_features).convertTo(img_points, CV_64FC2);

        Mat rvec_trans, tvec_trans;
        transformToRotTrans(edge.transform, rvec_trans, tvec_trans, CV_64F);

        size_t feat_offset = (feat_idx + nRig - 1) * 6;
        Mat rvec_feat = X.colRange(feat_offset, feat_offset + 3);
        Mat tvec_feat = X.colRange(feat_offset + 3, feat_offset + 6);

        Mat rvec_cam, tvec_cam;
        size_t rig_offset = 0;
        Mat rig_camera_transform;
        (rig_cameras[rigCameraIdx].extrinsic).convertTo(rig_camera_transform, CV_64F);
        Mat rig_transform = Mat::eye(4, 4, CV_64F);
        if (rigIdx != 0)
        {
            rig_offset = (rigIdx - 1) * 6;
            Mat rvec_rig = X.colRange(rig_offset, rig_offset + 3);
            Mat tvec_rig = X.colRange(rig_offset + 3, rig_offset + 6);
            rotTransToTransform(rvec_rig, tvec_rig, rig_transform, CV_64F);
        }
        Mat camera_transform = rig_camera_transform * rig_transform;
        transformToRotTrans(camera_transform, rvec_cam, tvec_cam, CV_64F);

        Mat J_feat, J_cam, error;
        Mat intrinsic = rig_cameras[rigCameraIdx].intrinsic;
        Mat distortion = rig_cameras[rigCameraIdx].distortion.getDistortionVector();
        double xi = 0;
        bool is_omni = false;

        computePhotoCameraJacobian(rvec_feat, tvec_feat, rvec_cam, tvec_cam, rvec_trans, tvec_trans,
            obj_points, img_points, intrinsic, distortion, xi, J_feat, J_cam, error, is_omni);

        if (rigIdx != 0) {
            J_cam.copyTo(J.rowRange(pointsLocation[edge_idx], pointsLocation[edge_idx + 1]).
                colRange(rig_offset, rig_offset + 6));
        }


        J_feat.copyTo(J.rowRange(pointsLocation[edge_idx], pointsLocation[edge_idx + 1]).
            colRange(feat_offset, feat_offset + 6));
        error.copyTo(E.rowRange(pointsLocation[edge_idx], pointsLocation[edge_idx + 1]));
    }

    JtJ_inv = (J.t() * J + 1e-10).inv();
    JtE = J.t() * E;

    return 0.0;
}

double RigExtrinsicOptimizer::computeProjectError(Mat& parameters, std::vector<Camera>& rig_cameras)
{
    assert(graph->numCameras() % rig_cameras.size() == 0);
    size_t nRig = graph->numCameras() / rig_cameras.size();
    size_t nVertex = nRig + graph->numFeatures();
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
        Mat transform, photoTransform, cameraTransform, rigCameraTransform, rigTransform;
        int cameraVertex = edgeList[edgeIdx].camera_vert;
        int photoVertex = edgeList[edgeIdx].feature_vert;
        int photoIndex = graph->getFeatureIndex(photoVertex);

        int rigIdx = cameraVertex / rig_cameras.size();
        int rigCameraIdx = cameraVertex % rig_cameras.size();

        int realPhotoVertex = photoVertex + nRig - 1;
        rotTransToTransform(Mat(rvecVertex[realPhotoVertex]), Mat(tvecVertex[realPhotoVertex]), photoTransform);

        (rig_cameras[rigCameraIdx].extrinsic).convertTo(rigCameraTransform, CV_32F);
        rigTransform = Mat::eye(4, 4, CV_32F);
        if (rigIdx != 0)
        {
            rotTransToTransform(Mat(rvecVertex[rigIdx - 1]), Mat(tvecVertex[rigIdx - 1]), rigTransform);
        }
        cameraTransform = rigCameraTransform * rigTransform;
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

            //if (sqrt(err[0] * err[0] + err[1] * err[1]) > 5)
            //{
            //    std::cout << "rig_idx: " << rigIdx << ", rig_camera_idx: " << rigCameraIdx << 
            //        ", photo_idx: " << photoIndex << ", error: " << sqrt(err[0] * err[0] + err[1] * err[1]) << std::endl;
            //}
        }
    }
    double meanReProjError = totalError / totalNPoints;
    return meanReProjError;
}

bool RigExtrinsicOptimizer::computePhotoCameraJacobian(const Mat& rvec_feat, const Mat& tvec_feat, const Mat& rvec_cam,
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

