#include "CameraGraph.h"

#include <opencv2/core/quaternion.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

namespace {
    const int INVALID = -2;

    /**
     * Fill the extrinsic vector with the R.
     *
     * \param transform the 4x4 transform matrix
     * \param extrinsic_vector 1xn extrinsic vector, n is the
     * total number of elements of the camera parameters
     * \param offset current position of the parameter vector
     */
    void fillExtrinsicVector(const Mat& transform, Mat& params_vector, size_t offset)
    {
        Mat R, T;
        transformToRotTrans(transform, R, T);
        R.reshape(1, 1).copyTo(params_vector.colRange(offset, offset + 3));
        T.reshape(1, 1).copyTo(params_vector.colRange(offset + 3, offset + 6));
    }

    /**
     * Fill the extrinsic vector with the R.
     *
     * \param intrinsic the 3x3 camera intrinsic matrix
     * \param distortion the 1x5 camera distortion parameters matrix
     * \param extrinsic_vector 1xn extrinsic vector, n is the
     * total number of elements of the camera parameters
     * \param offset current position of the parameter vector
     */
    void fillIntrinsicVector(const Mat& intrinsic, const Mat& distortion, Mat& params_vector, size_t offset) {
        // convert the 3x3 intrinsic matrix to a 1x4 vector with 
        // the order [fx, fy, cx, cy]
        Mat intrinsic_params(1, 4, CV_64F);
        intrinsic_params.at<double>(0, 0) = intrinsic.at<double>(0, 0);
        intrinsic_params.at<double>(0, 1) = intrinsic.at<double>(1, 1);
        intrinsic_params.at<double>(0, 2) = intrinsic.at<double>(0, 2);
        intrinsic_params.at<double>(0, 3) = intrinsic.at<double>(1, 2);

        intrinsic_params.copyTo(params_vector.colRange(offset, offset + 4));
        distortion.copyTo(params_vector.colRange(offset + 4, offset + 9));
    }

    void findRowNonZero(const Mat& row, Mat& idx)
    {
        CV_Assert(!row.empty() && row.rows == 1 && row.channels() == 1);
        Mat _row;
        std::vector<int> _idx;
        row.convertTo(_row, CV_32F);
        for (int i = 0; i < (int)row.total(); ++i)
        {
            if (_row.at<float>(i) != 0)
            {
                _idx.push_back(i);
            }
        }
        idx.release();
        idx.create(1, (int)_idx.size(), CV_32S);
        for (int i = 0; i < (int)_idx.size(); ++i)
        {
            idx.at<int>(i) = _idx[i];
        }
    }

    void graphTraverse(const Mat& G, int begin, std::vector<int>& order, std::vector<int>& pre)
    {
        CV_Assert(!G.empty() && G.rows == G.cols);
        int nVertex = G.rows;
        order.resize(0);
        pre.resize(nVertex, INVALID);
        pre[begin] = -1;
        std::vector<bool> visited(nVertex, false);
        std::queue<int> q;
        visited[begin] = true;
        q.push(begin);
        order.push_back(begin);

        while (!q.empty())
        {
            int v = q.front();
            q.pop();
            Mat idx;
            // use my findNonZero maybe
            findRowNonZero(G.row(v), idx);
            for (int i = 0; i < (int)idx.total(); ++i)
            {
                int neighbor = idx.at<int>(i);
                if (!visited[neighbor])
                {
                    visited[neighbor] = true;
                    q.push(neighbor);
                    order.push_back(neighbor);
                    pre[neighbor] = v;
                }
            }
        }
    }

    vector<double> getQuaternion(const Mat& R)
    {
        double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
        vector<double> Q(4, 0.0);

        if (trace > 0.0)
        {
            double s = sqrt(trace + 1.0);
            Q[3] = (s * 0.5);
            s = 0.5 / s;
            Q[0] = ((R.at<double>(2, 1) - R.at<double>(1, 2)) * s);
            Q[1] = ((R.at<double>(0, 2) - R.at<double>(2, 0)) * s);
            Q[2] = ((R.at<double>(1, 0) - R.at<double>(0, 1)) * s);
        }

        else
        {
            int i = R.at<double>(0, 0) < R.at<double>(1, 1) ? (R.at<double>(1, 1) < R.at<double>(2, 2) ? 2 : 1) : (R.at<double>(0, 0) < R.at<double>(2, 2) ? 2 : 0);
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;

            double s = sqrt(R.at<double>(i, i) - R.at<double>(j, j) - R.at<double>(k, k) + 1.0);
            Q[i] = s * 0.5;
            s = 0.5 / s;

            Q[3] = (R.at<double>(k, j) - R.at<double>(j, k)) * s;
            Q[j] = (R.at<double>(j, i) + R.at<double>(i, j)) * s;
            Q[k] = (R.at<double>(k, i) + R.at<double>(i, k)) * s;
        }

        return Q;
    }
}


CameraGraph::CameraGraph(std::vector<Camera>& cameras)
{
    cameras_ptr = std::make_shared<std::vector<Camera>>(cameras);

    updateGraph();
}

CameraGraph::~CameraGraph()
{
    flush();
    cameras_ptr = nullptr;
}

bool CameraGraph::updateReferrenceCameraID(size_t referrence_camera_id)
{
    if (referrence_camera_id >= numCameras()) {
        cerr << "Failed to update referrence camera ID: invalid input." << endl;
        cerr << "num_cameras: " << numCameras() << ", input: " << referrence_camera_id << endl;
        return false;
    }

    root = referrence_camera_id;

    return true;
}

bool CameraGraph::updateCameraExtrinsics(const std::vector<Camera>& cameras)
{
    if (cameras.size() != numCameras()) {
        cerr << "Failed to update camera extrinsics: number of cameras mismatch." << endl;
        return false;
    }

    for (int i = 0; i < numCameras(); ++i) {
        Mat transform = cameras[i].extrinsic;
        camera_vertices[i].transform = transform;
    }

    return true;
}

bool CameraGraph::updateGraph()
{
    if (cameras_ptr == nullptr || cameras_ptr->empty()) {
        cerr << "Failed to construct graph: no camera." << endl;
        return false;
    }

    size_t features_size = cameras_ptr->at(0).all_features.size();
    if (features_size == 0) {
        cerr << "Failed to construct graph: no features are detected." << endl;
        return false;
    }

    for (auto const& cam : *cameras_ptr) {
        if (cam.all_features.size() != features_size) {
            cerr << "Failed to construct graph: features count mismatch." << endl;
            return false;
        }
    }

    for (size_t i = 0; i < cameras_ptr->size(); ++i) {
        CameraVertex cam_vert;
        camera_vertices.push_back(cam_vert);
    }

    for (size_t i = 0; i < camera_vertices.size(); ++i) {
        for (size_t j = 0; j < (*cameras_ptr)[i].all_features.size(); ++j) {
            Features const& feat = (*cameras_ptr)[i].all_features[j];
            if (feat.is_valid) {
                camera_vertices[i].feats.push_back(j);
                if (!feature_vertices.contains(j)) {
                    feature_vertices[j] = FeatureVertex();
                    feature_indices.push_back(j);
                    feature_vertices[j].idx = feature_vertices.size() - 1;
                }
                feature_vertices[j].cams.push_back(i);

                Mat transform;
                const Features& feat = (*cameras_ptr)[i].all_features[j];
                rotTransToTransform(feat.rotation, feat.translation, transform);
                edges.push_back(SceneEdge(i, feature_vertices[j].idx, transform));
            }
        }
    }

    return true;
}


bool CameraGraph::saveCameraParamsToFile(std::string filename, bool with_intrinsic, bool with_dist, float reproj_err, 
    bool save_rig_only, size_t num_rig_cams)
{
    ofstream fs(filename);
    if (!fs.is_open()) {
        cerr << "Cannot create file " << filename << endl;
        return false; 
    }

    fs << "# ";
    if (with_intrinsic) {
        fs << "fu u0 v0 ar s | ";
    }

    if (with_dist) {
        fs << "k1 k2 p1 p2 k3 | ";
    }

    fs << "quaternion(scalar part first) translation | ";

    fs << "width height | ";

    fs << "\n";

    // calib reproj error 
    fs << "# reprojection error: " << std::to_string(reproj_err) << "\n";

    char buf[1024];

    for (size_t i = 0; i < numCameras(); ++i) {
        if (save_rig_only && i % num_rig_cams != root) {
			continue;
		}

        if (with_intrinsic) {
            /* The intrinsic is saved in 5 parameters: focal length in x pixels, 
            * principal point coordinates in pixels, aspect ratio [i.e. focalY/focalX]
            * and skew factor
            */
            Mat intrinsic = cameras_ptr->at(i).intrinsic;
            if (intrinsic.type() != CV_64F) {
                intrinsic.convertTo(intrinsic, CV_64F);
            }

            sprintf_s(buf, "%lf %lf %lf %lf %lf ",
                intrinsic.at<double>(0, 0),
                intrinsic.at<double>(0, 2),
                intrinsic.at<double>(1, 2),
                intrinsic.at<double>(0, 0) / intrinsic.at<double>(1, 1),
                intrinsic.at<double>(0, 1));

            fs << buf;
        }

        if (with_dist) {
            auto kc = cameras_ptr->at(i).distortion;
            sprintf_s(buf, "%lf %lf %lf %lf %lf ",
                kc.k_1, kc.k_2, kc.p_1, kc.p_2, kc.k_3);
            fs << buf;
        }

        Mat transform = camera_vertices[i].transform;
        if (transform.type() != CV_64F) {
            transform.convertTo(transform, CV_64F);
        }

        Mat R = transform.rowRange(0, 3).colRange(0, 3);
        auto quat = getQuaternion(R);
        // Quatd quat = Quatd::createFromRotMat(R);
        sprintf_s(buf, "%lf %lf %lf %lf ",
        //    quat.w, quat.x, quat.y, quat.z);
            quat[3], quat[0], quat[1], quat[2]);
        fs << buf;

        Mat T = transform.rowRange(0, 3).col(3);
        sprintf_s(buf, "%lf %lf %lf ",
            T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0));
        fs << buf;

        sprintf_s(buf, "%d %d", cameras_ptr->at(i).width, cameras_ptr->at(i).height);
        fs << buf;

        fs << "\n";
    }

    fs.close();
    return true;
}

bool CameraGraph::saveFeaturesToFile(std::string filename, bool with_cov)
{
    ofstream fs(filename);
    if (!fs.is_open()) {
        cerr << "Cannot create file " << filename << endl;
        return false;
    }

    char buf[1024];

    for (const auto& [fid, feat] : feature_vertices) {
        Mat transform = feat.transform;
        if (transform.type() != CV_64F) {
            transform.convertTo(transform, CV_64F);
        }

        Mat R = transform.rowRange(0, 3).colRange(0, 3);
        Mat T = transform.rowRange(0, 3).col(3);

        size_t num_cam = feat.cams.size();
        ChArUcoBoardPtr board_ptr = cameras_ptr->at(feat.cams[0]).all_features[0].board_ptr;
        size_t num_points = board_ptr->chessboardCorners.size();

        Mat cp = Mat::zeros(num_cam, num_points, CV_32S);
        for (size_t i = 0; i < num_cam; ++i) {
            const Camera& cam = cameras_ptr->at(feat.cams[i]);
            const auto& point_ids = cam.all_features[fid].feature_ids;
            for (auto j : point_ids) {
                cp.at<int>(i, j) = 1;
            }
        }

        for (size_t j = 0; j < num_points; ++j) {
            Point3f p3f = board_ptr->chessboardCorners[j];
            Mat point(3, 1, CV_64F);
            point.at<double>(0, 0) = (double)p3f.x;
            point.at<double>(1, 0) = (double)p3f.y;
            point.at<double>(2, 0) = (double)p3f.z;
            Mat new_point = R * point + T;
            int nframes = countNonZero(cp.col(j));

            if (nframes == 0) {
                continue;
            }

            sprintf_s(buf, "%lf %lf %lf %d ",
                new_point.at<double>(0, 0), new_point.at<double>(1, 0), new_point.at<double>(2, 0), nframes);

            fs << buf;

            for (size_t i = 0; i < num_cam; ++i) {
                if (cp.at<int>(i, j) == 1) {
                    const Camera& cam = cameras_ptr->at(feat.cams[i]);
                    const auto& point_ids = cam.all_features[fid].feature_ids;
                    auto find_it = find(point_ids.begin(), point_ids.end(), j);
                    size_t pid = find_it - point_ids.begin();
                    const auto& reproject_point = cam.all_features[fid].reprojected_features[pid];

                    sprintf_s(buf, "%d %lf %lf ",
                        (int)feat.cams[i], (double)reproject_point[0], (double)reproject_point[1]);

                    fs << buf;
                }
            }

            fs << "\n";
        }
    }

    fs.close();
    return true;
}

void CameraGraph::flush()
{
    camera_vertices.clear();
    feature_vertices.clear();
    edges.clear();
}


bool CameraGraph::calculateTransforms() {

    if (cameras_ptr == nullptr) {
        cerr << "Failed to traverse graph: no camera." << endl;
        return false;
    }

    if (numCameras() == 0 || numFeatures() == 0) {
        cerr << "No vertex detected, try to reconstruct the graph." << endl;
        camera_vertices.clear();
        updateGraph();
    }

    if (numCameras() == 0 || numFeatures() == 0) {
        cerr << "Reconstruction failed, check the cameras." << endl;
        return false;
    }

    // build the edges with a breadth first search
    Mat G = Mat::zeros(numCameras() + numFeatures(), numCameras() + numFeatures(), CV_32S);
    for (int edge_idx = 0; edge_idx < numEdges(); ++edge_idx) {
        auto& edge = edges[edge_idx];
        G.at<int>(edge.camera_vert, edge.feature_vert + camera_vertices.size()) = edge_idx + 1;
    }
    G = G + G.t();


    // traverse the graph
    std::vector<int> pre, order;
    graphTraverse(G, root, order, pre);

    for (int i = 0; i < numCameras(); ++i)
    {
        if (pre[i] == INVALID)
        {
            std::cout << "camera" << i << "is not connected" << std::endl;
        }
    }

    camera_vertices[root].transform = Mat::eye(4, 4, CV_32F);

    for (int i = 1; i < (int)order.size(); ++i)
    {
        int vertexIdx = order[i];
        int preIdx = pre[vertexIdx];
        int edgeIdx = G.at<int>(vertexIdx, pre[vertexIdx]) - 1;
        Mat transform = edges[edgeIdx].transform;

        if (vertexIdx < numCameras())
        {
            preIdx = feature_indices[preIdx - numCameras()];
            Mat prePose = feature_vertices[preIdx].transform;

            camera_vertices[vertexIdx].transform = transform * prePose.inv();
        }
        else
        {
            Mat prePose = camera_vertices[pre[vertexIdx]].transform;
            vertexIdx = feature_indices[vertexIdx - numCameras()];

            feature_vertices[vertexIdx].transform = prePose.inv() * transform;
        }
    }

    return true;
}

Mat CameraGraph::buildExtrinsicParams()
{
    int nVertex = numCameras() + numFeatures();

    Mat extrinParam(1, (nVertex - 1) * 6, CV_32F);
    int offset = 0;

    for (size_t i = 0; i < numCameras(); ++i) {
        if (i == root)
            continue;

        auto& cam_vert = camera_vertices[i];
        fillExtrinsicVector(cam_vert.transform, extrinParam, offset);
        offset += 6;
    }

    for (size_t i = 0; i < numFeatures(); ++i) {
        auto& feat_vert = feature_vertices[feature_indices[i]];
        fillExtrinsicVector(feat_vert.transform, extrinParam, offset);
        offset += 6;
    }

    return extrinParam;
}

Mat CameraGraph::buildRigExtrinsicParams(size_t num_rig_cameras)
{
    assert(num_rig_cameras > 0 && numCameras() % num_rig_cameras == 0);

    int num_rigs = numCameras() / num_rig_cameras;
    int nVertex = num_rigs + numFeatures();

    Mat extrinParam(1, (nVertex - 1) * 6, CV_32F);
    int offset = 0;

    for (size_t i = 1; i < num_rigs; ++i) {
        auto& cam_vert = camera_vertices[i * num_rig_cameras + root];
        fillExtrinsicVector(cam_vert.transform, extrinParam, offset);
        offset += 6;
    }

    for (size_t i = 0; i < numFeatures(); ++i) {
        auto& feat_vert = feature_vertices[feature_indices[i]];
        fillExtrinsicVector(feat_vert.transform, extrinParam, offset);
        offset += 6;
    }

    return extrinParam;
}