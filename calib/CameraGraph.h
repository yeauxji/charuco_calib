#pragma once

#include "Camera.h"

#include <queue>
#include <unordered_map>

struct CameraVertex {
	// all feature indices captured by the camera
	std::vector<size_t> feats;
	// a 4x4 matrix of transformation from feature to camera
	cv::Mat transform = cv::Mat::zeros(4, 4, CV_32F);
};

struct FeatureVertex {
	// The feature index of camera captured features
	size_t idx;
	// all cameras have captured this feature
	std::vector<size_t> cams;
	// a 4x4 matrix of transformation from feature to camera
	cv::Mat transform = cv::Mat::zeros(4, 4, CV_32F);
};

struct SceneEdge {
	size_t camera_vert;
	size_t feature_vert;
	// the 4x4 transform matrix from the scene to the camera
	cv::Mat transform;
};

/**
 * A graph used for camera bundle adjustment.
 *
 * The graph is bipartite, a CameraVertex saves a camera index
 * and a PatternVertex saves a pattern index. If a pattern is
 * successfully captured by a camera, we use a scene edge to link
 * the corresponding CameraVertex and PatternVertex.
 */
class CameraGraph
{
public:
	explicit CameraGraph(std::vector<Camera>& cameras);

	~CameraGraph();

	/**
	 * Update the referrence camera id. The size should be
	 * in the range [0, num_cameras).
	 *
	 * \param referrence_camera_id set the referrence camera,
	 * the transform matrix of this camera is identity.
	 *
	 * \return true if the camera id is valid.
	 */
	bool updateReferrenceCameraID(size_t referrence_camera_id);

	bool updateCameraExtrinsics(const std::vector<Camera>& cameras);


	/**
	 * traverse the graph and calculate the transformation 
	 * from the referrence camera to the nodes in the graph.
	 *
	 * \return true if the process is successful
	 */
	bool calculateTransforms();


	cv::Mat buildExtrinsicParams();

	cv::Mat buildRigExtrinsicParams(size_t num_rig_cameras);

	/**
	 * Return the total number of camera vertices.
	 */
	size_t numCameras() const {
		return camera_vertices.size();
	}

	/**
	 * Return the total number of feature vertices, the number
	 * indicates the number of valid calibration board positions.
	 * 
	 * \return 
	 */
	size_t numFeatures() const {
		return feature_vertices.size();
	}

	/**
	 * Return the total number of edges, the number indicates
	 * the number of valid photos captured by all cameras.
	 */
	size_t numEdges() const {
		return edges.size();
	}

	/**
	 * Return the index of the referrence camera in the camera
	 * vertices array.
	 */
	size_t getReferrenceCameraID() const {
		return root;
	}

	/**
	 * Return the const referrences of vertices and edges.
	 */
	const std::vector<CameraVertex>& getCameraVertices() const {
		return camera_vertices;
	}

	const std::vector<size_t>& getFeatureIndices() const {
		return feature_indices;
	}

	size_t getFeatureIndex(size_t idx) const {
		return feature_indices[idx];
	}

	const std::unordered_map<size_t, FeatureVertex>& getFeatureVertices() const {
		return feature_vertices;
	}

	FeatureVertex& getFeatureVertex(size_t feat_idx) {
		return feature_vertices[feat_idx];
	}

	const std::vector<SceneEdge>& getEdges() const {
		return edges;
	}

	const SceneEdge& getEdge(size_t idx) const {
		return edges[idx];
	}

	/**
	 * Return the pointer to all cameras.
	 */
	const Sp<std::vector<Camera>> getCameras() const {
		return cameras_ptr;
	}

	bool saveCameraParamsToFile(std::string filename, bool with_intrinsic = true, bool with_dist = true, 
		float reproj_err = 0.0, bool save_rig_only = false, size_t num_rig_cams = 0);

	bool saveFeaturesToFile(std::string filename, bool with_cov = false);

	/**
	 * Reset the graph to 0.
	 *
	 */
	void flush();

private:
	// A shared pointer to the camera array
	Sp<std::vector<Camera>> cameras_ptr = nullptr;
	// Camera vertices in the graph
	std::vector<CameraVertex> camera_vertices;
	// Feature vertices in the graph
	std::unordered_map<size_t, FeatureVertex> feature_vertices;
	// Feature vertices index
	std::vector<size_t> feature_indices;

	// All edges in the array, each edge links a camera vertex
	// and a feature vertex
	std::vector<SceneEdge> edges;

	size_t root = 0;

	/**
	 * Construct the graph by updating num_cameras and
	 * num_features. It works because we only need the
	 * vertex index for referrence, and finding the edges
	 * is trivial because the camera-feature relations
	 * are saved in each camera.
	 *
	 * \return true if the graph is valid, that means the
	 * camera number > 0 and all features sizes are equal.
	 */
	bool updateGraph();
};