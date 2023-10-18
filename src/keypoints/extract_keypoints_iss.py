# Author: suhaot
# Date: 2023/10/4
# Description: extract_keypoint_iss
import open3d as o3d
import numpy as np
from src.descriptors.extract_descriptors_cov import get_cov_descriptor


def get_iss_keypoints(fragment):
    vertices = fragment[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))

    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)

    # 使用K-D树查找每个关键点在原始点云中的索引
    search_tree = o3d.geometry.KDTreeFlann(pcd)
    keypoint_indexes = []
    for point in keypoints.points:
        _, idx, _ = search_tree.search_knn_vector_3d(point, 1)
        keypoint_indexes.append(idx[0])

    keypoints = fragment[keypoint_indexes]

    return get_cov_descriptor(fragment, keypoints, keypoint_indexes)
