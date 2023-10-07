# Author: suhaot
# Date: 2023/10/4
# Description: extract_keypoints_harris
import os
from copy import deepcopy
import numpy as np
from sklearn.decomposition import PCA
from src.keypoints.tools.neighborhoords import k_ring_delaunay_adaptive
from src.keypoints.tools.tools import polyfit3d
from src.keypoints.tools.transformation import centering_centroid
from src.descriptors.extract_descriptors_cov import get_cov_descriptor


def get_harris_keypoints(fragment, npoints, outputpath):
    vertices = fragment[:, :3]
    points = deepcopy(vertices)
    # parameters
    delta = 0.025
    k = 0.04

    # subsample for big pointclouds
    # if len(points) > 5000:
    #     samp_idx = np.random.choice(len(points), 5000, replace=False)
    #     points = points[samp_idx]

    # initialisation of the solution
    resp = np.zeros(len(points))

    # compute neighborhood
    neighborhood = k_ring_delaunay_adaptive(points, delta)

    for i in neighborhood.keys():
        points_centred, _ = centering_centroid(points)

        # best fitting point
        points_pca = PCA(n_components=3).fit_transform(
            np.transpose(points_centred))
        _, eigenvectors = np.linalg.eigh(points_pca)

        # rotate the cloud
        for i in range(points.shape[0]):
            points[i, :] = np.dot(np.transpose(eigenvectors), points[i, :])

        # restrict to XY plane and translate
        points_2D = points[:, :2] - points[i, :2]

        # fit a quadratic surface
        m = polyfit3d(points_2D[:, 0], points_2D[:, 1], points[:, 2], order=2)
        m = m.reshape((3, 3))

        # Compute the derivative
        fx2 = m[1, 0] * m[1, 0] + 2 * m[2, 0] * m[2, 0] + 2 * m[1, 1] * m[1, 1]  # A
        fy2 = m[1, 0] * m[1, 0] + 2 * m[1, 1] * m[1, 1] + 2 * m[0, 2] * m[0, 2]  # B
        fxfy = m[1, 0] * m[0, 1] + 2 * m[2, 0] * m[1, 1] + 2 * m[1, 1] * m[0, 2]  # C

        # Compute response
        resp[i] = fx2 * fy2 - fxfy * fxfy - k * (fx2 + fy2) * (fx2 + fy2)

    # Select interest points at local maxima
    candidate = []
    for i in neighborhood.keys():
        if resp[i] >= np.max(resp[neighborhood[i]]):
            candidate.append([i, resp[i]])
    # sort by decreasing order
    candidate.sort(reverse=True, key=lambda x: x[1])
    candidate = np.array(candidate)

    keypoint_indexes = np.array(
        candidate[:npoints, 0], dtype=np.int64)

    keypoints = fragment[keypoint_indexes]

    get_cov_descriptor(fragment, keypoints, keypoint_indexes, outputpath)