# Author: suhaot
# Date: 2023/10/12
# Description: extract_keypoints_SD
import numpy as np
from scipy.spatial import KDTree
from src.descriptors.extract_descriptors_cov import get_cov_descriptor


def compute_SD_point(neighbourhood, points, normals, p_idx):
    p_i = points[p_idx]
    n_p_i = normals[p_idx]
    p_i_bar = np.mean(points[neighbourhood], axis=0)
    v = p_i - p_i_bar
    SD = np.dot(v, n_p_i)
    return SD


def get_SD_keypoints(fragment, normals, nkeypoints, r=0.05):
    """
    Returns the SD keypoints with a score value normalized.
    Following https://ieeexplore.ieee.org/document/9279208.
    """
    vertices = fragment[:, :3]
    n_points = len(vertices)
    tree = KDTree(vertices)

    # Compute SD
    SD = np.zeros((n_points))
    neighbourhoods = tree.query_ball_point(vertices, r)
    for i in range(n_points):
        SD[i] = compute_SD_point(np.asarray(
            neighbourhoods[i]), vertices, normals, i)

    indices_to_keep = np.argsort(np.abs(SD))[-nkeypoints:]
    keypoints = np.array(vertices[indices_to_keep])
    scores = np.array(np.abs(SD)[indices_to_keep])
    scores = scores / np.max(scores)
    scores = scores[:, None]
    keypoints = np.append(keypoints, scores, axis=1)
    return get_cov_descriptor(fragment, keypoints, indices_to_keep)