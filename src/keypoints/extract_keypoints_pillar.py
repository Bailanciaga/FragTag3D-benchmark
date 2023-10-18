# Author: suhaot
# Date: 2023/10/4
# Description: extract_keypoints_pillar
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import norm
from src.descriptors.extract_descriptors_cov import get_cov_descriptor


def compute_smoothness(vertices, n_neighbors):
    n_p = len(vertices)
    tree = KDTree(vertices)
    _, neighbors = tree.query(vertices, p=2, k=n_neighbors)

    c = []
    for hood in neighbors:
        point = hood[0]
        neigh = hood[1:]
        diff = [[vertices[point] - vertices[n]] for n in neigh]
        c.append(norm(np.sum(diff, axis=0), 2) / (n_p * norm(vertices[point], 2)))
    return c


def get_pillar_keypoints(fragment, n_neighbors, n_keypoints, sharp_percentage=0.5):
    vertices = fragment[:, :3]
    c = np.array(compute_smoothness(vertices, n_neighbors))
    # normalize it and argsort to get lowest and highest values
    idx_sorted = np.argsort(c)
    # get the first and last ones as best keypoints
    n_sharp = int(n_keypoints * sharp_percentage)
    n_plan = n_keypoints - n_sharp

    # get start for planar regions
    start_planar = next((i for i, val in enumerate(c[idx_sorted]) if val > 5e-3), 0)
    planar_idx = idx_sorted[start_planar:n_plan + start_planar]
    sharp_idx = idx_sorted[-n_sharp:]
    indices_to_keep = np.append(planar_idx, sharp_idx)

    # kpts_planar = np.array(vertices[planar_idx])
    # kpts_sharp = np.array(vertices[sharp_idx])
    # kpts = np.append(kpts_planar, kpts_sharp, axis=0)

    # scores_planar = np.array(c[planar_idx])
    # scores_sharp = np.array(c[sharp_idx])
    # scores = np.append(scores_planar, scores_sharp, axis=0)
    # scores = scores / np.max(scores)

    # keypoints = np.column_stack((kpts, scores))
    keypoints = fragment[indices_to_keep]
    keypoints_idxs = indices_to_keep

    return get_cov_descriptor(fragment, keypoints, keypoints_idxs)