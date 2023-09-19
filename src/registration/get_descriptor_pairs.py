# Author: suhaot
# Date: 2023/9/14
# Description: get_descriptor_pairs
import numpy as np
import networkx as nx
import itertools
from scipy.spatial.distance import cdist
def get_descriptor_pairs_classical(kp1, kp2):
    # Find pairs of keypoints of two fragments with similar descriptors
    # Input: kp (keypoints struct), index fragment1, index fragment2
    print_flag = False
    # Prepare output
    d_pairs = []
    d_dist = []
    gt_dist = []

    # Config
    max_pairs = 30
    # Threshold value. Pairs with d>max_d will be discarded
    # Use 0.5 for experiments with Noise. For noise-free datasets use 0.2
    max_d = 0.5
    # Parameters for Descriptor
    desc_features = 3  # No of geometric features (3 or 7)
    desc_scales = 5  # (No of radii (multi-scale))
    # Use 3 and 5 for all sample datasets provided or set them according to the
    # params used in keypoint extraction

    desc_mat_sz = desc_features * desc_features
    X = kp1[:, 6:]
    Y = kp2[:, 6:]

    # Multi-scale average of Frobenius Norm
    k = 1
    # D = pdist(Y[:,(k-1)*desc_mat_sz:k*desc_mat_sz], X[:,(k-1)*desc_mat_sz:k*desc_mat_sz], 'euclidean')
    a = Y[:, (k - 1) * desc_mat_sz:k * desc_mat_sz]
    b = X[:, (k - 1) * desc_mat_sz:k * desc_mat_sz]

    D = cdist(a, b, metric='euclidean')

    # D = np.sqrt(np.sum(np.square(Y[:,(k-1)*desc_mat_sz:k*desc_mat_sz] - X[:,(k-1)*desc_mat_sz:k*desc_mat_sz]), axis=1))
    for k in range(2, desc_scales + 1):
        # break
        # D += pdist(Y[:,(k-1)*desc_mat_sz:k*desc_mat_sz], X[:,(k-1)*desc_mat_sz:k*desc_mat_sz], 'euclidean')
        a = Y[:, (k - 1) * desc_mat_sz:k * desc_mat_sz]
        b = X[:, (k - 1) * desc_mat_sz:k * desc_mat_sz]
        D = D + cdist(a, b, metric='euclidean')
        # D += np.sqrt(np.sum(np.square(Y[:,(k-1)*desc_mat_sz:k*desc_mat_sz] - X[:,(k-1)*desc_mat_sz:k*desc_mat_sz]), axis=1))
    D /= desc_scales
    I = np.argmin(D, axis=0)
    D = np.min(D, axis=0)

    # Create array with point ID1, ID2, distance
    Array = np.zeros((len(I), 3))
    Array[:, 0] = np.arange(len(I))
    Array[:, 1] = I
    Array[:, 2] = D

    # Sort by distance ascending
    Array = Array[Array[:, 2].argsort()]

    # Cut off after max_pairs
    Array = Array[:max_pairs, :]

    # Filter out d > max_d
    filter = Array[:, 2] <= max_d
    Array = Array[filter, :]

    # Prepare output
    d_pairs = Array[:, 0:2].astype(int)
    d_dist = Array[:, 2]
    gt_dist = np.zeros((Array.shape[0],))

    # Calculate ground truth distance
    # for k in range(len(d_dist)):
    #     A = [kp[index1]['gt']['x'][d_pairs[k, 0]], kp[index1]['gt']['y'][d_pairs[k, 0]],
    #          kp[index1]['gt']['z'][d_pairs[k, 0]]]
    #     B = [kp[index2]['gt']['x'][d_pairs[k, 1]], kp[index2]['gt']['y'][d_pairs[k, 1]],
    #          kp[index2]['gt']['z'][d_pairs[k, 1]]]
    #     gt_dist[k] = np.linalg.norm(np.array(A) - np.array(B))
    if print_flag:
        print('Point 1, point 2, distance (feature space), distance (ground truth)')
        # print(np.concatenate((Array, gt_dist.reshape(-1, 1)), axis=1))
        print(np.concatenate(Array))
        print('')

    return d_pairs, d_dist