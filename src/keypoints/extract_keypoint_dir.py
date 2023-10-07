# Author: suhaot
# Date: 2023/10/5
# Description: extract_keypoint_dir
import os
import numpy as np
from src.keypoints.extract_keypoints_iss import get_iss_keypoints
from src.keypoints.extract_keypoints_pillar import get_pillar_keypoints
from src.keypoints.extract_keypoints_harris import get_harris_keypoints

npoints = 512


def extract_key_point_by_dir(dataset_dir, method, useflags):
    fragments = os.listdir(dataset_dir)
    fragments = [x for x in fragments if x.endswith(".npy")]

    for fragment_name in fragments:
        print(f"Fragment: {fragment_name}")
        fragment_path = os.path.join(dataset_dir, fragment_name)
        keypoints_dir = os.path.join(dataset_dir, "keypoints_" + method)
        if not os.path.exists(keypoints_dir):
            os.mkdir(keypoints_dir)

        fragment = np.load(fragment_path)
        colors = fragment[:, 6:9]
        if useflags:
            # 创建一个布尔掩码，检查colors中的每一行是否不与[255, 255, 255]相匹配
            mask = ~np.all(colors == [1, 1, 1], axis=1)
            # 使用掩码筛选vertices和normals
            fragment = fragment[mask]
        else:
            fragment = fragment

        output_path = os.path.join(keypoints_dir, fragment_name)

        if method == 'iss':
            get_iss_keypoints(fragment, output_path)
        elif method == 'pillar':
            get_pillar_keypoints(fragment, 12, npoints, output_path)
        elif method == 'harris':
            get_harris_keypoints(fragment, npoints, output_path)