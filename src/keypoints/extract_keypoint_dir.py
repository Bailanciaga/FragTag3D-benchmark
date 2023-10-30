# Author: suhaot
# Date: 2023/10/5
# Description: extract_keypoint_dir
import os
import numpy as np
from src.keypoints.extract_keypoints_iss import get_iss_keypoints
from src.keypoints.extract_keypoints_pillar import get_pillar_keypoints
from src.keypoints.extract_keypoints_harris import get_harris_keypoints
from src.keypoints.extract_keypoints_SD import get_SD_keypoints

npoints = 256


def extract_key_point_by_dir(dataset_dir, method, mode):
    fragments = os.listdir(dataset_dir)
    fragments = [x for x in fragments if x.endswith(".npy")]
    print("***************************Keypoint & Descriptors Extraction***************************")
    for fragment_name in fragments:
        print(f"Fragment: {fragment_name}")
        fragment_path = os.path.join(dataset_dir, fragment_name)
        keypoints_dir = os.path.join(dataset_dir, "keypoints_" + method)
        if not os.path.exists(keypoints_dir):
            os.mkdir(keypoints_dir)
        fragment = np.load(fragment_path)
        colors = fragment[:, 6:9]
        output_path = os.path.join(keypoints_dir, fragment_name)
        fragment_kp = np.empty((0, 54))
        if not mode == 3:
            # 创建一个布尔掩码，检查colors中的每一行是否不与[255, 255, 255]相匹配
            mask = ~np.all(colors == [1, 1, 1], axis=1)
            # 使用掩码筛选vertices和normals
            fragment = fragment[mask]
            normals = fragment[:, 3:6]
            if mode == 1:
                unique_colors = np.unique(np.int64(np.round(fragment[:, 6:9] * 255)), axis=0)
                for color in unique_colors:
                    print("-------Extracting fracture face color is (" + ', '.join(map(str, color)) + ")")  # 打印当前颜色
                    specific_color_mask = np.all(np.isclose(fragment[:, 6:9] * 255, color, atol=1e-5), axis=1)
                    specific_color_fragment_face = fragment[specific_color_mask]
                    if method == 'iss':
                        output_matrix = get_iss_keypoints(specific_color_fragment_face)
                    elif method == 'pillar':
                        output_matrix = get_pillar_keypoints(specific_color_fragment_face, 12, npoints)
                    elif method == 'harris':
                        output_matrix = get_harris_keypoints(specific_color_fragment_face, npoints)
                    elif method == 'SD':
                        output_matrix = get_SD_keypoints(specific_color_fragment_face, normals, npoints, r=0.05)
                    fragment_kp = np.vstack((fragment_kp, output_matrix))
                np.save(output_path, fragment_kp)
            if mode == 2:
                if method == 'iss':
                    output_matrix = get_iss_keypoints(fragment)
                elif method == 'pillar':
                    output_matrix = get_pillar_keypoints(fragment, 12, npoints)
                elif method == 'harris':
                    output_matrix = get_harris_keypoints(fragment, npoints)
                elif method == 'SD':
                    output_matrix = get_SD_keypoints(fragment, normals, npoints, r=0.05)
                np.save(output_path, output_matrix)
        else:
            fragment = fragment
            normals = fragment[:, 3:6]
            if method == 'iss':
                np.save(output_path, get_iss_keypoints(fragment))
            elif method == 'pillar':
                np.save(output_path, get_pillar_keypoints(fragment, 12, npoints, output_path))
            elif method == 'harris':
                np.save(output_path, get_harris_keypoints(fragment, npoints))
            elif method == "SD":
                output_matrix = get_SD_keypoints(fragment, normals, npoints, r=0.05)
