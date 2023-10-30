# Author: suhaot
# Date: 2023/9/18
# Description: generate_rand_pose
import numpy as np
import os


# 对点云添加随机的旋转和平移
def random_pose_transform(pc, kp, seed):
    """
    Apply a random pose transformation to a set of 3D points.
    :param points: Nx3 numpy array of points.
    :return: Transformed Nx3 numpy array of points.
    """
    # Generate a random rotation matrix.
    np.random.seed(seed)
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.uniform(-1, 1, 3)
    axis = axis / np.linalg.norm(axis)
    rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)

    # Generate a random translation vector.
    translation_vector = np.random.uniform(-1, 1, 3)

    # Apply the transformation.
    transformed_pc_points = np.dot(pc, rotation_matrix.T) + translation_vector
    transformed_kp_points = np.dot(kp, rotation_matrix.T) + translation_vector
    # 获得变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector
    return transformed_pc_points, transformed_kp_points, transform_matrix


def rotation_matrix_from_axis_angle(axis, angle):
    """
    Generate a rotation matrix given an axis and an angle.
    """
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return rotation_matrix


def apply_transform(pcpath, kppath, seed):
    pcdata = np.load(pcpath)
    kpdata = np.load(kppath)
    pc_coordinates = pcdata[:, :3]
    kp_coordinates = kpdata[:, :3]
    transformed_pc_coordinates, transformed_kp_coordinates, transform_matrix = random_pose_transform(pc_coordinates,
                                                                                                     kp_coordinates,seed)
    pcdata[:, :3] = transformed_pc_coordinates
    kpdata[:, :3] = transformed_kp_coordinates
    return pcdata, kpdata, transform_matrix
