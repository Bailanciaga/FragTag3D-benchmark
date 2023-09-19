import shutil

import numpy as np
import open3d as o3d
import open3d as o3d
import networkx as nx
import pickle
import frag_relation_graph as frg
import os
import matplotlib
import matplotlib.pyplot as plt
from src.dataprocess import open3d_resample,extract_keypoints_SD
from src.registration import get_descriptor_pairs
from src.dataprocess import generate_rand_pose
from src.registration import registration_use_ransac

inputpath = '../../data/TUWien/brick/'
# matplotlib.use('Qt5Agg')


def dowmSample_and_extract(inputpath, params=None):
    print("DownSampling....")
    output_path = open3d_resample.collect_pointclouds(inputpath)

    if params == None:
        n_keypoints = 512
        keypoint_radius = 0.04
        r_vals = [0.08, 0.09, 0.10, 0.11, 0.12]
        nms_rad = 0.04
    else:
        n_keypoints = params["n_keypoints"]
        keypoint_radius = params["keypoint_radius"]
        r_vals = params["r_vals"]
        nms_rad = params["nms_rad"]
    print("Extracting Key Points...")
    # TODO change the file address to windows format
    extract_keypoints_SD.extract_key_point_by_dir(output_path, n_keypoints, keypoint_radius, r_vals, nms_rad, useflags=True)
    return output_path

def extract_by_color(point_cloud, target_color):
    colors = np.asarray(point_cloud.colors)
    mask = np.all(np.isclose(colors * 255, target_color, atol=1e-5), axis=1)
    return point_cloud.select_by_index(np.where(mask)[0])


def extract_keypoints(pc1, pc2, pc_dict, graph):

    # 获取点云的名字（即图的节点名）
    node1 = next((k for k, v in pc_dict.items() if v == pc1), None)
    node2 = next((k for k, v in pc_dict.items() if v == pc2), None)

    # 根据点云的名字在图中找到对应的边颜色
    edge_color = graph[node1][node2]['color']

    # 从每个点云中提取关键点
    keypoints_pc1 = extract_by_color(pc1, edge_color)
    keypoints_pc2 = extract_by_color(pc2, edge_color)

    return keypoints_pc1, keypoints_pc2


# def register_point_clouds(source, target, threshold=0.02, trans_init=None):
#     """
#     使用ICP配准两个点云。
#
#     参数:
#     - source: 源点云 (需要被变换的点云)
#     - target: 目标点云
#     - threshold: 用于寻找对应关系的距离阈值
#     - trans_init: 初始变换矩阵
#
#     返回:
#     - 点云配准后的变换矩阵
#     """
#
#     # 如果未提供初始变换矩阵，使用单位矩阵作为初始变换
#     if trans_init is None:
#         trans_init = np.eye(4)
#
#     # 使用ICP进行点云配准
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         source, target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     )
#
#     # 返回最优变换矩阵
#     return reg_p2p.transformation

def npy_visualization(pc_data, kp_data):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_data[:, :3])
    pc.normals = o3d.utility.Vector3dVector(pc_data[:, 3:6])
    pc.colors = o3d.utility.Vector3dVector(np.ones((len(pc_data), 3)))

    kp = o3d.geometry.PointCloud()
    kp.points = o3d.utility.Vector3dVector(kp_data[:, :3])
    kp.normals = o3d.utility.Vector3dVector(kp_data[:, 3:6])
    kp.colors = o3d.utility.Vector3dVector(kp_data[:, 6:9])

    # 创建可视化器并添加两个点云
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 2  # 设置点大小

    vis.add_geometry(pc)
    vis.add_geometry(kp)
    vis.run()
    vis.destroy_window()


# 加载之前保存的数据
graph, ply_files_dict, edge_color_map = frg.create_graph(inputpath)

# 使用DFS创建生成树
dfs_tree = nx.dfs_tree(graph)

# 创建一个图形实例
plt.figure(figsize=(12, 8))

# 使用networkx的draw函数绘制生成树
pos = nx.spring_layout(dfs_tree)  # 使用spring_layout布局，你也可以更改为其他布局
nx.draw(dfs_tree, pos, with_labels=True, node_color="skyblue", node_size=1500, width=4, edge_color="gray", font_size=15)

# 添加标题和显示图形
plt.title("DFS Tree Visualization")
plt.show()

#下采样、转换格式并提取关键点
# outputpath = dowmSample_and_extract(inputpath)

pc_files = [f for f in os.listdir('/home/suhaot/PycharmProjects/FragTag3D/data/TUWien/brick/npy') if f.endswith('.npy')]
kp_files = [f for f in os.listdir(os.path.join('/home/suhaot/PycharmProjects/FragTag3D/data/TUWien/brick/npy', 'keypoints')) if f.endswith('.npy')]

# 初始化一个字典来存储随机变换矩阵 以便后续验证用
transformation_random_matrices = {}

# 初始化一个字典来存储节点名字和随机旋转后的ply对象
pcrp_dict = {}
kprp_dict = {}

combined_rand_pointcloud = np.empty((0,9))
combined_rand_keypoint = np.empty((0,9))
for file in pc_files:
    pc_path = os.path.join('/home/suhaot/PycharmProjects/FragTag3D/data/TUWien/brick/npy', file)
    kp_path = os.path.join('/home/suhaot/PycharmProjects/FragTag3D/data/TUWien/brick/npy', 'keypoints', file)
    pcrp, kprp, transform_matrix = generate_rand_pose.apply_transform(pc_path, kp_path)
    pcrp_dict[os.path.basename(file).split('.')[0]] = pcrp
    kprp_dict[os.path.basename(file).split('.')[0]] = kprp
    combined_rand_pointcloud = np.vstack((combined_rand_pointcloud, pcrp))
    combined_rand_keypoint = np.vstack((combined_rand_keypoint, kprp[:, :9]))

npy_visualization(combined_rand_pointcloud, combined_rand_keypoint)

transformation_matrices = {}
# 遍历生成树的每条边，执行点云配准
for edge in dfs_tree.edges():
    node1, node2 = edge

    # 读取两个点云
    kp1 = kprp_dict[node1]
    kp2 = kprp_dict[node2]

    d_pairs, d_dist = get_descriptor_pairs.get_descriptor_pairs_classical(kp1, kp2)
    # 使用点云配准计算变换矩阵
    R, T = registration_use_ransac.assy_use_ransac(kp1, kp2, d_pairs, d_dist)

    # 保存变换矩阵
    transformation_matrices[edge] = transformation

# 你现在有一个变换矩阵的字典，可以使用这些矩阵来结合所有的点云

# 初始化combined_point_cloud为DFS树的第一个点云
combined_point_cloud = pc_dict[next(iter(dfs_tree.nodes()))]  # 从第一个点开始

# 遍历DFS的每一条边
for edge in nx.dfs_edges(dfs_tree):
    node1, node2 = edge
    point_cloud_to_add = pc_dict[node2]

    # 获取和应用对应的变换矩阵
    if edge in transformation_matrices:
        matrix = transformation_matrices[edge]
        point_cloud_to_add.transform(matrix)

    combined_point_cloud += point_cloud_to_add

# 现在，combined_point_cloud是结合了所有点云的结果

o3d.visualization.draw_geometries([combined_point_cloud])
