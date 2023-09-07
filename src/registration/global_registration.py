import numpy as np
import open3d as o3d
import open3d as o3d
import networkx as nx
import pickle
import frag_relation_graph as frg
import os
import matplotlib.pyplot as plt

folder_path = '../../data/TUWien/brick/'
ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]

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


def register_point_clouds(source, target, threshold=0.02, trans_init=None):
    """
    使用ICP配准两个点云。

    参数:
    - source: 源点云 (需要被变换的点云)
    - target: 目标点云
    - threshold: 用于寻找对应关系的距离阈值
    - trans_init: 初始变换矩阵

    返回:
    - 点云配准后的变换矩阵
    """

    # 如果未提供初始变换矩阵，使用单位矩阵作为初始变换
    if trans_init is None:
        trans_init = np.eye(4)

    # 使用ICP进行点云配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # 返回最优变换矩阵
    return reg_p2p.transformation


def random_transformation():
    """生成随机的旋转和平移矩阵"""

    # 生成随机旋转矩阵
    rot = o3d.geometry.get_rotation_matrix_from_xyz(
        np.random.uniform(low=-np.pi, high=np.pi, size=3)
    )

    # 生成随机平移向量
    trans = np.random.uniform(low=-0.5, high=0.5, size=3)

    # 合成4x4的变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    return transform


# 对点云添加随机的旋转和平移
def perturb_pointcloud(point_cloud):
    """给点云添加随机旋转和平移"""
    transform = random_transformation()
    point_cloud.transform(transform)
    return point_cloud


# 加载之前保存的数据
graph, ply_files_dict = frg.create_graph(folder_path)

# 使用DFS创建生成树
dfs_tree = nx.dfs_tree(graph)

# 创建一个图形实例
plt.figure(figsize=(12, 8))

# 使用networkx的draw函数绘制生成树
pos = nx.spring_layout(dfs_tree)  # 使用spring_layout布局，你也可以更改为其他布局
nx.draw(dfs_tree, pos, with_labels=True, node_color="skyblue", node_size=1500, width=2, edge_color="gray", font_size=15)

# 添加标题和显示图形
plt.title("DFS Tree Visualization")
plt.show()


# 初始化一个字典来存储变换矩阵
transformation_matrices = {}

# 初始化一个字典来存储节点名字和随机旋转后的ply对象
pc_dict = {}
combined_rand_point_cloud = o3d.geometry.PointCloud()
for file in ply_files:
    full_path = os.path.join(folder_path, file)
    point_cloud = o3d.io.read_point_cloud(full_path)
    point_cloud = perturb_pointcloud(point_cloud)
    pc_dict[os.path.basename(file)] = point_cloud
    combined_rand_point_cloud += point_cloud

o3d.visualization.draw_geometries([combined_rand_point_cloud])

# 遍历生成树的每条边，执行点云配准
for edge in dfs_tree.edges():
    node1, node2 = edge

    # 读取两个点云
    source = pc_dict[node1]
    target = pc_dict[node2]

    # 提取关键点，这里假设你有一个函数 `extract_keypoints` 来执行此操作
    keypoints_source, keypoints_target = extract_keypoints(source, target, pc_dict, graph)

    # 使用点云配准计算变换矩阵
    # 这里假设你有一个函数 `register_point_clouds` 来完成这个工作
    transformation = register_point_clouds(keypoints_source, keypoints_target)

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
