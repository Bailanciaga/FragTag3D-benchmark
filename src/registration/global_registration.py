import numpy as np
import open3d as o3d
import networkx as nx
import frag_relation_graph as frg
import os
import matplotlib.pyplot as plt
from src.dataprocess import open3d_resample
from src.keypoints import extract_keypoints_SD
from src.registration import get_descriptor_pairs
from src.dataprocess import generate_rand_pose
from src.registration import registration_use_ransac
from src.keypoints import extract_keypoint_dir
from src.registration.tools import plot_kp_pairs
# inputpath = '../../data/TUWien/brick/'
# inputpath = '../../data/thingi10k/41272_5_seed_1/'
inputpath = '../../data/thingi10k/37627/37627_5_seed_0/'


def dowmSample_and_extract(inputpath, method, params=None):
    print("DownSampling....")
    output_path = open3d_resample.collect_pointclouds(inputpath)

    if params == None:
        n_keypoints = 512
        keypoint_radius = 0.06
        r_vals = [0.06, 0.07, 0.08, 0.10, 0.12]
        nms_rad = 0.06
    else:
        n_keypoints = params["n_keypoints"]
        keypoint_radius = params["keypoint_radius"]
        r_vals = params["r_vals"]
        nms_rad = params["nms_rad"]
    print("Extracting Key Points...")
    # TODO change the file address to windows format
    if method == 'SD':
        extract_keypoints_SD.extract_key_point_by_dir(output_path, n_keypoints, keypoint_radius, r_vals, nms_rad, useflags=True)
    else:
        extract_keypoint_dir.extract_key_point_by_dir(output_path, method, useflags=True)

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

def transform_point_cloud(pc, R, T):
    # 构建变换矩阵
    transform = np.eye(4)
    transform[0:3, 0:3] = R
    transform[0:3, 3] = T

    # 将点云转化为齐次坐标
    homogenous_pc = np.ones((pc.shape[0], 4))
    homogenous_pc[:, 0:3] = pc

    # 应用变换
    transformed_pc = homogenous_pc.dot(transform.T)
    return transformed_pc[:, 0:3]

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

# 选择关键点提取算法：SD、harris、iss、pillar
method = 'SD'
#下采样、转换格式并提取关键点
outputpath = dowmSample_and_extract(inputpath, method)

pc_files = [f for f in os.listdir(outputpath) if f.endswith('.npy')]
kp_files = [f for f in os.listdir(os.path.join(outputpath, 'keypoints_' + method)) if f.endswith('.npy')]

# 初始化一个字典来存储节点名字和随机旋转后的ply对象
pcrp_dict = {}
kprp_dict = {}

combined_rand_pointcloud = np.empty((0,9))
combined_rand_keypoint = np.empty((0,9))
for file in pc_files:
    pc_path = os.path.join(outputpath, file)
    kp_path = os.path.join(outputpath, 'keypoints_' + method, file)
    pcrp, kprp, transform_matrix = generate_rand_pose.apply_transform(pc_path, kp_path)
    pcrp_dict[os.path.basename(file).split('.')[0]] = pcrp
    kprp_dict[os.path.basename(file).split('.')[0]] = kprp
    combined_rand_pointcloud = np.vstack((combined_rand_pointcloud, pcrp))
    combined_rand_keypoint = np.vstack((combined_rand_keypoint, kprp[:, :9]))

npy_visualization(combined_rand_pointcloud, combined_rand_keypoint)

transformation_matrices = {}

max_attempts = 20   # 设置最大尝试次数，防止无限循环

previous_start_nodes = set()

for attempt in range(max_attempts):
    start_node = np.random.choice(graph.nodes())
    while start_node in previous_start_nodes and len(previous_start_nodes) < len(graph.nodes()):
        start_node = np.random.choice(graph.nodes())

    previous_start_nodes.add(start_node)

    dfs_tree = nx.dfs_tree(graph, source=start_node)

    # 初始化combined_point_cloud为DFS树的第一个点云
    combined_pointcloud = pcrp_dict[next(iter(dfs_tree.nodes()))]
    combined_keypoint = kprp_dict[next(iter(dfs_tree.nodes()))]

    nan_detected = False  # 设置一个标志来检测是否出现了nan

    # 遍历生成树的每条边，执行点云配准
    for edge in dfs_tree.edges():
        node1, node2 = edge

        # 读取两个点云
        kp1 = kprp_dict[node1]
        kp2 = kprp_dict[node2]

        pc1 = pcrp_dict[node1]
        pc2 = pcrp_dict[node2]

        d_pairs, d_dist = get_descriptor_pairs.get_descriptor_pairs_classical(kp2, kp1)
        # 使用点云配准计算变换矩阵
        R, T = registration_use_ransac.assy_use_ransac(kp2, kp1, d_pairs, d_dist)


        # 检测R和T中是否有nan
        if np.isnan(R).any() or np.isnan(T).any():
            nan_detected = True
            print("计算R、T失败，重新构建生成树中…………")
            break  # 跳出内部循环

        plot_kp_pairs.kp_plot(kp1, kp2, pc1, pc2, d_pairs)

        # 更新node2的关键点
        for i in range(len(kp2)):
            coord = np.array(kp2[i][:3]).reshape(3, 1)  # 取出前三维的坐标
            new_coord = np.dot(R, coord) + T.reshape(3, 1)
            kp2[i][:3] = new_coord.ravel()

        kprp_dict[node2] = kp2  # 更新字典

        # 更新pcrp_dict。相似的方式可以用于pcrp_dict
        for i in range(len(pc2)):
            coord = np.array(pc2[i][:3]).reshape(3, 1)  # 取出前三维的坐标
            new_coord = np.dot(R, coord) + T.reshape(3, 1)
            pc2[i][:3] = new_coord.ravel()

        pcrp_dict[node2] = pc2  # 更新字典

        plot_kp_pairs.kp_plot(kp1, kp2, pc1, pc2, d_pairs)

        combined_keypoint = np.vstack((combined_keypoint, kprp_dict[node2]))
        combined_pointcloud = np.vstack((combined_pointcloud, pcrp_dict[node2]))

    if not nan_detected:  # 如果没有检测到nan，则跳出主循环
        break

    # 创建一个图形实例
    plt.figure(figsize=(12, 8))

    # 使用networkx的draw函数绘制生成树
    pos = nx.spring_layout(dfs_tree)  # 使用spring_layout布局，你也可以更改为其他布局
    nx.draw(dfs_tree, pos, with_labels=True, node_color="skyblue", node_size=1500, width=4, edge_color="gray",
            font_size=15)

    # 添加标题和显示图形
    plt.title("DFS Tree Visualization")
    plt.show()

npy_visualization(combined_pointcloud, combined_keypoint)


