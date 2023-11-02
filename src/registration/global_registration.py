import shutil

import numpy as np
import open3d as o3d
import networkx as nx
import frag_relation_graph as frg
import os
import re
import matplotlib.pyplot as plt
from src.dataprocess import open3d_resample
from src.keypoints import extract_keypoints_SDplus
from src.registration import get_descriptor_pairs
from src.dataprocess import generate_rand_pose
from src.registration import registration_use_ransac
from src.keypoints import extract_keypoint_dir
from src.registration.tools import plot_kp_pairs
from src.evaluation import charmfer_dist
from src.evaluation import normal_consistency
from src.evaluation import hausdorff_dist
from src.registration.tools import matrix_to_ply

inputpath = '../../data/TUWien/41272_5_seed_1/'
# inputpath = '../../data/thingi10k/40179/40179_5_seed_0/'
# inputpath = '../../data/thingi10k/125760/1/'
# inputpath = '../../data/thingi10k/41272_5_seed_1/'
mode = 3  # 1:提取重叠面关键点 2：提取断裂面关键点 3：提取全局关键点
method = 'SD+'  # 选择关键点提取算法：SD、harris、iss、pillar、SD+


def dowmSample_and_extract(inputpath, method, params=None):
    print("DownSampling....")
    if not mode == 1:
        output_path = open3d_resample.collect_pointclouds(inputpath, every_k_points=5)
    else:
        output_path = open3d_resample.collect_pointclouds(inputpath, every_k_points=2)
    print("Extracting Key Points...")
    if method == 'SD+':
        extract_keypoints_SDplus.extract_key_point_by_dir(output_path, mode)
    else:
        extract_keypoint_dir.extract_key_point_by_dir(output_path, method, mode)

    return output_path


def extract_by_color(pc, target_color):
    colors = pc[:, 6:9]
    mask = np.all(np.isclose(colors * 255, target_color, atol=1e-5), axis=1)
    return pc[mask]


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
graph, ply_files_dict, edge_color_map, _, _ = frg.create_graph(inputpath)

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


# 下采样、转换格式并提取关键点
outputpath = dowmSample_and_extract(inputpath, method)

pc_files = [f for f in os.listdir(outputpath) if
            f.endswith('.npy')]
kp_files = [f for f in os.listdir(
    os.path.join(outputpath,
                 'keypoints_' + method)) if f.endswith('.npy')]

# 创建配准之前的随机位姿存储路径
random_path = os.path.join(outputpath, 'ramdom')
if not os.path.exists(random_path):
    os.mkdir(random_path)
else:
    shutil.rmtree(random_path)
    os.mkdir(random_path)
# 创建配准完毕的存储路径
result_path = os.path.join(outputpath,
                           'keypoints_' + method, 'result')

if not os.path.exists(result_path):
    os.mkdir(result_path)
else:
    shutil.rmtree(result_path)
    os.mkdir(result_path)

# 初始化一个字典来存储节点名字和随机旋转后的ply对象
pcrp_dict = {}
kprp_dict = {}

combined_rand_pointcloud = np.empty((0, 9))
combined_rand_keypoint = np.empty((0, 9))
for file in pc_files:
    pc_path = os.path.join(outputpath, file)
    kp_path = os.path.join(outputpath,
                           'keypoints_' + method, file)
    # 随机位姿所用的随机数种子为每个文件名称最后一位数字
    seed = int(re.findall(r'\d+', file)[-1]) if re.findall(r'\d+', file) else None
    pcrp, kprp, transform_matrix = generate_rand_pose.apply_transform(pc_path, kp_path, seed)
    pcrp_dict[os.path.basename(file).split('.')[0]] = pcrp
    kprp_dict[os.path.basename(file).split('.')[0]] = kprp
    matrix_to_ply.matrix2ply(pcrp, os.path.join(random_path, file[:-4] + '.ply'))
    combined_rand_pointcloud = np.vstack((combined_rand_pointcloud, pcrp))
    combined_rand_keypoint = np.vstack((combined_rand_keypoint, kprp[:, :9]))

# npy_visualization(combined_rand_pointcloud, combined_rand_keypoint)

transformation_matrices = {}

max_attempts = 10 # 设置最大尝试次数，防止无限循环

previous_start_nodes = set()
# 记录下次开始的节点，初始为空
next_start_node = None

for attempt in range(max_attempts):

    # 使用上一次循环中的node2作为起点，或从图中随机选择一个开始节点
    if next_start_node:
        start_node = next_start_node
        next_start_node = None
    else:
        start_node = np.random.choice(graph.nodes())
        while start_node in previous_start_nodes and len(previous_start_nodes) < len(graph.nodes()):
            start_node = np.random.choice(graph.nodes())
    previous_start_nodes.add(start_node)

    dfs_tree = nx.dfs_tree(graph, source=start_node)

    # 初始化combined_point_cloud为DFS树的第一个点云
    combined_pointcloud = pcrp_dict[next(iter(dfs_tree.nodes()))]
    combined_keypoint = kprp_dict[next(iter(dfs_tree.nodes()))]

    nan_detected = False  # 设置一个标志来检测是否出现了nan

    print("***************************Registration***************************")
    # 遍历生成树的每条边，执行点云配准
    for edge in dfs_tree.edges():
        node1, node2 = edge
        sorted_edge = tuple(sorted(edge))

        color = edge_color_map[sorted_edge]
        # 读取两个点云
        kp1 = kprp_dict[node1]
        kp2 = kprp_dict[node2]

        pc1 = pcrp_dict[node1]
        pc2 = pcrp_dict[node2]

        if mode == 1:
            kp1_colored = extract_by_color(kp1, color)
            kp2_colored = extract_by_color(kp2, color)

            print("-------Fragment : " + node1 + " & " + node2 + " (" + ', '.join(map(str, color)) + ")")

            d_pairs, d_dist = get_descriptor_pairs.get_descriptor_pairs_classical(kp2_colored, kp1_colored)
            # 使用点云配准计算变换矩阵
            R, T = registration_use_ransac.assy_use_ransac(kp2_colored, kp1_colored, d_pairs, d_dist)

            # 检测R和T中是否有nan
            if not np.isnan(R).any() and not np.isnan(T).any():
                # 如果R和T中没有nan，则从图中移除node1及其关联的边
                graph.remove_edge(*edge)

                plot_kp_pairs.kp_plot(kp1_colored, kp2_colored, pc1, pc2, d_pairs)

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
                matrix_to_ply.matrix2ply(pc1, os.path.join(result_path, node1 + '.ply'))
                matrix_to_ply.matrix2ply(pc2, os.path.join(result_path, node2 + '.ply'))
                kp2_colored = extract_by_color(kp2, color)
                plot_kp_pairs.kp_plot(kp1_colored, kp2_colored, pc1, pc2, d_pairs)

            else:
                next_start_node = node2
                matrix_to_ply.matrix2ply(pc1, os.path.join(result_path, node1 + '.ply'))
                matrix_to_ply.matrix2ply(pc2, os.path.join(result_path, node2 + '.ply'))
                print("Calculation of R and T failed, rebuilding the generation tree.......")
                break  # 跳出内部循环

        else:
            print("-------Fragment : " + node1 + " & " + node2)

            d_pairs, d_dist = get_descriptor_pairs.get_descriptor_pairs_classical(kp2, kp1)
            # 使用点云配准计算变换矩阵
            R, T = registration_use_ransac.assy_use_ransac(kp2, kp1, d_pairs, d_dist)

            plot_kp_pairs.kp_plot(kp1, kp2, pc1, pc2, d_pairs)
            # 检测R和T中是否有nan
            if not np.isnan(R).any() and not np.isnan(T).any():
                # 如果R和T中没有nan，则从图中移除node1及其关联的边
                graph.remove_node(node1)

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
                matrix_to_ply.matrix2ply(pc1, os.path.join(result_path, node1 + '.ply'))
                matrix_to_ply.matrix2ply(pc2, os.path.join(result_path, node2 + '.ply'))
                plot_kp_pairs.kp_plot(kp1, kp2, pc1, pc2, d_pairs)
            else:
                next_start_node = node2
                plot_kp_pairs.kp_plot(kp1, kp2, pc1, pc2, d_pairs)
                matrix_to_ply.matrix2ply(pc1, os.path.join(result_path, node1 + '.ply'))
                matrix_to_ply.matrix2ply(pc2, os.path.join(result_path, node2 + '.ply'))
                print("Calculation of R and T failed, rebuilding the generation tree.......")
                break  # 跳出内部循环
        # 计算得分
        CD_score = charmfer_dist.chamfer_distance(kp1[:, :3], kp2[:, :3])
        NC_score = normal_consistency.normal_consistency(kp1[:, :3], kp1[:, 3:6], kp2[:, :3], kp2[:, 3:6])
        HD_score = hausdorff_dist.hausdorff_dist(kp1[:, :3], kp2[:, :3])

        print(node1 + " & " + node2 + " CD score is: " + str(CD_score))
        print(node1 + " & " + node2 + " NC score is: " + str(NC_score))
        print(node1 + " & " + node2 + " HD score is: " + str(HD_score))

        combined_keypoint = np.vstack((combined_keypoint, kprp_dict[node2]))
        combined_pointcloud = np.vstack((combined_pointcloud, pcrp_dict[node2]))

    if not next_start_node:  # 如果遍历完毕整个图，则跳出主循环
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

    if attempt == 20:
        print("Registration failed   :(   ")

npy_visualization(combined_pointcloud, combined_keypoint)
