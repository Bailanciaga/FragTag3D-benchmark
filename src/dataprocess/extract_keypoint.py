# Author: suhaot
# Date: 2023/9/5
# Description: extract_keypoint
import numpy as np
import open3d as o3d

def extract_keypoints(ply1_path, ply2_path, ply_files_dict, graph):
    # 读取点云
    pc1 = o3d.io.read_point_cloud(ply1_path)
    pc2 = o3d.io.read_point_cloud(ply2_path)

    # 获取点云的名字（即图的节点名）
    node1 = os.path.basename(ply1_path)
    node2 = os.path.basename(ply2_path)

    # 根据点云的名字在图中找到对应的边颜色
    edge_color = graph[node1][node2]['color']

    # 定义一个函数提取与特定颜色匹配的点
    def extract_color_points(point_cloud, color):
        colors = np.asarray(point_cloud.colors)
        # 这里我们假设颜色匹配精确到小数点后3位
        mask = np.all(np.isclose(colors, color, atol=0.003), axis=1)
        return point_cloud.select_by_index(np.where(mask)[0])

    # 从每个点云中提取关键点
    keypoints_pc1 = extract_color_points(pc1, edge_color)
    keypoints_pc2 = extract_color_points(pc2, edge_color)

    return keypoints_pc1, keypoints_pc2
