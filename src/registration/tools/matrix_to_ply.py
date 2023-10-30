# Author: suhaot
# Date: 2023/10/27
# Description: matrix_to_ply
import open3d as o3d
import numpy as np

def matrix2ply(pc1, filename):
    # 创建一个Pointcloud对象
    pcd = o3d.geometry.PointCloud()

    # 设置点云对象的坐标
    pcd.points = o3d.utility.Vector3dVector(pc1[:, :3])

    # 设置点云对象的法线
    pcd.normals = o3d.utility.Vector3dVector(pc1[:, 3:6])

    # 设置点云对象的颜色
    # pcd.colors = o3d.utility.Vector3dVector(pc1[:, 6:] / 255)  # 注意：颜色值应在[0,1]范围内

    # 将点云对象写入PLY文件
    o3d.io.write_point_cloud(filename, pcd)
