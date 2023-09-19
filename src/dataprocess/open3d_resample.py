import shutil

import numpy as np
import open3d as o3d
import os
import plotly.graph_objects as go


"""
Script to convert Pointclouds with high density
Downsamples pointcloud, saves a html plot and generates a csv with neighboring information

Folder structure:
    '-root
        '-npy
            '-OUTPUT FOLDERS
        '-obj
            '-INPUT FOLDERS

INPUT FORMAT: .obj pointcloud with normals
OUTPUT FORMAT: .npy array with [x y z nx ny nz] values

Loops over all subfolders of input_folder and outputs to same folder in output_folder
Fragments with <1000 points after downsampling won't get downsampled
"""

# root = "./"
# #input_folder = f"{root}/obj"  # should contain pointclouds in .obj format
# #output_folder = f"{root}/npy"  # will be generated for output
# input_folder = f"/home/suhaot/PycharmProjects/3d-fracture-reassembly/dataset/TUWien"
# output_folder = f"/home/suhaot/PycharmProjects/3d-fracture-reassembly/dataset_kp/TUWien_seg"
sample_size = 0.01


# def convert_pointclouds(input_folder, output_folder):
#     """
#     Entry point to loop over subfolders
#     """
#     folders = [
#         x for x in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, x))
#     ]
#     print(f"Folders: {folders}")
#     for dir in folders:
#         collect_pointclouds(dir, input_folder, output_folder)


def collect_pointclouds(inputpath):
    """
    convert and save all pointcloud data from a single folder
    """


    pc_dict = {}
    b_max = np.zeros((3,))
    b_min = np.zeros((3,))
    print(f"Object: {dir}")
    for x in os.listdir(inputpath):
        if x.endswith('.obj'):
            file = x[:-4]
            mesh = o3d.io.read_triangle_mesh(os.path.join(inputpath, f"{file}.obj"))
            pc = o3d.geometry.PointCloud()
            pc.points = mesh.vertices
            pc.normals = mesh.vertex_normals
            b_max = np.max([pc.get_max_bound(), b_max], axis=0)
            b_min = np.min([pc.get_min_bound(), b_min], axis=0)
            pc_dict[file] = pc
        if x.endswith('.ply'):
            file = x[:-4]
            pc = o3d.io.read_point_cloud(os.path.join(inputpath, f"{file}.ply"))
            color = np.asarray(pc.colors)
            b_max = np.max([pc.get_max_bound(), b_max], axis=0)
            b_min = np.min([pc.get_min_bound(), b_min], axis=0)
            pc_dict[file] = pc
        if x.endswith('.pcd'):
            file = x[:-4]
            pc = o3d.io.read_point_cloud(os.path.join(inputpath, f"{file}.pcd"))
            b_max = np.max([pc.get_max_bound(), b_max], axis=0)
            b_min = np.min([pc.get_min_bound(), b_min], axis=0)
            pc_dict[file] = pc
    scale = np.max(b_max - b_min)
    if scale < 0.01:
        assert f"SCALE TOO SMALL -- Folder: {inputpath}"

    return _downsample_and_save(inputpath, pc_dict, scale)

def _downsample_and_save(inputpath, pc_dict, scale=2):
    """
    helper for downsamplng and plotting
    """
    # Resample and save to .npy, also output a html scatterplot for easy visualization
    outputpath = os.path.join(inputpath, 'npy')
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        print(f"'{outputpath}' 已被创建。")
    else:
        print(f"'{outputpath}' 已存在,正在重新创建。")
        shutil.rmtree(outputpath)
        os.makedirs(outputpath)

    fig = go.Figure()
    for file, pcloud in pc_dict.items():
        pcloud.scale(2 / scale, [0, 0, 0])
        pc_d = pcloud.voxel_down_sample(voxel_size=sample_size)

        points = np.asarray(pc_d.points)
        normals = np.asarray(pc_d.normals)
        colors = np.asarray(pc_d.colors)
        if points.shape[0] < 1000:
            points = np.asarray(pcloud.points)
            normals = np.asarray(pcloud.normals)
            colors = np.asarray(pcloud.colors)
            print(f"{file} resampled to < 1000 points, using original: #{points.shape[0]}")
        print(f'{file} resampled to {points.shape[0]}')
        data = np.concatenate([points, normals, colors], axis=1)
        fig.add_trace(
            go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                name=f"{file}.npy",
                mode="markers",
                marker=dict(size=2),
            )
        )

        out_file = os.path.join(outputpath, file)
        np.save(out_file, data)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=5,
                range=[-1.5, 1.5],
            ),
            yaxis=dict(
                nticks=5,
                range=[-1.5, 1.5],
            ),
            zaxis=dict(
                nticks=5,
                range=[-1.5, 1.5],
            ),
            aspectmode='cube'
        )
    )
    fig.write_html(os.path.join(outputpath, "scatter_plot.html"))
    return outputpath

# if __name__ == "__main__":
#     convert_pointclouds(input_folder, output_folder)
