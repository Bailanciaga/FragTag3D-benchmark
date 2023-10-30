# Author: suhaot
# Date: 2023/10/7
# Description: plot_kp_pairs
import plotly.graph_objects as go
import open3d as o3d
from plotly.subplots import make_subplots
import numpy as np


def create_3d_line(coord1, coord2):
    line = go.Scatter3d(
        x=[coord1[0], coord2[0]],
        y=[coord1[1], coord2[1]],
        z=[coord1[2], coord2[2]],
        mode='lines',
        line=dict(
            width=5,
            dash='dash',
        )
    )
    return line


def pc_down_sample(pc, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    down_pcd = pcd.voxel_down_sample(voxel_size)
    down_pcd = np.asarray(down_pcd.points)
    return down_pcd


def kp_plot(kp1, kp2, pc1, pc2, d_pairs):
    down_pcd1 = pc_down_sample(pc1, 0.05)
    down_pcd2 = pc_down_sample(pc2, 0.05)
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                showbackground=False,
                showline=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showbackground=False,
                showline=False,
                zeroline=False,
                showticklabels=False
            ),
            zaxis=dict(
                showbackground=False,
                showline=False,
                zeroline=False,
                showticklabels=False
            )
        )
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter3d(
        x=down_pcd1[:, 0],
        y=down_pcd1[:, 1],
        z=down_pcd1[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.6)
    ))
    fig.add_trace(go.Scatter3d(
        x=down_pcd2[:, 0],
        y=down_pcd2[:, 1],
        z=down_pcd2[:, 2],
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.6)
    ))
    # 添加关键点
    fig.add_trace(go.Scatter3d(
        x=kp1[:, 0],
        y=kp1[:, 1],
        z=kp1[:, 2],
        mode='markers',
        marker=dict(size=3, color='yellow', opacity=0.8, line=dict(
                width=2,
                color='DarkSlateGrey'
            ))
    ))
    fig.add_trace(go.Scatter3d(
        x=kp2[:, 0],
        y=kp2[:, 1],
        z=kp2[:, 2],
        mode='markers',
        marker=dict(size=3, color='yellow', opacity=0.8, line=dict(
                width=2,
                color='DarkSlateGrey'
            ))
    ))
    for pair in d_pairs:
        idx2, idx1 = pair
        coord1 = kp1[idx1]
        coord2 = kp2[idx2]
        fig.add_trace(create_3d_line(coord1, coord2))
        fig.add_trace(go.Scatter3d(
            x=[coord1[0], coord2[0]],
            y=[coord1[1], coord2[1]],
            z=[coord1[2], coord2[2]],
            mode='markers',
            marker=dict(size=7, color='purple', line=dict(
                width=5,
                color='black'
            ))
        ))
    fig.show()
