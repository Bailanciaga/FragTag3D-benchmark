# Author: suhaot
# Date: 2023/10/7
# Description: plot_kp_pairs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_3d_line(coord1, coord2):
    line = go.Scatter3d(
        x=[coord1[0], coord2[0]],
        y=[coord1[1], coord2[1]],
        z=[coord1[2], coord2[2]],
        mode='lines',
        line=dict(
            width=2,
            dash='dash',
        )
    )
    return line


def kp_plot(kp1, kp2, pc1, pc2, d_pairs):
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
        x=pc1[:, 0],
        y=pc1[:, 1],
        z=pc1[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue')
    ))
    fig.add_trace(go.Scatter3d(
        x=pc2[:, 0],
        y=pc2[:, 1],
        z=pc2[:, 2],
        mode='markers',
        marker=dict(size=2, color='red')
    ))
    # 添加关键点
    fig.add_trace(go.Scatter3d(
        x=kp1[:, 0],
        y=kp1[:, 1],
        z=kp1[:, 2],
        mode='markers',
        marker=dict(size=5, color='yellow')
    ))
    fig.add_trace(go.Scatter3d(
        x=kp2[:, 0],
        y=kp2[:, 1],
        z=kp2[:, 2],
        mode='markers',
        marker=dict(size=5, color='yellow')
    ))
    for pair in d_pairs:
        idx2, idx1 = pair
        coord1 = kp1[idx1]
        coord2 = kp2[idx2]
        fig.add_trace(create_3d_line(coord1, coord2))
    fig.show()