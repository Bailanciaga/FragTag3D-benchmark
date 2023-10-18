# Author: suhaot
# Date: 2023/10/5
# Description: extract_descriptors_cov
import numpy as np
from scipy import spatial
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

useflags = True
keypoint_radius = 0.08
r_vals = [0.08, 0.09, 0.1, 0.12, 0.14]
num_features = 3
plot_features = False
cov_mats = []
output_path = ''


def calculate_deltas(vertices, p_i_neighbourhood):
    C_p_i = np.cov(vertices[p_i_neighbourhood].T) * len(p_i_neighbourhood)
    U, S, Vt = np.linalg.svd(C_p_i)
    lambda1, lambda2, lambda3 = S
    delta1 = (lambda1 - lambda2) / lambda1
    delta2 = (lambda2 - lambda3) / lambda1
    delta3 = lambda3 / lambda1
    return np.array([delta1, delta2, delta3])


def calculate_H(p_i_neighbourhood_points, p_i_point, p_i_normal):
    x = p_i_neighbourhood_points[:, 0][:, None]
    y = p_i_neighbourhood_points[:, 1][:, None]
    z = p_i_neighbourhood_points[:, 2][:, None]
    X = np.concatenate([x ** 2, y ** 2, x * y, x, y, np.ones_like(x)], axis=1)
    w = np.linalg.lstsq(X, z, rcond=None)[0]
    a0, a1, a2, a3, a4, a5 = w
    # xdata = p_i_neighbourhood_points[:, :2]
    # ydata = p_i_neighbourhood_points[:, 2]
    # popt, _ = curve_fit(objective, xdata, ydata, p0=[0,0,0,0,0,0])
    # a0, a1, a2, a3, a4, a5 = popt
    r_x = np.zeros((3))
    r_xy = np.zeros((3))
    r_xx = np.zeros((3))
    r_y = np.zeros((3))
    r_yy = np.zeros((3))
    r_x[2] = 2 * a0 * p_i_point[0] + a2 * p_i_point[1] + a3
    r_xx[2] = 2 * a0
    r_xy[2] = a2
    r_y[2] = 2 * a1 * p_i_point[1] + a2 * p_i_point[0] + a4
    r_yy[2] = 2 * a1
    r_x[0] = 1
    r_y[1] = 1
    E = np.dot(r_x, r_x)
    F = np.dot(r_x, r_y)
    G = np.dot(r_y, r_y)
    L = np.dot(r_xx, p_i_normal)
    M = np.dot(r_xy, p_i_normal)
    N = np.dot(r_yy, p_i_normal)
    H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F ** 2))
    # if(np.linalg.norm(p_i_point- np.array([-0.2411, 0.2888, -0.1318])) < 0.001):
    #     print(H)
    return H


def feature_plot(vertices, H_lut, deltas_lut):
    # scatter plot H values
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
        subplot_titles=(
            "H Value",
            "Delta1 Value",
            "Delta2 Value",
            "Delta3 Value",
        ),
    )
    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=np.log(np.abs(H_lut[0, :]) + 0.00001),
                showscale=True,
                colorscale="thermal",
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=np.log(np.abs(deltas_lut[0, :, 0]) + 0.00001),
                showscale=True,
                colorscale="thermal",
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=np.log(np.abs(deltas_lut[0, :, 1]) + 0.00001),
                showscale=True,
                colorscale="thermal",
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=np.log(np.abs(deltas_lut[0, :, 2]) + 0.00001),
                showscale=True,
                colorscale="thermal",
            ),
        ),
        row=2,
        col=2,
    )

    fig.show()


# Compute the neighbourhoods in all r vals
def get_cov_descriptor(fragment, keypoint, keypoint_indices):
    vertices = fragment[:, :3]
    # self.vertices = vertices
    normals = fragment[:, 3:6]
    colors = fragment[:, 6:9]
    tree = spatial.KDTree(vertices)
    neighbourhoods = {}
    H_lut = np.zeros((len(r_vals), vertices.shape[0]))

    deltas_lut = np.zeros((len(r_vals), vertices.shape[0], 3))
    print("Building neighbourhoods, H and deltas")
    for r in r_vals:
        neighbourhoods[r] = tree.query_ball_point(vertices, r, workers=-1)

        if num_features <= 3:
            continue
        # save H and deltas for ALL points
        for p_i in tqdm(range(vertices.shape[0])):
            deltas_lut[r_vals.index(r), p_i, :] = calculate_deltas(
                vertices, neighbourhoods[r][p_i]
            )
            if num_features <= 6:
                continue
            H_lut[r_vals.index(r), p_i] = calculate_H(
                vertices[neighbourhoods[r][p_i]],
                vertices[p_i],
                normals[p_i],
            )
    if plot_features:
        feature_plot(vertices, H_lut, deltas_lut)

    # Output
    n_features_used = num_features
    output_matrix = np.zeros(
        (
            len(keypoint_indices),
            3 + 3 + 3 + n_features_used * n_features_used * len(r_vals),  # 添加了额外的3个空间来保存颜色信息
        )
    )

    print("Extracting features")
    # For each keypoint
    for n_keypoint, keypoint_index in enumerate(tqdm(keypoint_indices)):
        # Get keypoint, normal of the keypoint and color of the keypoint
        vertice = vertices[keypoint_index]
        normal = normals[keypoint_index]
        color = colors[keypoint_index]  # 获取关键点的颜色

        # Set output matrix
        output_matrix[n_keypoint, :3] = vertice
        output_matrix[n_keypoint, 3:6] = normal
        output_matrix[n_keypoint, 6:9] = color  # 保存关键点的颜色
        keypoint_cov_mats = []

        # For each radius r, compute the matrix with the features
        for r_idx, r in enumerate(r_vals):
            # Get neighbourhood of keypoint获得关键点的近邻
            keypoint_neighbourhood = neighbourhoods[r][keypoint_index]

            # Initialize Phi 初始化Phi
            Phi = np.zeros((n_features_used, len(keypoint_neighbourhood)))

            # For each point in the keypoint neighbourhood对关键点邻域中每个点进行遍历
            for idx, p_i in enumerate(keypoint_neighbourhood):
                if p_i == keypoint_index:
                    continue

                # Get neighbourhood of p_i
                p_i_neighbourhood = neighbourhoods[r][p_i]

                # Compute cosines
                p_i_point = vertices[p_i]
                p_i_normal = normals[p_i]
                p_p_i_vector = p_i_point - vertice
                cos_alpha = np.dot(p_p_i_vector, p_i_normal) / np.linalg.norm(
                    p_p_i_vector
                )
                cos_beta = np.dot(p_p_i_vector, normal) / np.linalg.norm(
                    p_p_i_vector
                )
                cos_gamma = np.dot(p_i_normal, normal)
                # phi_i = np.array([cos_alpha, cos_beta, cos_gamma])

                if num_features <= 3:
                    phi_i = [cos_alpha, cos_beta, cos_gamma]
                    Phi[:, idx] = phi_i
                    continue
                # Compute C_p_i and delta1, delta2, delta3
                delta1, delta2, delta3 = deltas_lut[r_idx, p_i, :]

                if num_features <= 6:
                    phi_i = [cos_alpha, cos_beta, cos_gamma, delta1, delta2, delta3]
                    Phi[:, idx] = phi_i
                    continue
                # Compute H
                H = H_lut[r_idx, p_i]

                # Set the value
                phi_i = [cos_alpha, cos_beta, cos_gamma, delta1, delta2, delta3, H]
                Phi[:, idx] = phi_i
            # if len(Phi) == 0:
            # 	pdb.set_trace()
            # 	ValueError('Phi为空！！')
            # 寻找包含nan的列
            cols_with_nan = np.any(np.isnan(Phi), axis=0)

            # 删除这些列
            Phi = Phi[:, ~cols_with_nan]
            C_r = np.cov(Phi)

            # Compute log and save
            S, R = np.linalg.eig(C_r)
            S = np.log(S + 1e-5)
            log_C_r = R @ np.diag(S) @ R.T
            keypoint_cov_mats.append(log_C_r)
            output_matrix[
            n_keypoint,
            9
            + r_idx * n_features_used * n_features_used: 9
                                                         + (r_idx + 1) * n_features_used * n_features_used,
            ] = log_C_r.ravel()
        cov_mats.append(keypoint_cov_mats)
    return output_matrix