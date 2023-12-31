# Author: suhaot
# Date: 2023/9/1
# Description: extract_keypoints

import numpy as np
from scipy import spatial
import os, sys
import argparse
import time
import scipy
import plotly
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from multiprocessing import Pool
import pandas as pd
import pdb

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


def objective(X, a0, a1, a2, a3, a4, a5):
    x = X[:, 0]
    y = X[:, 1]
    return a0 * (x ** 2) + a1 * (y ** 2) + a2 * (x * y) + a3 * x + a4 * y + a5


class Extractor(object):
    def __init__(
            self,
            fragment_face,
            output_path,
            num_ftrs,
            keypoint_radius,
            r_values,
            n_keypoints,
            neighbours,
            lbd_value,
            nms,
            nms_rad,
            plot_features=False,
    ):
        self.fragment_face = fragment_face
        self.output_path = output_path
        self.num_features = num_ftrs
        self.r_vals = r_values
        self.keypoint_radius = keypoint_radius
        self.n_keypoints = n_keypoints
        self.cov_mats = []
        self.vertices = None
        self.keypoints = None
        self.k = neighbours
        self.lbd = lbd_value
        self.nms = nms
        self.nms_rad = nms_rad
        self.plot_features = plot_features

    def compute_SD_point(self, neighbourhood, points, normals, p_idx):
        p_i = points[p_idx]
        n_p_i = normals[p_idx]
        p_i_bar = np.mean(points[neighbourhood], axis=0)
        v = p_i - p_i_bar
        SD = np.dot(v, n_p_i)
        return SD

    # Assembling the above
    def get_SD_for_vertices(self, vertices, normals, neighbourhood):
        n_points = len(vertices)
        # Compute SD
        SD = np.zeros((n_points))
        for i in range(n_points):
            SD[i] = self.compute_SD_point(
                np.asarray(neighbourhood[i]), vertices, normals, i
            )
        return SD

    def calculate_H(self, p_i_neighbourhood_points, p_i_point, p_i_normal):
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

    def calculate_deltas(self, p_i_neighbourhood):
        C_p_i = np.cov(self.vertices[p_i_neighbourhood].T) * len(p_i_neighbourhood)
        U, S, Vt = np.linalg.svd(C_p_i)
        lambda1, lambda2, lambda3 = S
        delta1 = (lambda1 - lambda2) / lambda1
        delta2 = (lambda2 - lambda3) / lambda1
        delta3 = lambda3 / lambda1
        return np.array([delta1, delta2, delta3])

    def extract(self):
        fragment = self.fragment_face
        self.vertices = fragment[:, :3]
        # self.vertices = vertices
        normals = fragment[:, 3:6]
        colors = fragment[:, 6:9]
        vertices = self.vertices

        # Get all radius r neighbourhoods for each r
        keypoint_radius = self.keypoint_radius
        tree = spatial.KDTree(vertices)

        # Extract keypoints
        nbhd = tree.query_ball_point(vertices, keypoint_radius, workers=-1)
        SD = self.get_SD_for_vertices(vertices, normals, nbhd)
        print(f"zeros: {np.sum(np.isclose(SD, 0))}, out of {len(SD)}")
        # if SD == 0 we would set the normal to 0, which is bad
        SD[SD == 0] = 1e-10
        # Fix normals
        normals = normals * np.sign(SD[:, None])

        # Edge filtering
        dists_edges, nbhd_edges = tree.query(x=vertices, k=self.k, workers=-1)
        is_edge = []
        for i in range(vertices.shape[0]):
            # get neighboring points
            closest_points = nbhd_edges[i]
            if np.max(closest_points) < vertices.shape[0]:
                C_i = np.mean(vertices[closest_points], axis=0)
            else:
                print(f"Invalid indices found: {closest_points}")
            Z_i = dists_edges[i][1]

            score = np.linalg.norm(C_i - vertices[i])
            is_edge.append(score > self.lbd * Z_i)
        is_edge = np.asarray(is_edge)
        print(f"Points on edge out of total: {np.sum(is_edge)} / {len(is_edge)}")
        SD[is_edge] = 0  # set as 0 if edge

        # Extract keypoint indices

        # Perfrom Non-maximal Suppression
        if self.nms:
            keypoint_indices = np.argsort(-np.abs(SD))[:]
            keypoint_indices_nms = []
            rem_list = []
            nms_rad = self.nms_rad
            nbhd_nms = tree.query_ball_point(vertices, nms_rad, workers=-1)

            for i in keypoint_indices:
                if (len(keypoint_indices_nms) > self.n_keypoints):
                    break
                if i not in rem_list:
                    rem_list.extend(nbhd_nms[i])
                    keypoint_indices_nms.append(i)

            keypoint_indices = np.asarray(keypoint_indices_nms)[:self.n_keypoints]
        else:
            keypoint_indices = np.argsort(np.abs(SD))[-self.n_keypoints:]

        self.keypoints = self.vertices[keypoint_indices]

        # Compute the neighbourhoods in all r vals
        neighbourhoods = {}
        H_lut = np.zeros((len(self.r_vals), vertices.shape[0]))
        deltas_lut = np.zeros((len(self.r_vals), vertices.shape[0], 3))
        print("Building neighbourhoods, H and deltas")
        for r in self.r_vals:
            neighbourhoods[r] = tree.query_ball_point(vertices, r, workers=-1)

            if self.num_features <= 3:
                continue
            # save H and deltas for ALL points
            for p_i in tqdm(range(vertices.shape[0])):
                deltas_lut[self.r_vals.index(r), p_i, :] = self.calculate_deltas(
                    neighbourhoods[r][p_i]
                )
                if self.num_features <= 6:
                    continue
                H_lut[self.r_vals.index(r), p_i] = self.calculate_H(
                    vertices[neighbourhoods[r][p_i]],
                    vertices[p_i],
                    normals[p_i],
                )
        if self.plot_features:
            self.feature_plot(H_lut, deltas_lut)

        # Output
        n_features_used = self.num_features
        output_matrix = np.zeros(
            (
                len(keypoint_indices),
                3 + 3 + 3 + n_features_used * n_features_used * len(self.r_vals),  # 添加了额外的3个空间来保存颜色信息
            )
        )

        print("Extracting features")
        # For each keypoint
        for n_keypoint, keypoint_index in enumerate(tqdm(keypoint_indices)):
            # Get keypoint, normal of the keypoint and color of the keypoint
            keypoint = vertices[keypoint_index]
            keypoint_normal = normals[keypoint_index]
            keypoint_color = colors[keypoint_index]  # 获取关键点的颜色

            # Set output matrix
            output_matrix[n_keypoint, :3] = keypoint
            output_matrix[n_keypoint, 3:6] = keypoint_normal
            output_matrix[n_keypoint, 6:9] = keypoint_color  # 保存关键点的颜色
            keypoint_cov_mats = []

            # For each radius r, compute the matrix with the features
            for r_idx, r in enumerate(self.r_vals):
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
                    p_p_i_vector = p_i_point - keypoint
                    cos_alpha = np.dot(p_p_i_vector, p_i_normal) / np.linalg.norm(
                        p_p_i_vector
                    )
                    cos_beta = np.dot(p_p_i_vector, keypoint_normal) / np.linalg.norm(
                        p_p_i_vector
                    )
                    cos_gamma = np.dot(p_i_normal, keypoint_normal)
                    # phi_i = np.array([cos_alpha, cos_beta, cos_gamma])

                    if self.num_features <= 3:
                        phi_i = [cos_alpha, cos_beta, cos_gamma]
                        Phi[:, idx] = phi_i
                        continue
                    # Compute C_p_i and delta1, delta2, delta3
                    delta1, delta2, delta3 = deltas_lut[r_idx, p_i, :]

                    if self.num_features <= 6:
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

                if np.any(np.isnan(C_r)) or np.any(np.isinf(C_r)):
                    continue

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
            self.cov_mats.append(keypoint_cov_mats)
        return output_matrix

    def feature_plot(self, H_lut, deltas_lut):
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
                x=self.vertices[:, 0],
                y=self.vertices[:, 1],
                z=self.vertices[:, 2],
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
                x=self.vertices[:, 0],
                y=self.vertices[:, 1],
                z=self.vertices[:, 2],
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
                x=self.vertices[:, 0],
                y=self.vertices[:, 1],
                z=self.vertices[:, 2],
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
                x=self.vertices[:, 0],
                y=self.vertices[:, 1],
                z=self.vertices[:, 2],
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


def f(fragment_face, output_path, num_fts, keypoint_radius, r_vals, n_keypoints, k, lbd, nms, nms_rad):
    extractor = Extractor(
        fragment_face, output_path, num_fts, keypoint_radius, r_vals, n_keypoints, k, lbd, nms, nms_rad
    )
    return extractor.extract()


def visualize_matches(extractor1, extractor2, n_points, n_scales, threshold, num_features):
    x_lines = []
    y_lines = []
    z_lines = []
    c = [2, 0, 0]
    colors1 = ["blue" for _ in range(n_points)]
    colors2 = ["blue" for _ in range(n_points)]
    min_d = 99
    dist = []
    n_matches = 0
    for s in range(n_scales):
        cov_m1 = np.array(extractor1.cov_mats)[:, s].reshape((n_points, num_features ** 2))
        cov_m2 = np.array(extractor2.cov_mats)[:, s].reshape((n_points, num_features ** 2))
        dist.append(spatial.distance.cdist(cov_m1, cov_m2, "euclidean"))
    for i in range(n_points):
        for j in range(n_points):
            d = 0.0
            for s in range(n_scales):
                d += dist[s][i, j]
            d /= n_scales
            if d < min_d:
                min_d = d
                print(f"Min: {min_d}")
            if d < threshold:
                n_matches += 1

                colors1[i] = "green"
                colors2[j] = "green"

                x_lines.append(extractor1.keypoints[i][0])
                x_lines.append(extractor2.keypoints[j][0] + c[0])
                x_lines.append(None)

                y_lines.append(extractor1.keypoints[i][1])
                y_lines.append(extractor2.keypoints[j][1] + c[1])
                y_lines.append(None)

                z_lines.append(extractor1.keypoints[i][2])
                z_lines.append(extractor2.keypoints[j][2] + c[2])
                z_lines.append(None)
    print(f"Matches: {n_matches}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=extractor1.vertices[:, 0],
            y=extractor1.vertices[:, 1],
            z=extractor1.vertices[:, 2],
            mode="markers",
            marker=dict(
                size=1,
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=extractor1.keypoints[:, 0],
            y=extractor1.keypoints[:, 1],
            z=extractor1.keypoints[:, 2],
            mode="markers",
            marker=dict(size=3, color=colors1),
        )
    )

    extractor2.keypoints[:, 0] += c[0]
    extractor2.vertices[:, 0] += c[0]
    extractor2.keypoints[:, 1] += c[1]
    extractor2.vertices[:, 1] += c[1]
    extractor2.keypoints[:, 2] += c[2]
    extractor2.vertices[:, 2] += c[2]
    fig.add_trace(
        go.Scatter3d(
            x=extractor2.vertices[:, 0],
            y=extractor2.vertices[:, 1],
            z=extractor2.vertices[:, 2],
            mode="markers",
            marker=dict(
                size=1,
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=extractor2.keypoints[:, 0],
            y=extractor2.keypoints[:, 1],
            z=extractor2.keypoints[:, 2],
            mode="markers",
            marker=dict(size=3, color=colors2),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
        )
    )
    fig.show()


def extract_key_point_by_dir(dataset_dir, mode):
    fragments = os.listdir(dataset_dir)
    fragments = [x for x in fragments if x.endswith(".npy")]

    # f_args = [] # it is for multi thread method
    num_features = 3  # the default value of this parameter is 3
    threshold = 0.2  # the default value of this parameter is 0.2
    k = 200  # the default value of this parameter is 150 [HT is 200]
    lbd = 2  # the default value of this parameter is 2
    nms = True  # the default value of this parameter is True

    for fragment in fragments:
        print(f"Fragment: {fragment}")
        fragment_path = os.path.join(dataset_dir, fragment)
        keypoints_dir = os.path.join(dataset_dir, "keypoints_SD+")
        if not os.path.exists(keypoints_dir):
            os.mkdir(keypoints_dir)

        output_path = os.path.join(keypoints_dir, fragment)
        fragment = np.load(fragment_path)
        colors = fragment[:, 6:9]
        fragment_kp = np.empty((0, 54))
        if not mode == 3:
            # 创建一个布尔掩码，检查colors中的每一行是否不与[255, 255, 255]相匹配
            mask = ~np.all(colors == [1, 1, 1], axis=1)
            # 使用掩码筛选vertices和normals
            fragment = fragment[mask]
            if mode == 1:
                unique_colors = np.unique(np.int64(np.round(fragment[:, 6:9] * 255)), axis=0)
                for color in unique_colors:
                    print("-------Extracting fracture face color is (" + ', '.join(map(str, color)) + ")")  # 打印当前颜色
                    specific_color_mask = np.all(np.isclose(fragment[:, 6:9] * 255, color, atol=1e-5), axis=1)
                    specific_color_fragment_face = fragment[specific_color_mask]
                    # single thread
                    output_matrix = f(specific_color_fragment_face,
                                      output_path,
                                      num_features,
                                      keypoint_radius=0.015,
                                      r_vals=[0.015, 0.02, 0.03, 0.04, 0.05],
                                      n_keypoints=512,
                                      k=200, lbd=2,
                                      nms=True,
                                      nms_rad=0.02,
                                      )
                    fragment_kp = np.vstack((fragment_kp, output_matrix))
                np.save(output_path, fragment_kp)
            if mode == 2:
                output_matrix = f(fragment,
                                  output_path,
                                  num_features,
                                  keypoint_radius=0.04,
                                  r_vals=[0.04, 0.05, 0.06, 0.08, 0.10],
                                  n_keypoints=512,
                                  k=200, lbd=2,
                                  nms=0.04,
                                  nms_rad=0.04
                                  )
                np.save(output_path, output_matrix)
        if mode == 3:
            output_matrix = f(fragment,
                              output_path,
                              num_features,
                              keypoint_radius=0.04,
                              r_vals=[0.04, 0.05, 0.06, 0.08, 0.10],
                              n_keypoints=512,
                              k=200, lbd=2,
                              nms=0.04,
                              nms_rad=0.04)
            np.save(output_path, output_matrix)


import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=2, type=int)
    parser.add_argument("--dataset_dir", default="Cube_dense_8_seed_0", type=str)
    parser.add_argument("--num_features", default=3, type=int)
    parser.add_argument("--keypoint_radius", type=float, default=0.04)
    parser.add_argument("--n_keypoints", type=int, default=2048)
    parser.add_argument("--r_vals", nargs="+", default=[0.04, 0.05, 0.06, 0.08, 0.10], type=float)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--k", type=float, default=200)
    parser.add_argument("--lbd", type=float, default=2)
    parser.add_argument("--nms", action='store_true')
    parser.add_argument("--nms_rad", type=float, default=0.04)
    parser.add_argument("--useflags", type=bool, default=True)
    parser.add_argument(
        "--fragment1", type=str,
        default=os.path.dirname(os.path.abspath(__file__)) + "/Cube_dense_8_seed_0/Cube_shard_3.npy"
    )
    parser.add_argument(
        "--fragment2", type=str,
        default=os.path.dirname(os.path.abspath(__file__)) + "/Cube_dense_8_seed_0/Cube_shard_5.npy"
    )
    args = parser.parse_args()

    if args.mode == 1:
        fragments = os.listdir(args.dataset_dir)
        fragments = [x for x in fragments if x.endswith(".npy")]

        f_args = []

        for fragment in fragments:
            print(f"Fragment: {fragment}")
            fragment_path = os.path.join(args.dataset_dir, fragment)
            keypoints_dir = os.path.join(args.dataset_dir, "keypoints")
            if os.path.exists(keypoints_dir):
                shutil.rmtree(keypoints_dir)
                os.mkdir(keypoints_dir)
            else:
                os.mkdir(keypoints_dir)

            output_path = os.path.join(keypoints_dir, fragment)
            f_args.append(
                (
                    fragment_path,
                    output_path,
                    args.num_features,
                    args.keypoint_radius,
                    args.r_vals,
                    args.n_keypoints,
                    args.k,
                    args.lbd,
                    args.nms,
                    args.nms_rad,
                    args.useflags
                )
            )

        with Pool(processes=2) as pool:
            pool.starmap(f, f_args)

    elif args.mode == 2:
        # Just for visualization purposes
        r = args.keypoint_radius
        scales = args.r_vals
        n_points = args.n_keypoints
        extractor1 = Extractor(
            args.fragment1, "temp", args.num_features, r, scales, n_points, args.k, args.lbd, args.nms, args.nms_rad,
            plot_features=False
        )
        extractor1.extract()
        extractor2 = Extractor(
            args.fragment2, "temp", args.num_features, r, scales, n_points, args.k, args.lbd, args.nms, args.nms_rad,
            plot_features=False
        )
        extractor2.extract()

        visualize_matches(extractor2, extractor1, n_points, len(scales), args.threshold, args.num_features)
