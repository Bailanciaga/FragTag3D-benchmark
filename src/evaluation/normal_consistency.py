import numpy as np
import faiss


def normal_consistency(A_points, A_normals, B_points, B_normals):
    # 将数据转为float32类型
    A_points = A_points.astype('float32')
    B_points = B_points.astype('float32')

    # 构建索引
    index_B = faiss.IndexFlatL2(B_points.shape[1])
    index_A = faiss.IndexFlatL2(A_points.shape[1])

    # 添加向量到索引
    index_B.add(B_points)
    index_A.add(A_points)

    # 寻找A中的每个点到B中的最近点的索引
    _, indices_A_to_B = index_B.search(A_points, 1)
    mean_diff_A_to_B = np.mean(
        [np.dot(A_normals[i], B_normals[indices_A_to_B[i][0]]) for i in range(A_points.shape[0])])

    # 寻找B中的每个点到A中的最近点的索引
    _, indices_B_to_A = index_A.search(B_points, 1)
    mean_diff_B_to_A = np.mean(
        [np.dot(B_normals[i], A_normals[indices_B_to_A[i][0]]) for i in range(B_points.shape[0])])

    return (mean_diff_A_to_B + mean_diff_B_to_A) / 2