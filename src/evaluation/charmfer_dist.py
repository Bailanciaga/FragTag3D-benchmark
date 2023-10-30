import numpy as np
import faiss


def chamfer_distance(A, B):
    # 将数据转为float32类型
    A = A.astype('float32')
    B = B.astype('float32')

    # 构建索引
    index_A = faiss.IndexFlatL2(A.shape[1])
    index_B = faiss.IndexFlatL2(B.shape[1])

    # 添加向量到索引
    index_A.add(A)
    index_B.add(B)

    # 寻找A中的每个点到B中的最近点的距离
    _, indices_A = index_A.search(B, 1)
    dist_A = np.mean(np.sum((B - A[indices_A.ravel()]) ** 2, axis=1))

    # 寻找B中的每个点到A中的最近点的距离
    _, indices_B = index_B.search(A, 1)
    dist_B = np.mean(np.sum((A - B[indices_B.ravel()]) ** 2, axis=1))

    return dist_A + dist_B

# from scipy.spatial import KDTree
#
#
# def chamfer_distance(A, B):
#     tree_A = KDTree(A)
#     tree_B = KDTree(B)
#
#     # 寻找A中的每个点到B中的最近点的距离
#     dist_A, _ = tree_A.query(B)
#     dist_A = np.sum(dist_A**2) / len(A)
#
#     # 寻找B中的每个点到A中的最近点的距离
#     dist_B, _ = tree_B.query(A)
#     dist_B = np.sum(dist_B**2) / len(B)
#
#     return dist_A + dist_B
