# Author: suhaot
# Date: 2023/10/29
# Description: hausdorff_dist
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def hausdorff_dist(pc1, pc2):
    # 计算有向豪斯多夫距离
    hd1 = directed_hausdorff(pc1, pc2)[0]
    hd2 = directed_hausdorff(pc2, pc1)[0]

    # 真正的豪斯多夫距离是两个有向距离中的最大值
    return max(hd1, hd2)