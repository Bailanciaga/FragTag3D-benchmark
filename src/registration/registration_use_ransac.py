# Author: suhaot
# Date: 2023/9/19
# Description: registration_use_ransac
import numpy as np

from src.registration.estimateGeometricTransform3D import estimate_rigid_transform

print_flag = True


def assy_use_ransac(kp1, kp2, d_pairs, d_dist):
    nb_non_matches = 0
    A_rp_pair = np.hstack([kp1[:, :1][d_pairs[:, 0]],
                          kp1[:, 1:2][d_pairs[:, 0]],
                          kp1[:, 2:3][d_pairs[:, 0]]])

    # A_gt_pair = np.array([kp[comb_pairwise[i][0]]['gt']['x'][d_pairs[:, 0]],
    #                       kp[comb_pairwise[i][0]]['gt']['y'][d_pairs[:, 0]],
    #                       kp[comb_pairwise[i][0]]['gt']['z'][d_pairs[:, 0]]]).T

    B_rp_pair = np.hstack([kp2[:, :1][d_pairs[:, 1]],
                          kp2[:, 1:2][d_pairs[:, 1]],
                          kp2[:, 2:3][d_pairs[:, 1]]])

    # B_gt_pair = np.array([kp[comb_pairwise[i][1]]['gt']['x'][d_pairs[:, 1]],
    #                       kp[comb_pairwise[i][1]]['gt']['y'][d_pairs[:, 1]],
    #                       kp[comb_pairwise[i][1]]['gt']['z'][d_pa irs[:, 1]]]).T
    if print_flag:
        print('Number of keypoint pairs: ', len(A_rp_pair))
    # Prepare input data for solver
    ptsA = A_rp_pair.T
    ptsB = B_rp_pair.T
    zcA = np.zeros((1, ptsA.shape[1]))
#    ptsA_z = np.vstack((ptsA, zcA))
    zcB = np.zeros((1, ptsB.shape[1]))
#    ptsB_z = np.vstack((ptsB, zcB))

    # Transformation (R, T) to map pts1 to pts2 (pts1_transformed = R_ransac*ptsA+T_ransac)
    if len(A_rp_pair) > 1:
        tformEst, inlierIndex = estimate_rigid_transform(ptsA, ptsB, max_distance=0.3)
        R_ransac = tformEst["Rotation"].T
        T_ransac = tformEst["Translation"].T
        # tformEst, inlierIndex = estimateGeometricTransform3D(ptsA.T, ptsB.T, transform_type='rigid',maxDistance=0.05)
        # R_ransac = tformEst[:3, :3]
        # T_ransac = tformEst[:, 3].reshape((3, 1))
        x = np.sum(inlierIndex)
        y = inlierIndex.shape[0] - x
        if print_flag:
            print("Inliers")
            print(x)
            print("Outliers")
            print(y)

        matching_pairwise = {}
        if x - y >= 5:
            if print_flag:
                print('Valid solution for T_ransac')

            matching_pairwise['transformation_A'] = np.vstack((np.hstack((R_ransac, T_ransac.reshape(-1, 1))),
                                                                  np.array([0, 0, 0, 1])))

            nb_non_matches += 1

            # Apply transformation (R,T) to fragment A
            # ptsA_transformed = R_ransac @ A_rp_pair.T + T_ransac.reshape(-1, 1)
            # ptsA_all_transformed = R_ransac @ A_rp.T + T_ransac.reshape(-1, 1)

            # Update match pairwise array
            # match_pairwise = 1

        else:
            if print_flag:
                print('No valid solution for T_ransac -> create NaN R, T')
            R_ransac = np.full((3, 3), np.nan)
            T_ransac = np.full((3, 1), np.nan)
            matching_pairwise['transformation_A'] = np.vstack(
                (np.hstack((R_ransac, T_ransac.reshape(-1, 1))), np.array([0, 0, 0, 1])))
            nb_non_matches += 1

    else:
        if print_flag:
            print('No valid solution for T_ransac -> create NaN R, T')
        matching_pairwise = {}  # In the if part have it
        R_ransac = np.full((3, 3), np.nan)
        T_ransac = np.full((3, 1), np.nan)
        matching_pairwise['transformation_A'] = np.vstack(
            (np.hstack((R_ransac, T_ransac.reshape(-1, 1))), np.array([0, 0, 0, 1])))
        nb_non_matches += 1
    return R_ransac, T_ransac