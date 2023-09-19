# Author: suhaot
# Date: 2023/9/19
# Description: estimateGeometricTransform3D
import numpy as np
from scipy.spatial.transform import Rotation
import warnings
from sklearn.neighbors import NearestNeighbors


def estimate_rigid_transform(ptsA, ptsB, max_distance=0.05):
    failedMatrix = np.eye(4, dtype='double')
    statusCode = {
        'NoError': 0,
        'NotEnoughPts': 1,
        'NotEnoughInliers': 2
    }
    ransacFuncs = {
        'checkFunc': checkTForm,
        'evalFunc': evaluateTform3d,
        'fitFunc': compute_rigid_3d
    }
    ransacParams = {
        'maxNumTrials': 1000,
        'confidence': 99,
        'maxDistance': max_distance,
        'recomputeModelFromInliers': 1,
        'sampleSize': 3
    }
    ptsA = ptsA.T
    ptsB = ptsB.T
    status = 0

    if status == statusCode['NoError']:
        pts = np.concatenate((ptsA[..., None], ptsB[..., None]), axis=2).astype('double')
        pts = np.transpose(pts, (2, 0, 1))
        isFound, tmatrix, inlierIdx = msac(pts, ransacParams, ransacFuncs)
        # BUG numpy.linalg.LinAlgError: 1-dimensional array given. Array must be at least two-dimensional [maybe this bug is fixed ]
        if tmatrix.ndim == 1 or np.linalg.det(tmatrix) == 0 or not np.isfinite(tmatrix).all():
            status = statusCode['NotEnoughInliers']
            tmatrix = failedMatrix
    else:
        inlierIdx = np.zeros(ptsA.shape[0], dtype=bool)
        tmatrix = failedMatrix

    if status != statusCode['NoError']:
        tmatrix = failedMatrix

    tform = {
        'Dimensionality': 3,
        'T': tmatrix,
        'Rotation': tmatrix[:3, :3],
        'Translation': tmatrix[3, :3].T
    }
    return tform, inlierIdx


# ---MASC算法---
def msac(allPoints, params, funcs):
    confidence = params['confidence']
    sampleSize = params['sampleSize']
    maxDistance = params['maxDistance']

    threshold = np.array(maxDistance, dtype=allPoints.dtype)
    numPts = allPoints.shape[1]
    idxTrial = 1
    numTrials = params['maxNumTrials']
    maxDis = np.array(threshold * numPts, dtype=allPoints.dtype)
    bestDis = maxDis.copy()
    if 'defaultModel' in params:
        bestModelParams = params['defaultModel']
    else:
        bestModelParams = np.zeros((0,), dtype=allPoints.dtype)

    if 'maxSkipTrials' in params:
        maxSkipTrials = params['maxSkipTrials']
    else:
        maxSkipTrials = params['maxNumTrials'] * 10

    skipTrials = 0
    reachedMaxSkipTrials = False
    bestInliers = np.zeros((numPts,), dtype=bool)

    while idxTrial <= numTrials and skipTrials < maxSkipTrials:
        # Random selection without replacement
        indices = np.random.permutation(numPts)[:sampleSize]
        # indices = np.array([5, 2, 1])
        # Compute a model from samples
        samplePoints = allPoints[:, indices]
        modelParams = funcs['fitFunc'](samplePoints)

        # Validate the model
        isValidModel = funcs['checkFunc'](modelParams)

        if isValidModel:
            # Evaluate model with truncated loss
            model, dis, accDis = evaluateModel(funcs['evalFunc'], modelParams,
                                               allPoints, threshold)

            # Update the best model found so far
            if accDis < bestDis:
                bestDis = accDis
                bestInliers = dis < threshold
                bestModelParams = model
                inlierNum = np.sum(dis < threshold).astype(allPoints.dtype)
                num = computeLoopNumber(sampleSize, confidence, numPts, inlierNum)
                numTrials = min(numTrials, num)

            idxTrial += 1
        else:
            skipTrials += 1
        isFound = funcs['checkFunc'](bestModelParams.ravel()) and bestInliers is not None and sum(bestInliers.ravel()) >= sampleSize
        if isFound:
            inliers = bestInliers

            if numTrials >= int(params['maxNumTrials']):
                warnings.warn('vision:ransac:maxTrialsReached')
        else:
            inliers = np.zeros((allPoints.shape[1],), dtype=bool)
    # reachedMaxSkipTrials = skipTrials >= maxSkipTrials
    return isFound, bestModelParams, inliers


def evaluateModel(evalFunc, modelIn, allPoints, threshold):
    dis = evalFunc(modelIn, allPoints)
    dis[dis > threshold] = threshold
    accDis = np.sum(dis)
    if isinstance(modelIn, list):
        minIdx = np.argmin(accDis)
        sumDistances = accDis[minIdx]
        distances = dis[:, minIdx]
        modelOut = modelIn[minIdx]
    else:
        distances = dis
        modelOut = modelIn
        sumDistances = accDis
    return modelOut, distances, sumDistances

def computeLoopNumber(sampleSize, confidence, pointNum, inlierNum):
    pointNum = np.array(pointNum, dtype=inlierNum.dtype)
    sampleSize = np.array(sampleSize, dtype=inlierNum.dtype)

    inlierProbability = (inlierNum / pointNum) ** sampleSize

    if inlierProbability < np.finfo(inlierNum.dtype).eps:
        N = np.iinfo(np.int32).max
    else:
        conf = np.array(0.01 * confidence, dtype=inlierNum.dtype)
        one = np.ones(1, dtype=inlierNum.dtype)
        num = np.log10(one - conf)
        den = np.log10(one - inlierProbability)
        N = np.int32(np.ceil(num / den))

    return N

def compute_rigid_3d(points):
    points1 = points[0, :, :]
    points2 = points[1, :, :]

    # Get transform
    R, t = computeRigidTransform(points1, points2)
    T = np.eye(4, dtype=points.dtype)
    T[:3, :3] = R.T
    T[3, :3] = t.T
    return T

def computeRigidTransform(p, q):
    # Find data centroid and deviations from centroid
    centroid1 = np.mean(p, axis=0)
    centroid2 = np.mean(q, axis=0)

    normPoints1 = p - centroid1
    normPoints2 = q - centroid2

    # Covariance matrix
    C = normPoints1.T @ normPoints2

    U, _, Vt = np.linalg.svd(C)

    # Handle the reflection case
    #R = V @ np.diag([1] * (p.shape[1] - 1) + [np.linalg.det(U @ V.T)]) @ U.T

    V = Vt.T
    scaling_factor = np.linalg.det(U @ Vt)
    scaling_sign = np.sign(scaling_factor)
    scaling_matrix = np.diag(np.concatenate((np.ones(p.shape[1] - 1), [scaling_sign])))
    R = V @ scaling_matrix @ U.T

    # Compute the translation
    t = centroid2.T - R @ centroid1.T

    return R, t.ravel()

def evaluateTform3d(tform, points):
    points1 = points[0, :, :]
    points2 = points[1, :, :]

    numPoints = points1.shape[0]
    pt1h = np.hstack((points1, np.ones((numPoints, 1), dtype=points.dtype)))

    tpoints1 = np.dot(pt1h, tform)

    dis = np.sqrt((tpoints1[:, 0] - points2[:, 0]) ** 2 +
                  (tpoints1[:, 1] - points2[:, 1]) ** 2 +
                  (tpoints1[:, 2] - points2[:, 2]) ** 2)
    return dis

def checkTForm(tform):
    tf = np.all(np.isfinite(tform))
    return tf


if __name__ == "__main__":
    # Transformation (R, T) to map ptsA to ptsB (ptsB_transformed = R.dot(ptsA.T) + T)
    # where ptsA and ptsB are Nx3 matrices representing point clouds
    A = [[1.73986572676213, 1.79221744032004, 1.80468203816702, 1.82248241155384, 1.75215968154292, 1.77878684016807,
          0.280575102668862],
         [0.871076124746639, 1.09019942689827, 1.07651825218343, 1.04765687735218, 0.847665570818205, 1.10375929365596,
          0.698418860525351],
         [0.187435290591583, 0.226246509683135, 0.186920266571646, 0.112313939045105, 0.128186910215224,
          0.264994113499425,
          0.0758196801247627]]
    B = [[0.850600561346547, 0.261569017247584, 0.554649059044200, 0.261569017247584, 0.744358798597328,
          0.601402259818511,
          0.781820023370595],
         [-0.522316969861640, -0.331480369946977, -0.463791586713353, -0.331480369946977, -0.509849667876063,
          -0.476248554501728, -0.314820863431970],
         [1.49252412225960, 1.48507686499326, 1.63796942445409, 1.48507686499326, 1.47613460992145, 1.65290837996204,
          1.49074075676114]]
    ptsA = np.array(A)
    ptsB = np.array(B)
    tform, inlierIdx = estimate_rigid_transform(ptsA, ptsB, max_distance=0.05)
    print(tform)
    print(inlierIdx)
