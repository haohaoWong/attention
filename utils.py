import numpy as np
import random

def build_K(K):
    a = np.zeros((3,3))
    a[0][0] = K[0]
    a[0][2] = K[2]
    a[1][1] = K[1]
    a[1][2] = K[3]
    a[2][2] = 1
    return a

def skew_matrix(v):
    matrix = [[0 for _ in range(3)]for _ in range(3)]
    matrix[1][0] = v[2]; matrix[0][1] = -v[2]
    matrix[2][0] = -v[1]; matrix[0][2] = v[1]
    matrix[2][1] = v[0]; matrix[1][2] = -v[0]
    return np.array(matrix)


def construct_matrix_A(pts1_norm , pts2_norm):
    n = pts1_norm.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = pts1_norm[i, :2]
        x2, y2 = pts2_norm[i, :2]
        A[i, :] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1 , 1]
    return A

# 八点法计算F矩阵
def eight_point_algo(pts1, pts2):
    A = construct_matrix_A(pts1 , pts2)
    _, _, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)
    U, D, V = np.linalg.svd(F)
    D[-1] = 0
    F = np.dot(U, np.dot(np.diag(D), V))
    return F


def normalize_points(pts):
    n = pts.shape[0]
    mean = np.mean(pts, axis=0)
    s = np.sqrt(2) / np.linalg.norm(pts-mean,axis=1).mean()
    T = np.array([
    [s, 0, -s * mean[0]],
    [0, s, -s * mean[1]],
    [0, 0, 1]
    ])
    pts_norm = np.ones((n, 3))
    pts_norm[:, :2] = (pts - mean) * s
    return pts_norm , T

def denormalize_matrix(F_norm, pts1, pts2):
    T2, T1 = normalize_points(pts2)[1], normalize_points(pts1)[1]
    F = np.dot(T2.T, np.dot(F_norm, T1))
    return F / F[2, 2]


def eight_point_algo_robust(pts1, pts2):
    pts1_norm , _ = normalize_points(pts1) # 归一化
    pts2_norm , _ = normalize_points(pts2) # 归一化
    A = construct_matrix_A(pts1_norm , pts2_norm)
    _, _, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)
    U, D, V = np.linalg.svd(F)
    D[-1] = 0
    F = np.dot(U, np.dot(np.diag(D), V))
    F = denormalize_matrix(F, pts1,pts2) # 反归一化
    return F


# 利用F矩阵计算E矩阵
def essential_from_fundamental(F, K):
    E = np.dot(K.T, np.dot(F, K))
    U, D, V = np.linalg.svd(E)
    D = np.diag([1, 1, 0])
    E = np.dot(U, np.dot(D, V))
    return E/E[2,2]

# 从E矩阵计算RT
def compute_rotation_translation(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Wt = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(Wt, Vt))
    T1 = U[:, 2]
    T2 = -U[:,2]
    return R1, R2, T1,T2



# RANSAC估计E矩阵
def ransac_essential_matrix(points_1, points_2, threshold=1.0, confidence=0.99, max_iterations=1000):
    assert points_1.shape == points_2.shape, "Input array must have the same shape"
    num_points = points_1.shape[0]
    best_inliers = []
    best_E = None

    iterations = max_iterations
    if 0< confidence <1:
        iterations = int(np.log(1- confidence) / np.log(1- (1-0.5**8)**8))
    for _ in range(iterations):
        indices = random.sample(range(num_points), 8)
        sample_1 = points_1[indices]
        sample_2 = points_2[indices]


        E = eight_point_algo(sample_1, sample_2)

        line_1 = np.dot(E.T, points_1.T).T
        line_2 = np.dot(E, points_1.T).T
        
        distances_1 = np.abs(np.sum(line_1 * points_1, axis=1)) / np.sqrt(line_1[:,0]**2 + line_1[:,1]**2)
        distances_2 = np.abs(np.sum(line_2 * points_1, axis=1)) / np.sqrt(line_2[:,0]**2 + line_2[:,1]**2)

        total_distance = distances_1 + distances_2

        inliers = np.where(total_distance< threshold)[0]

        if len(inliers)>len(best_inliers):
            best_inliers = inliers
            best_E = E

            if 0<confidence<1:
                inliers_ratio = len(inliers) / num_points
                print(inliers_ratio)

                if inliers_ratio>0:
                    iterations = int(np.log(1-confidence) / np.log(1-(1-inliers_ratio)**8+1e-8))
                    iterations = min(iterations, max_iterations)
    mask = np.zeros((num_points, 1), dtype=np.uint8)
    mask[best_inliers] = 1
    return best_E, mask

# 三角化

def triangulate_dlt(K, R, T, points1 , points2):
    P1 = K@np.concatenate((np.eye(3),np.zeros((3,1))),axis=-1)
    P2 = K@np.concatenate((R,T.reshape(3,1)),axis=-1)
    num_points = points1.shape[0]
    points_3d = np.zeros((num_points , 3))
    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A = np.array([
                        [x1 * P1[2] - P1[0]],
                        [y1 * P1[2] - P1[1]],
                        [x2 * P2[2] - P2[0]],
                        [y2 * P2[2] - P2[1]]
                        ]).reshape(4, 4)
        _, _, V = np.linalg.svd(A)
        X_homogeneous = V[-1]
        X = X_homogeneous[:3] / X_homogeneous[3]
        points_3d[i] = X
    return points_3d


# 四点法估计H矩阵
def four_point_homography(pts1, pts2):
    pts1_norm , T1 = normalize_points(pts1)
    pts2_norm , T2 = normalize_points(pts2)
    A = np.zeros((2*pts1_norm.shape[0], 9))
    for i in range(pts1_norm.shape[0]):
        x, y = pts1_norm[i][0], pts1_norm[i][1]
        u, v = pts2_norm[i][0], pts2_norm[i][1]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]
    # Solve for the homography matrix
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))
    # Denormalize the homography matrix
    H = np.dot(np.dot(np.linalg.inv(T2), H), T1)
    return H


def homography_calibrated(K, H_uncalib):
    H = np.linalg.inv(K) @ H_uncalib @ K
    eigvals = np.linalg.eigvals(H.T@H)
    H = H/np.sqrt(eigvals[1])
    return H

# H矩阵分解计算R，T
def get_motion_from_homography(H, K):
    H_calib = homography_calibrated(K, H)
    HtH = H_calib.T @ H_calib
    eigvals , eigvecs = np.linalg.eig(HtH)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    v1 = eigvecs[:, 0]
    v2 = eigvecs[:, 1]
    u1 = (np.sqrt(1-eigvals[2])*eigvecs[:, 0]+np.sqrt(eigvals[0]-1)*eigvecs[:, 2])/np.sqrt(eigvals[0]-eigvals[2])
    u2 = (np.sqrt(1-eigvals[2])*eigvecs[:, 0]-np.sqrt(eigvals[0]-1)*eigvecs[:, 2])/np.sqrt(eigvals[0]-eigvals[2])
    U1 = np.stack([v2,u1,skew_matrix(v2)@u1],axis=1)
    W1 = np.stack([H_calib@v2 ,H_calib@u1 ,skew_matrix(H_calib@v2)@H_calib@u1],axis=1)
    U2 = np.stack([v2,u2,skew_matrix(v2)@u2],axis=1)
    W2 = np.stack([H_calib@v2 ,H_calib@u2 ,skew_matrix(H_calib@v2)@H_calib@u2],axis=1)
    R1 = W1@U1.T
    N1 = skew_matrix(v2)@u1
    T1 = (H_calib -R1)@N1
    R2 = W2@U2.T
    N2 = skew_matrix(v2)@u2
    T2 = (H_calib -R2)@N2
    R3 = R1
    N3 = -N1
    T3 = -T1
    R4 = R2
    N4 = -N2
    T4 = -T2
    return (R1,T1), (R2,T2), (R3,T3), (R4,T4)