import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d
from utils import*
from pysift import get_sift_points
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_1', type=str, default="assignment/pair1/000006.png")
    parser.add_argument('--image_2', type=str, default="assignment/pair1/000007.png")
    parser.add_argument('--K_path', type=str, default="assignment/pair1/K.txt")
    args = parser.parse_args()
    return args


if __name__ =="__main__":
    args = get_args()

    path = args.K_path
    data_path = result = path[:path.rfind('/') + 1]
    

    if os.path.exists(data_path+'data.npz'):
        data = np.load(data_path+"data.npz")
        x2d_0 = data['x2d_0']
        x2d_1 = data['x2d_1']
    else:
        x2d_0, x2d_1 = get_sift_points(args.img_path_1, args.img_path_2)
        np.savez(data_path+'data.npz', x2d_0=x2d_0, x2d_1=x2d_1)
    
    # K0 = np.loadtxt('assignment/pair1/K.txt', delimiter=',')
    K0 = np.loadtxt(args.K_path, delimiter=',')
    K = build_K(K0)

    """
    # method_1
    F_est =eight_point_algo(x2d_0, x2d_1)
    F_est/= F_est[2,2]
    E = essential_from_fundamental(F_est, K)
    R0, R1, T0, T1 = compute_rotation_translation(E)
    R = R1@R0.T
    T = (T1-R@T0).flatten()
    points_3D = triangulate_dlt(K, R, T, x2d_0 , x2d_1)
    """

    # method_2 
    x2d_0h = np.concatenate((x2d_0, np.ones((x2d_0.shape[0], 1))), axis=-1)
    x2d_1h = np.concatenate((x2d_1, np.ones((x2d_1.shape[0], 1))), axis=-1)
    x2d_0n = (np.linalg.inv(K)@(x2d_0h.T)).T
    x2d_1n = (np.linalg.inv(K)@(x2d_1h.T)).T
    E_best, mask = ransac_essential_matrix(x2d_0n, x2d_1n, threshold=0.5)
    R1, R2, T1, T2 = compute_rotation_translation(E_best)


    putative_poses = [
        [R1, T1],
        [R1, T2],
        [R2, T1],
        [R2, T2]
    ]

    num_positives = []
    for pose in putative_poses:
        X3d = triangulate_dlt(K, pose[0], pose[1], 
                                np.delete(x2d_0n[mask.flatten()==1], -1, 1),
                                np.delete(x2d_1n[mask.flatten()==1], -1, 1))
        
        num_pos_1 = X3d[:,-1]>0
        x3d = (pose[0]@X3d.T + pose[1][:,None]).T
        num_pos_2 = x3d[:, -1]>0

        num_positives.append(np.sum(num_pos_1*num_pos_2))

    final_pose = putative_poses[np.argmax(num_positives)]
    final_X = triangulate_dlt(K, final_pose[0], final_pose[1], 
                                np.delete(x2d_0n[mask.flatten()==1], -1, 1),
                                np.delete(x2d_1n[mask.flatten()==1], -1, 1))





    final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_X))
    o3d.visualization.draw_geometries([final_pcd])
