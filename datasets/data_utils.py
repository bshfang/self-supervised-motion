import json
from datetime import datetime
import os
from collections import Counter
from tqdm import tqdm
import yaml
import os.path as osp
import numpy as np
import math
from pypcd import pypcd
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import Box
from nuscenes.utils.data_classes import LidarPointCloud


class MyLidarPointCloud(LidarPointCloud):
    def get_ego_mask(self):
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5),
        )
        return ego_mask


def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data['x'])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data['y'])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data['z'])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data['intensity'])
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points


def mask_points_by_range(points, limit_range):
    """
    Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """

    mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3])\
           & (points[:, 1] > limit_range[1]) & (
                   points[:, 1] < limit_range[4]) \
           & (points[:, 2] > limit_range[2]) & (
                   points[:, 2] < limit_range[5])

    points = points[mask]

    return points,mask


def get_nusc_pc(pc_file, transform_mat=None):
    """
    get lidar point cloud of nuscenes frame
    """
    if pc_file.endswith('.pcd.bin'):
        lidar_pc = MyLidarPointCloud.from_file(pc_file)
        ego_mask = lidar_pc.get_ego_mask()
        lidar_pc.points = lidar_pc.points[:, np.logical_not(ego_mask)]
        if transform_mat is not None:
            lidar_pc.transform(transform_mat)

        points_tf = np.array(lidar_pc.points[:4].T, dtype=np.float32)
    elif pc_file.endswith('.pcd'):
        pcd = read_pcd(pc_file)
        lidar_pc = MyLidarPointCloud(pcd.T)
        ego_mask = lidar_pc.get_ego_mask()
        lidar_pc.points = lidar_pc.points[:, np.logical_not(ego_mask)]
        if transform_mat is not None:
            lidar_pc.transform(transform_mat)
        points_tf = np.array(lidar_pc.points[:4].T, dtype=np.float32)
    else:
        return None
    
    return points_tf


def get_waymo_pc(pc_file, transform_mat, limit_range=None):
    """
    get lidar point cloud of nuscenes frame
    """
    assert pc_file.endswith('.pcd'), "Point Cloud File Type Error!"
        
    if pc_file.endswith('.pcd'):
        pcd = read_pcd(pc_file)
        lidar_pc = MyLidarPointCloud(pcd.T)
        ego_mask = lidar_pc.get_ego_mask()
        
        lidar_pc.points = lidar_pc.points[:, np.logical_not(ego_mask)]
        lidar_pc.transform(transform_mat)
        
        points_tf = np.array(lidar_pc.points[:4].T, dtype=np.float32)
    else:
        return None
    
    return points_tf

