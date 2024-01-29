import json
from datetime import datetime
import os
from collections import Counter
from tqdm import tqdm
import yaml
import os.path as osp
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

from pypcd import pypcd

from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from argoverse.utils.cuboid_interior import (filter_point_cloud_to_bbox_3D_vectorized)
from argoverse.utils.se3 import SE3


from pypcd import pypcd

import cv2
import torch


def calc_displace_vector(points: np.array, curr_box: Box, next_box: Box):
    """
    Calculate the displacement vectors for the input points.
    This is achieved by comparing the current and next bounding boxes. Specifically, we first rotate
    the input points according to the delta rotation angle, and then translate them. Finally we compute the
    displacement between the transformed points and the input points.
    :param points: The input points, (N x d). Note that these points should be inside the current bounding box.
    :param curr_box: Current bounding box.
    :param next_box: The future next bounding box in the temporal sequence.
    :return: Displacement vectors for the points.
    """
    assert points.shape[1] == 3, "The input points should have dimension 3."

    # Make sure the quaternions are normalized
    curr_box.orientation = curr_box.orientation.normalised
    next_box.orientation = next_box.orientation.normalised

    delta_rotation = curr_box.orientation.inverse * next_box.orientation
    rotated_pc = (delta_rotation.rotation_matrix @ points.T).T
    rotated_curr_center = np.dot(delta_rotation.rotation_matrix, curr_box.center)
    delta_center = next_box.center - rotated_curr_center

    rotated_tranlated_pc = rotated_pc + delta_center

    pc_displace_vectors = rotated_tranlated_pc - points

    return pc_displace_vectors



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

    return points


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def resize_img(img, img_hw):
    '''
    Input size (N*H, W, 3)
    Output size (N*H', W', 3), where (H', W') == self.img_hw
    '''
    img_h, img_w = img.shape[0], img.shape[1]
    img_new = cv2.resize(img, (img_hw[1], img_hw[0]))
    return img_new

def read_img(img_path,img_hw):
    img = cv2.imread(img_path)
    img=resize_img(img,img_hw)
    img = img.transpose(2,0,1)    
    img = img / 255.0
    return torch.from_numpy(img).float().unsqueeze(0)

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

def read_cam_para(vis_id,data_info,root_path):
    value=data_info[vis_id]
    #print(value)
    intrin_file = osp.join(root_path, value['calib_camera_intrinsic_path'])
    cam_K = load_json(intrin_file)["cam_K"]
    cam_intrin = np.array(cam_K).reshape([3, 3], order="C")
    
    # extrinsic
    extrin_file = osp.join(root_path, value['calib_virtuallidar_to_camera_path'])
    extrin_json = load_json(extrin_file)
    l2r_r = np.array(extrin_json["rotation"])
    l2r_t = np.array(extrin_json["translation"])

    return cam_intrin,l2r_r,l2r_t

def lidar2cam(vis_id,pcd,data_info,root_path):
    pc_points = np.array(pcd[:, :3]).T

    cam_intrin,l2r_r,l2r_t=read_cam_para(vis_id,data_info,root_path)

    # transform to cam coordinates
    pc_points = l2r_r @ pc_points + l2r_t
    pc_points_2d = cam_intrin @ pc_points
    pc_points_2d = pc_points_2d.T

    # normalize
    pc_points_2d = pc_points_2d[pc_points_2d[:, 2] > 0]
    pc_points_2d = pc_points_2d[:, :2] / pc_points_2d[:, 2:3]

    pc_points_2d[:,1]=pc_points_2d[:,1]*(1088/1080)

    mask = (pc_points_2d[:, 0] > 0) & (pc_points_2d[:, 0] < 1920) & \
            (pc_points_2d[:, 1] > 0) & (pc_points_2d[:, 1] < 1088)

    
    return mask,pc_points_2d


def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [center_lidar[0], center_lidar[1], center_lidar[2]]

    lidar_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T

    return corners_3d_lidar.T


def euler2quat(euler):
    ret =R.from_euler('x',euler,degrees=False)
    Q = ret.as_quat()
    return Q


def read_label_bboxes(label_path):
    with open(label_path, "r") as load_f:
        labels = json.load(load_f)

    boxes = []
    for label in labels:
        #print("label:",label.keys())
        obj_size = [
            float(label["3d_dimensions"]["w"]),
            float(label["3d_dimensions"]["l"]),
            float(label["3d_dimensions"]["h"]),
        ]
        yaw_lidar = float(label["rotation"])
        center_lidar = [
            float(label["3d_location"]["x"]),
            float(label["3d_location"]["y"]),
            float(label["3d_location"]["z"]),
        ]

        box = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)
        #print(yaw_lidar)
        quat=euler2quat(np.pi-yaw_lidar)
        #print(quat)
        box=np.concatenate([center_lidar,obj_size,quat])
        #print("box:",box.shape)
        boxes.append(box)

    return boxes


def point_in_hull_fast(points: np.array, bounding_box: Box):
    """
    Check if a point lies in a bounding box. We first rotate the bounding box to align with axis. Meanwhile, we
    also rotate the whole point cloud. Finally, we just check the membership with the aid of aligned axis.
    This implementation is fast.
    :param points: nd.array (N x d); N: the number of points, d: point dimension
    :param bounding_box: the Box object
    return: The membership of points within the bounding box
    """
    # Make sure it is a unit quaternion
    bounding_box.orientation = bounding_box.orientation.normalised

    # Rotate the point clouds
    pc = bounding_box.orientation.inverse.rotation_matrix @ points.T
    pc = pc.T

    orientation_backup = Quaternion(bounding_box.orientation)  # Deep clone it
    bounding_box.rotate(bounding_box.orientation.inverse)

    corners = bounding_box.corners()

    # Test if the points are in the bounding box
    idx = np.where((corners[0, 7] <= pc[:, 0]) & (pc[:, 0] <= corners[0, 0]) &
                   (corners[1, 1] <= pc[:, 1]) & (pc[:, 1] <= corners[1, 0]) &
                   (corners[2, 2] <= pc[:, 2]) & (pc[:, 2] <= corners[2, 0]))[0]

    # recover
    bounding_box.rotate(orientation_backup)

    return idx


def gen_flow(pc1,pc2,bboxes_aux,bbox_list_1,bbox_list_2,thresh=4):
    """
    pc1 (N1,3) current_pc (t=0)
    pc2 (N2.3) future_pc (t=i)
    bbox_list_1:[L1]
    bboxes_aux: aligned boxes (t=i-1) with boxes (t=0) [L1]
    bbox_list_2:[L2]   
    """
    
    n1 = pc1.shape[0]
        
    mask_tracks_flow_temp = []
    mask1_tracks_flow = []
    mask2_tracks_flow = []
    bbox_centers=[]
    refer_bbox_list=[]

    margin=0.2

    flow = np.zeros((n1, 3), dtype='float32')
    
    for i in range(len(bbox_list_1)):
        instance_box_data=bbox_list_1[i]

        box1 = Box(center=instance_box_data[:3], size=instance_box_data[3:6],
                          orientation=Quaternion(instance_box_data[6:]))
        box1.wlh = (1+margin)*box1.wlh 
        
        bbox1_3d = box1.corners().T
        inbox_pc1, is_valid = filter_point_cloud_to_bbox_3D_vectorized(bbox1_3d, pc1)
        
        #print("inbox_pc1:",inbox_pc1.shape)    
        indices = np.where(is_valid == True)[0]
        #print("indice:",indices.shape)
        #print(box1.wlh[0]*box1.wlh[1]*box1.wlh[2])
        if inbox_pc1.shape[0]>0 and box1.wlh[0]*box1.wlh[1]*box1.wlh[2]>2: #Eliminate Traffincone 
            #pc1_inbox.append(inbox_pc1)
            #print(inbox_pc1.shape)
            mask_tracks_flow_temp.append(indices)
            #mask1_tracks_flow.append(indices)
            instance_box_data_aux=bboxes_aux[i]
            #print(instance_box_data_aux[:3])
            if instance_box_data_aux is not None:
                bbox_centers.append(instance_box_data_aux[:3])
            else:
                bbox_centers.append(np.array([-100.0,-100.0,-10.0]))  #no_matched  
            refer_bbox_list.append(box1)
        else:
            ### keep len(ref_bbox_list)==len(bbox_aux)
            bbox_centers.append(np.array([-100.0,-100.0,-10.0]))
            refer_bbox_list.append(None)
            mask_tracks_flow_temp.append(None)
    
    bbox_centers=np.stack(bbox_centers,axis=0)

    new_bboxes_aux=[None for i in range(len(bbox_list_1))]
    for j in range(len(bbox_list_2)):
        instance_box_data=bbox_list_2[j]
        box2 = Box(center=instance_box_data[:3], size=instance_box_data[3:6],
                          orientation=Quaternion(instance_box_data[6:]))
        box2.wlh = (1+margin)*box2.wlh
        distances=np.linalg.norm(bbox_centers-box2.center,axis=1)
        if np.min(distances)<thresh: #threshold
            corr_1=np.argmin(distances)
        else:
            continue
        
        new_bboxes_aux[corr_1]=instance_box_data[:3]

        box1=refer_bbox_list[corr_1]

        box_pose1 = SE3(rotation=box1.rotation_matrix, translation=np.array(box1.center))
        box_pose2 = SE3(rotation=box2.rotation_matrix, translation=np.array(box2.center))
        relative_pose_1_2 = box_pose2.right_multiply_with_se3(box_pose1.inverse())
        inbox_pc1=pc1[mask_tracks_flow_temp[corr_1]]
        mask1_tracks_flow.append(mask_tracks_flow_temp[corr_1])
        inbox_pc1_t = relative_pose_1_2.transform_point_cloud(inbox_pc1)

        translation = inbox_pc1_t - inbox_pc1
        
        flow[mask_tracks_flow_temp[corr_1], :] = translation
                
        bbox2_3d = box2.corners().T 
        inbox_pc2, is_valid2 = filter_point_cloud_to_bbox_3D_vectorized(bbox2_3d, pc2)
        mask2_tracks_flow.append(np.where(is_valid2 == True)[0])
    
    return flow,np.concatenate(mask1_tracks_flow),np.concatenate(mask2_tracks_flow),new_bboxes_aux