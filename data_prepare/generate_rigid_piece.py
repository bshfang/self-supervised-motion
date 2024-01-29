from tqdm import tqdm
import os
import random
import copy
import json
import argparse
import shutil
import cv2
from functools import reduce
import numpy as np
from pyquaternion import Quaternion
from collections import Counter
from datetime import datetime
from matplotlib import pyplot as plt
from pypcd import pypcd
from PIL import Image

import torch

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights,Raft_Small_Weights
from torchvision.models.optical_flow import raft_large,raft_small
from torchvision.utils import flow_to_image

from datasets.data_utils import mask_points_by_range, get_nusc_pc

import pickle
class pObject(object):
    def __init__(self):
        pass


def get_slic_result(img,
                    save_path=None, 
                    region_size=80, 
                    ruler=5.0, 
                    iterate_time=30,
                    algorithm='slic'):
    """
    img: np.array (h,w,3)
    algorithm: slic or slico or mslic
    """

    #img = cv2.imread(img_path)
    algorithms = {'slic':cv2.ximgproc.SLIC, 'slico':cv2.ximgproc.SLICO, 'mslic':cv2.ximgproc.MSLIC}
    if algorithm:
        slic = cv2.ximgproc.createSuperpixelSLIC(img,region_size=region_size,ruler=ruler, algorithm=algorithms[algorithm])
    else:
        slic = cv2.ximgproc.createSuperpixelSLIC(img,region_size=region_size,ruler=ruler)
    slic.iterate(iterate_time)
    mask_slic = slic.getLabelContourMask() 
    label_slic = slic.getLabels()        
    number_slic = slic.getNumberOfSuperpixels()  
    mask_inv_slic = cv2.bitwise_not(mask_slic)  
    img_slic = cv2.bitwise_and(img,img,mask=mask_inv_slic)
    if save_path:
        cv2.imwrite(save_path,img_slic)
    return mask_slic, label_slic, number_slic


def preprocess(img1_batch, img2_batch,model_weights,img_size=[224, 400]):
    transforms = model_weights.transforms()
    img1_batch = F.resize(img1_batch, size=img_size, antialias=False)
    img2_batch = F.resize(img2_batch, size=img_size, antialias=False)
    return transforms(img1_batch, img2_batch)


def resize_flow(flow, new_shape):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                           mode='bilinear', align_corners=True)
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow


def calc_flow(img1_batch, img2_batch, size='s'):
    '''
    args:
        img1_batch:[B,3,H,W]
        img2_batch:[B,3,H,W]
    return:
        predicted_flow:[B,2,H,W]
    '''
    if size=='s':       
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).cuda()
    else:
        model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_K_V2, progress=False).cuda()  # Raft_Large_Weights.DEFAULT
    model = model.eval()
    
    with torch.no_grad():
        list_of_flows = model(img1_batch.cuda(), img2_batch.cuda())
    
    # print(f"type = {type(list_of_flows)}")
    # print(f"length = {len(list_of_flows)} = number of iterations of the model")
    
    predicted_flows = list_of_flows[-1]
    
    return predicted_flows


def generate_rigid_piece():
    """
    generate instance rigid piece
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='5', help='gt_thresh')
    parser.add_argument('--sample_interval', nargs='+', type=int, help='sample_interval')
    parser.add_argument('--train_data_root', type=str, default='/path/to/save/processed/dataset/train', help='train_data_root')
    parser.add_argument('--save_path', type=str, default='/path/to/save/processed/rigid/piece', help='save_path')
    parser.add_argument('--static_mask_root', type=str, default='/path/to/save/processed/static/mask', help='static_mask_root')
    args = parser.parse_args()
    
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    # === parameters ===
    train_data_root = args.train_data_root
    save_path = args.save_path
    static_mask_root = args.static_mask_root
    sample_idxs = args.sample_interval
    
    dis_thresh = 2
    slic_region_size = 80
    slic_ruler = 0.1

    fuse_thresh = 0.1 
    static_ratio_thresh = 0.15
    filter_piece_num_thresh = 4

    print('static mask root:', static_mask_root)
    print('save_path:', save_path)
    print('slic region {}, ruler {}, static_ratio_thresh {}'.format(slic_region_size, slic_ruler, static_ratio_thresh))
    print('samples', sample_idxs)
    # ==================

    # load sample data
    samples = []
    for d in os.listdir(train_data_root):
        samples.append(d)
    print(len(samples))

    # optical flow model parameters and set up
    img_size = [896,1600]
    model_size = 'l'
    if model_size=='s':
        model_weights = Raft_Small_Weights.DEFAULT
    else:
        # model_weights = Raft_Large_Weights.DEFAULT
        model_weights = Raft_Large_Weights.C_T_SKHT_K_V2

    # nuscenes
    nusc_path = '/GPFS/public/nuScenes_0924/nuScenes/'
    nusc_version = 'v1.0-trainval'
    nusc = NuScenes(version=nusc_version, dataroot=nusc_path, verbose=True)
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    cam_use_frames = [3, 3, 3, 1, 3, 3]
    img_width = 1600
    img_height = 900

    # for sample_name in tqdm(samples[sample_idxs[0]:sample_idxs[1]]):
    for sample_name in tqdm(samples):
        with open(os.path.join(train_data_root, sample_name), 'rb') as fp:
            data = np.load(fp, allow_pickle=True)
            data = data.item()

        save_file = os.path.join(save_path, sample_name)
        save_file = save_file.replace('.npy', '.npz')
        if os.path.exists(save_file):
            continue

        # get curr point cloud
        cur_pc_file = data['past_pc_files'][2]
        trans_mat = data['past_transform'][2]
        curr_sd_token = data['all_sample_data_tokens'][2]
        # return
        pcd = get_nusc_pc(cur_pc_file, transform_mat=trans_mat)

        pcd, range_mask = mask_points_by_range(pcd, limit_range=[-32.0, -32.0, -3.0, 32.0, 32.0, 2.0])
        pcd = pcd[:, :3]
        pcd = pcd.T  # 3,n
        pcd_T = pcd.T

        static_mask_file = os.path.join(static_mask_root, curr_sd_token+'.npz')
        static_mask_data = np.load(static_mask_file)
        static_mask = static_mask_data['static_mask']
        fg_mask = static_mask_data['fg_mask']
        predicted_static_mask = static_mask_data['predicted_static_mask']
        pillar_static_mask = static_mask_data['pillar_static_mask']

        static_mask = static_mask[range_mask]
        fg_mask = fg_mask[range_mask]
        predicted_static_mask = predicted_static_mask[range_mask]
        pillar_static_mask = pillar_static_mask[range_mask]

        # === get ground truth ===
        lidar_range = [-32.0, -32.0, -3.0, 32.0, 32.0, 3.0]
        voxel_size = [0.25, 0.25, 6]
        nx = int((lidar_range[3] - lidar_range[0]) / voxel_size[0])
        current_pc_ = pcd.T
        coord_x = np.floor((current_pc_[:, 0:1] - lidar_range[0]) / voxel_size[0])
        coord_y = np.floor((current_pc_[:, 1:2] - lidar_range[1]) / voxel_size[1])
        coord = np.concatenate([coord_x, coord_y], axis=1)
        pidx = coord[:, 1] * nx + coord[:, 0]
        pidx = pidx.astype(int)
        # ========================

        # current lidar pose
        curr_sd = nusc.get('sample_data', curr_sd_token)
        curr_sample = nusc.get('sample', curr_sd['sample_token'])

        # cam slic results
        all_cam_slic_results = []
        all_in_cam_masks = []

        for cam_i, cam_name in enumerate(cam_names):
            cam_frame = cam_use_frames[cam_i]

            curr_cam_sd_token = curr_sample['data'][cam_name]
            curr_cam_sd = nusc.get('sample_data', curr_cam_sd_token)

            # -2,-1,0,1,2 sample_data
            prev_cam_sd_token = curr_cam_sd['prev']
            prev_cam_sd = nusc.get('sample_data', prev_cam_sd_token)
            prev_prev_cam_sd_token = prev_cam_sd['prev']
            prev_prev_cam_sd = nusc.get('sample_data', prev_prev_cam_sd_token)

            next_cam_sd_token = curr_cam_sd['next']
            next_cam_sd = nusc.get('sample_data', next_cam_sd_token)
            next_next_cam_sd_token = next_cam_sd['next']
            next_next_cam_sd = nusc.get('sample_data', next_next_cam_sd_token)
            all_cam_sds = [prev_prev_cam_sd, prev_cam_sd, curr_cam_sd, next_cam_sd, next_next_cam_sd]

            curr_im_path = os.path.join(nusc.dataroot, curr_cam_sd['filename'])
            curr_im = Image.open(curr_im_path)

            # cur camera params
            lidar_record = nusc.get('calibrated_sensor', curr_sd['calibrated_sensor_token'])
            lidar_ego_pose = nusc.get('ego_pose', curr_sd['ego_pose_token'])
            car_from_global = transform_matrix(
                lidar_ego_pose["translation"],
                Quaternion(lidar_ego_pose["rotation"]),
                inverse=True,
            )
            ref_from_car = transform_matrix(
                lidar_record["translation"], Quaternion(lidar_record["rotation"]),
                inverse=True,
            )

            cam_record = nusc.get('calibrated_sensor', curr_cam_sd['calibrated_sensor_token'])
            cam_ego_pose = nusc.get('ego_pose', curr_cam_sd['ego_pose_token'])
            cam_from_car = transform_matrix(
                    cam_record["translation"], Quaternion(cam_record["rotation"]),
                    inverse=True,
                )
            cam_car_from_global = transform_matrix(
                    cam_ego_pose['translation'],
                    Quaternion(cam_ego_pose['rotation']),
                    inverse=True,
                    )
            curr_p2cam = reduce(np.dot, [
                        cam_from_car, cam_car_from_global,
                        np.linalg.inv(car_from_global),
                        np.linalg.inv(ref_from_car)  # transform from lidar coordinate to image coordinate
                    ])
            cam_path, _, curr_cam_intrinsic = nusc.get_sample_data(curr_cam_sd['token'])
            p_cf = curr_p2cam @ np.vstack((pcd, np.ones(pcd.shape[1])))
            p_cf = curr_cam_intrinsic @ p_cf[:3, :]
            p_cf[:2] = p_cf[:2] / (p_cf[2:3, :] + 1e-8)
            cur_in_cam = reduce(np.logical_and, (p_cf[0, :] > 1, p_cf[1, :] > 1, 
                                            p_cf[0, :] < img_width - 1, p_cf[1,:] < img_height - 1, p_cf[2,:] >= 1))
            curr_p_cf = p_cf
            curr_p_coord = np.around(curr_p_cf).astype(np.int32).T

            for seq_i, cam_sd in enumerate(all_cam_sds):
                if seq_i != cam_frame:
                    continue

                # first, calculate the lidar2cam matrix
                cam_record = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
                cam_ego_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
                cam_from_car = transform_matrix(
                        cam_record["translation"], Quaternion(cam_record["rotation"]),
                        inverse=True,
                    )
                cam_car_from_global = transform_matrix(
                        cam_ego_pose['translation'],
                        Quaternion(cam_ego_pose['rotation']),
                        inverse=True,
                        )
                p2cam = reduce(np.dot, [
                            cam_from_car, cam_car_from_global,
                            np.linalg.inv(car_from_global),
                            np.linalg.inv(ref_from_car)  # transform from lidar coordinate to image coordinate
                        ])
                
                # second, project current point (t=0) to cam
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_sd['token'])
                p_cf = p2cam @ np.vstack((pcd, np.ones(pcd.shape[1])))
                p_cf = cam_intrinsic @ p_cf[:3, :]
                p_cf[:2] = p_cf[:2] / (p_cf[2:3, :] + 1e-8)

                # === get optical flow ===
                im = Image.open(cam_path)
                img1_batch, img2_batch = preprocess(curr_im, im, model_weights, img_size)
                img1_batch=img1_batch.unsqueeze(0)
                img2_batch=img2_batch.unsqueeze(0)
                predicted_flows = calc_flow(img1_batch, img2_batch, model_size)
                predicted_flows = resize_flow(predicted_flows, (900, 1600))
                predicted_flow_np = predicted_flows[0].detach().cpu().numpy().transpose([1,2,0])

                flow_imgs = flow_to_image(torch.from_numpy(predicted_flow_np).permute(2,0,1))
                # print(flow_imgs.shape)
                flow_img = flow_imgs.cpu().numpy().transpose([1,2,0])
                # ========================

                # slic over-segment
                slic_mask, slic_label, slic_number = get_slic_result(flow_img,
                                                                     region_size=slic_region_size,
                                                                     ruler=slic_ruler)
                # get the slic label of pointclouds
                pc_slic_label = np.zeros((pcd.shape[1], )) - 1
                pc_slic_label[cur_in_cam] = slic_label[curr_p_coord[cur_in_cam,1], curr_p_coord[cur_in_cam,0]]

                # === exclude occluded points from each slic point ===
                slic_labels = np.unique(pc_slic_label)
                # print(len(slic_labels))
                for slic_label in slic_labels:
                    if slic_label==-1:
                        continue
                    pc_slic_mask = pc_slic_label == slic_label
                    
                    non_ground_slic_mask = np.logical_and(pc_slic_mask, fg_mask)
                    non_ground_slic_pc = pcd_T[non_ground_slic_mask, :2]  # n,2

                    if len(non_ground_slic_pc) == 0:
                        continue

                    # for simplicity let the camera posiiton be (0,0) (get the camera position)
                    slic_pc_distance = np.linalg.norm(non_ground_slic_pc, axis=1)
                    min_distance = np.min(slic_pc_distance)

                    # set all points in a piece that have the distance (> d+dis_thresh) to -1
                    slic_pc = pcd_T[pc_slic_mask, :2]  # n,2
                    all_slic_pc_distance = np.linalg.norm(slic_pc, axis=1)
                    all_occluded_slic_mask = all_slic_pc_distance > (min_distance + dis_thresh)
                    new_pc_slic_label = np.full(all_occluded_slic_mask.shape, slic_label)
                    new_pc_slic_label[all_occluded_slic_mask] = -1.0
                    pc_slic_label[pc_slic_mask] = new_pc_slic_label
                # ====================================================

                # === aggregate slic result for each camera ===
                all_cam_slic_results.append(pc_slic_label)
                all_in_cam_masks.append(cur_in_cam)
                # =============================================

        # === generate fused rigid piece label ===
        aggregated_slic_label = np.zeros((pcd_T.shape[0],)) - 1

        new_slic_label = 0.0
        for cam_i in range(6):
            pc_slic_label = all_cam_slic_results[cam_i]  # n, float64
            in_cam_mask = all_in_cam_masks[cam_i]  # n, bool

            non_ground_pc_slic_label = np.copy(pc_slic_label)
            non_ground_pc_slic_label[np.logical_not(fg_mask)] = -1.0

            slic_labels = np.unique(non_ground_pc_slic_label)

            # --- fuse multi-view ---
            # fuse_thresh = 0.1  # ratio threshold
            for slic_label in slic_labels:
                if slic_label==-1:
                    continue
                pc_slic_mask = non_ground_pc_slic_label == slic_label

                exist_slic_values = aggregated_slic_label[pc_slic_mask]
                exist_slic_labels = np.unique(exist_slic_values)

                for exist_slic_label in exist_slic_labels:
                    if exist_slic_label == -1.0:
                        continue
                    overlap_ratio = np.count_nonzero(exist_slic_values==exist_slic_label) / np.count_nonzero(aggregated_slic_label==exist_slic_label)
                    if overlap_ratio > fuse_thresh:
                        aggregated_slic_label[aggregated_slic_label==exist_slic_label] = new_slic_label

                aggregated_slic_label[pc_slic_mask] = new_slic_label
                
                new_slic_label += 1
            # ------------------------
        
        # --- fuse height (pillar) ---
        all_indexes = np.unique(pidx)
        for index in all_indexes:
            idx_mask = pidx == index

            pidx_slic_values = aggregated_slic_label[idx_mask]

            pidx_slic_labels = np.unique(pidx_slic_values)
            if np.any(pidx_slic_labels != -1):
                new_slic_label= np.random.choice(pidx_slic_labels[pidx_slic_labels != -1])

                for slic_label in pidx_slic_labels:
                    if slic_label == -1:
                        continue
                    aggregated_slic_label[aggregated_slic_label==slic_label] = new_slic_label
        # ----------------------------

        # --- filter piece with few points ---
        slic_labels = np.unique(aggregated_slic_label)
        # print('filter_piece_num_thresh', filter_piece_num_thresh)
        for slic_label in slic_labels:
            if slic_label==-1:
                continue
            pc_slic_mask = aggregated_slic_label == slic_label

            if np.sum(pc_slic_mask) <= filter_piece_num_thresh:
                aggregated_slic_label[pc_slic_mask] = -1
        # ------------------------------------

        # --- generate rigid piece for non-static point ---
        non_static_aggregated_slic_label = np.copy(aggregated_slic_label)
        static_slic_num = 0
        static_slic_pc_num = 0
        slic_labels = np.unique(aggregated_slic_label)

        for slic_label in slic_labels:
            if slic_label==-1:
                continue
            pc_slic_mask = aggregated_slic_label == slic_label

            slic_static_label = pillar_static_mask[pc_slic_mask]
            static_ratio = np.sum(slic_static_label==False) / len(slic_static_label)
            if static_ratio > static_ratio_thresh:
                static_slic_num += 1
                static_slic_pc_num += len(slic_static_label)

                non_static_aggregated_slic_label[pc_slic_mask] = -1
        # -------------------------------------------------
        # ========================================

        # save rigid piece result
        np.savez(save_file,
                 aggregated_slic_label=aggregated_slic_label,
                 non_static_aggregated_slic_label=non_static_aggregated_slic_label,)

    # print("samples {} to {} done".format(sample_idxs[0], sample_idxs[1]))


if __name__=='__main__':
    generate_rigid_piece()