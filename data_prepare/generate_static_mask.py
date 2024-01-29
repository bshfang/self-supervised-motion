from tqdm import tqdm
import os
import random
import copy
import json
import cv2
import argparse
from functools import reduce
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime
from pypcd import pypcd
from PIL import Image

import torch

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights,Raft_Small_Weights
from torchvision.models.optical_flow import raft_large,raft_small

from datasets.data_utils import mask_points_by_range, get_nusc_pc


import pickle
class pObject(object):
    def __init__(self):
        pass


def calculate_interval(timestamp1, timestamp2):
    timestamp1 = datetime.fromtimestamp(float(timestamp1) / 1e6)
    timestamp2 = datetime.fromtimestamp(float(timestamp2) / 1e6)
    interval = (timestamp2 - timestamp1).total_seconds()
    return interval


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


def generate_static_mask():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='5', help='gt_thresh')
    parser.add_argument('--sample_interval', nargs='+', type=int, help='sample_interval')
    parser.add_argument('--sf_thresh', type=float, default=1, help='all_sf_thresh')
    parser.add_argument('--optf_thresh', type=float, default=5, help='all_optf_thresh')
    parser.add_argument('--train_data_root', type=str, default='/path/to/save/processed/dataset/train', help='train_data_root')
    parser.add_argument('--save_path', type=str, default='/path/to/save/processed/static/mask', help='save_path')
    args = parser.parse_args()

    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    nusc_path = '/path/to/nuScenes/data/'

    # === parameters ===
    fnb_thresh = 4.0  # default: None
    sf_thresh = args.sf_thresh
    optf_thresh = args.optf_thresh
    save_path = args.save_path
    sample_idxs = args.sample_interval
    train_data_root = args.train_data_root
    print(fnb_thresh, sf_thresh, optf_thresh, save_path, sample_idxs, train_data_root)
    # return
    # ==================

    # optical flow model parameters and set up
    img_size = [896,1600]
    model_size = 'l'
    if model_size=='s':
        model_weights = Raft_Small_Weights.DEFAULT
    else:
        model_weights = Raft_Large_Weights.C_T_SKHT_K_V2

    samples = []
    for d in os.listdir(train_data_root):
        samples.append(d)
    print(len(samples))
    # return

    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    cam_use_frames = [3, 3, 3, 1, 3, 3]
    img_width = 1600
    img_height = 900

    nusc_version = 'v1.0-trainval'
    nusc = NuScenes(version=nusc_version, dataroot=nusc_path, verbose=True)
    print(len(samples))

    # for sample_name in tqdm(samples[sample_idxs[0]:sample_idxs[1]]):
    for sample_name in tqdm(samples):

        with open(os.path.join(train_data_root,sample_name), 'rb') as fp:
            data = np.load(fp, allow_pickle=True)
            data = data.item()

        save_file = os.path.join(save_path, sample_name)
        save_file = save_file.replace('.npy', '.npz')
        if os.path.exists(save_file):
            continue

        # get curr point cloud
        past_frame = data['past_frame']
        cur_pc_file = data['past_pc_files'][past_frame-1]
        trans_mat = data['past_transform'][past_frame-1]
        curr_sd_token = data['all_sample_data_tokens'][past_frame-1]

        pcd = get_nusc_pc(cur_pc_file)
        pcd = pcd[:, :3]
        masked_pcd, range_mask = mask_points_by_range(pcd, limit_range=[-32.0, -32.0, -3.0, 32.0, 32.0, 2.0])
        pcd = pcd.T  # 3,n
        pcd_T = pcd.T

        # === get ground mask ===
        pc_file_noground = cur_pc_file.replace('/LIDAR_TOP/', '/LIDAR_TOP_noground/')[:-4]
        pcd_2 = get_nusc_pc(pc_file_noground)
        pcd_2 = pcd_2[:, :3]
        pcd_2 = pcd_2.T
        foreground_mask = np.isin(pcd.T, pcd_2.T).any(axis=1)

        gt_dict = data['flow_gt']
        dims = (256, 256, 1)
        num_future_pcs = data['future_frame']
        pixel_indices = gt_dict['pixel_indices']

        sparse_disp_field_gt = gt_dict['disp_field_gt']
        all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
        all_disp_field_gt[:, pixel_indices[:, 1], pixel_indices[:, 0], :] = sparse_disp_field_gt[:]  # y,x

        lidar_range = [-32.0, -32.0, -3.0, 32.0, 32.0, 3.0]
        voxel_size = [0.25, 0.25, 6]
        nx = int((lidar_range[3] - lidar_range[0]) / voxel_size[0])
        current_pc_ = masked_pcd
        coord_x = np.floor((current_pc_[:, 0:1] - lidar_range[0]) / voxel_size[0])
        coord_y = np.floor((current_pc_[:, 1:2] - lidar_range[1]) / voxel_size[1])
        coord = np.concatenate([coord_x, coord_y], axis=1)
        pidx = coord[:, 1] * nx + coord[:, 0]
        pidx = pidx.astype(int)
        # =======================

        # print(foreground_mask.shape)

        curr_sd = nusc.get('sample_data', curr_sd_token)
        curr_sample_token = curr_sd['sample_token']
        curr_sample = nusc.get('sample', curr_sample_token)

        # === generate delta flow ===
        cam_point_optf = np.zeros((5, 6, pcd.shape[1], 2))  # 5,6,n,3
        cam_ego_optf = np.zeros((5, 6, pcd.shape[1], 2))  # 5,6,n,3
        cam_point_optf2sf = np.zeros((5, 6, pcd.shape[1], 2))  # 5,6,n,3
        cam_point_egof2sf = np.zeros((5, 6, pcd.shape[1], 2))  # 5,6,n,3
        
        all_in_cam = np.zeros((6, pcd.shape[1],), dtype=np.bool)
        sf_all_cam_static_mask = np.zeros((6, pcd.shape[1],), dtype=np.bool)
        opt_all_cam_static_mask = np.zeros((6, pcd.shape[1],), dtype=np.bool)

        for cam_i, cam_name in enumerate(cam_names):
            cam_frame = cam_use_frames[cam_i]
            # print(cam_name, cam_frame)

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

            all_in_cam[cam_i, :] = cur_in_cam

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
                # default
                predicted_flows = calc_flow(img1_batch, img2_batch, model_size)
                predicted_flows = resize_flow(predicted_flows, (900, 1600))
                predicted_flow_np = predicted_flows[0].detach().cpu().numpy().transpose([1,2,0])
                # ========================

                # 1. fill raw optical flow value to point flow (point_optf)
                cam_point_optf[seq_i, cam_i, cur_in_cam, :2] = predicted_flow_np[curr_p_coord[cur_in_cam, 1], curr_p_coord[cur_in_cam, 0]]

                # 2. calculate the ego_motion flow; calculate ego_flow-opt_flow; fill the optf_delta to point flow 2
                ego_flow = p_cf.T - curr_p_cf.T
                cam_ego_optf[seq_i, cam_i, cur_in_cam, :2] = ego_flow[cur_in_cam, :2]

                # 3. project optical flow to scene flow
                optf_end_point = (curr_p_cf.T)[cur_in_cam, :2] + predicted_flow_np[curr_p_coord[cur_in_cam, 1], curr_p_coord[cur_in_cam, 0]]
                egof_end_point = (p_cf.T)[cur_in_cam, :2]

                pc_z = pcd[2, cur_in_cam]
                transform_mat = np.concatenate((curr_cam_intrinsic, np.zeros((3,1))), axis=1) @ curr_p2cam
                # numpy to torch
                optf_end_point = torch.tensor(optf_end_point, dtype=torch.float64)
                egof_end_point = torch.tensor(egof_end_point, dtype=torch.float64)
                pc_z = torch.tensor(pc_z, dtype=torch.float64)
                transform_mat = torch.tensor(transform_mat, dtype=torch.float64)

                vis_point_num = optf_end_point.shape[0]
                transform_mat = transform_mat.unsqueeze(0).repeat(vis_point_num,1,1)
                transform_mat[:, :, 2] = transform_mat[:, :, 2] * pc_z.unsqueeze(1)
                transform_mat[:, :, 2] = transform_mat[:, :, 2] + transform_mat[:, :, 3]
                transform_mat = transform_mat[:, :, :3]
                transform_mat_inv = torch.linalg.inv(transform_mat)

                # optical flow projection
                optf_end_point = torch.cat((optf_end_point, torch.ones((vis_point_num,1))), dim=1).unsqueeze(2)
                world_coord = transform_mat_inv @ optf_end_point
                world_coord = world_coord.squeeze()
                world_coord[:, :2] = world_coord[:, :2] / (world_coord[:, 2:])
                cam_point_optf2sf[seq_i, cam_i, cur_in_cam, :2] = world_coord[:, :2] - pcd[:2, cur_in_cam].T

                # ego flow
                egof_end_point = torch.cat((egof_end_point, torch.ones((vis_point_num,1))), dim=1).unsqueeze(2)
                world_coord = transform_mat_inv @ egof_end_point
                world_coord = world_coord.squeeze()
                world_coord[:, :2] = world_coord[:, :2] / world_coord[:, 2:]
                cam_point_egof2sf[seq_i, cam_i, cur_in_cam, :2] = world_coord[:, :2] - pcd[:2, cur_in_cam].T

                # delta flow
                interval = abs(calculate_interval(curr_cam_sd['timestamp'], cam_sd['timestamp']))
                if sf_thresh != -1.0:
                    sf_sd_thresh = sf_thresh * 2 * interval
                    delta_sf = cam_point_optf2sf[seq_i, cam_i, cur_in_cam, :2] - cam_point_egof2sf[seq_i, cam_i, cur_in_cam, :2]
                    cam_static_mask = np.linalg.norm(delta_sf, ord=2, axis=1) < sf_sd_thresh
                    sf_all_cam_static_mask[cam_i, cur_in_cam] = cam_static_mask
                if optf_thresh != -1.0:
                    optf_sd_thresh = optf_thresh * 2 * interval
                    delta_optf = cam_point_optf[seq_i, cam_i, cur_in_cam, :2] - cam_ego_optf[seq_i, cam_i, cur_in_cam, :2]
                    opt_cam_static_mask = np.linalg.norm(delta_optf, ord=2, axis=1) < optf_sd_thresh
                    opt_all_cam_static_mask[cam_i, cur_in_cam] = opt_cam_static_mask

        # ============================

        # generate static mask from 
        all_static_mask = np.ones((pcd.shape[1],), dtype=np.bool)
        for cam_i, _ in enumerate(cam_names):
            in_cam = all_in_cam[cam_i]
            if sf_thresh != -1 and optf_thresh != -1:
                sf_cam_static_mask = sf_all_cam_static_mask[cam_i, in_cam]
                opt_cam_static_mask = opt_all_cam_static_mask[cam_i, in_cam]
                cam_static_mask = np.logical_and(sf_cam_static_mask, opt_cam_static_mask)
                in_cam_mask = all_static_mask[in_cam]
                in_cam_mask[cam_static_mask] = 0
                all_static_mask[in_cam] = in_cam_mask
            elif sf_thresh == -1 and optf_thresh != -1:
                opt_cam_static_mask = opt_all_cam_static_mask[cam_i, in_cam]
                in_cam_mask = all_static_mask[in_cam]
                in_cam_mask[opt_cam_static_mask] = 0
                all_static_mask[in_cam] = in_cam_mask
            elif sf_thresh != -1 and optf_thresh == -1:
                sf_cam_static_mask = sf_all_cam_static_mask[cam_i, in_cam]
                in_cam_mask = all_static_mask[in_cam]
                in_cam_mask[sf_cam_static_mask] = 0
                all_static_mask[in_cam] = in_cam_mask
        
        # fnb_mask
        if fnb_thresh is not None:
            pcd_x = pcd_T[:, 0]  # n,
            pcd_y = pcd_T[:, 1]  # n, 
            fnb_mask = np.abs(pcd_y) > fnb_thresh * np.abs(pcd_x)
            all_static_mask[fnb_mask] = 1.0
        predicted_static_mask = np.copy(all_static_mask)

        # pillar static expand
        all_indexes = np.unique(pidx)
        pillar_static_mask = np.copy(predicted_static_mask)
        masked_pillar_static_mask = pillar_static_mask[range_mask]
        for index in all_indexes:
            idx_mask = pidx == index
            pillar_static = masked_pillar_static_mask[idx_mask]
            static_ratio = 1 - np.sum(pillar_static) / len(pillar_static)
            if static_ratio > 0.6:
                masked_pillar_static_mask[idx_mask] = 0
        pillar_static_mask[range_mask] = masked_pillar_static_mask

        # use_ground_mask:
        ground_mask = np.logical_not(foreground_mask)
        all_static_mask[ground_mask] = 0

        np.savez(save_file,
                 static_mask=all_static_mask,
                 fg_mask=foreground_mask,
                 predicted_static_mask=predicted_static_mask,
                 pillar_static_mask=pillar_static_mask)


if __name__=='__main__':
    generate_static_mask()