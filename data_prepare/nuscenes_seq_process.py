import json
from datetime import datetime
from tqdm import tqdm
import os
import numpy as np
from pyquaternion import Quaternion
from collections import Counter
from datetime import datetime
from matplotlib import pyplot as plt
from pypcd import pypcd

import torch
import torch.utils.data as data

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix
import nuscenes.utils.splits as splits

from canvas_bev import Canvas_BEV, Canvas_BEV_heading_right
from data_utils import mask_points_by_range, read_pcd, point_in_hull_fast, calc_displace_vector


class MyLidarPointCloud(LidarPointCloud):
    def get_ego_mask(self):
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5),
        )
        return ego_mask


def get_nusc_pc(pc_file, transform_mat):
    """
    get lidar point cloud of nuscenes frame
    """
    if pc_file.endswith('.pcd.bin'):
        lidar_pc = MyLidarPointCloud.from_file(pc_file)
        ego_mask = lidar_pc.get_ego_mask()
        lidar_pc.points = lidar_pc.points[:, np.logical_not(ego_mask)]

        lidar_pc.transform(transform_mat)

        points_tf = np.array(lidar_pc.points[:4].T, dtype=np.float32)
    elif pc_file.endswith('.pcd'):
        pcd = read_pcd(pc_file)
        lidar_pc = MyLidarPointCloud(pcd.T)
        ego_mask = lidar_pc.get_ego_mask()
        lidar_pc.points = lidar_pc.points[:, np.logical_not(ego_mask)]
        lidar_pc.transform(transform_mat)
        points_tf = np.array(lidar_pc.points[:4].T, dtype=np.float32)
    else:
        return None
    
    return points_tf


def get_global_pose(nusc, sd_token, inverse=False):
    sd = nusc.get("sample_data", sd_token)
    sd_ep = nusc.get("ego_pose", sd["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    if inverse is False:
        global_from_ego = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False
        )
        ego_from_sensor = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
        )
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True
        )
        ego_from_global = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True
        )
        pose = sensor_from_ego.dot(ego_from_global)

    return pose


def get_ego_global_pose(nusc, sd_token, inverse=False):
    sd = nusc.get("sample_data", sd_token)
    sd_ep = nusc.get("ego_pose", sd["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    if inverse is False:
        ego_from_sensor = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
        )
        pose = ego_from_sensor
    else:
        ego_from_global = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True
        )
        pose = ego_from_global

    return pose


def get_ann_of_instance(nusc, sample_record, instance_token):
    instance_ann_token = None
    cnt = 0
    for ann_token in sample_record['anns']:
        ann_record = nusc.get('sample_annotation', ann_token)
        if ann_record['instance_token'] == instance_token:
            instance_ann_token = ann_token
            cnt += 1

    assert cnt <= 1, 'One instance cannot associate more than 1 annotations.'

    if cnt == 1:
        return instance_ann_token
    else:
        return ""


def get_nusc_instance_box(nusc, sd_record, instance_token):
    curr_sample_record = nusc.get('sample', sd_record['sample_token'])

    # Get reference pose and timestamp
    ref_pose_rec = nusc.get('ego_pose', sd_record['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    ref_time = 1e-6 * sd_record['timestamp']

    instance_ann_token = get_ann_of_instance(nusc, curr_sample_record, instance_token)

    if instance_ann_token == "":  # no annotation for this instance in this sample
        return None, None, None
    
    sample_ann_rec = nusc.get('sample_annotation', instance_ann_token)

    # Get the attribute of this annotation
    if len(sample_ann_rec['attribute_tokens']) != 0:
        attr = nusc.get('attribute', sample_ann_rec['attribute_tokens'][0])['name']
    else:
        attr = None

    # Get the category of this annotation
    cat = sample_ann_rec['category_name']

    if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
        # If no previous annotations available, or if sample_data is keyframe just return the current ones.
        box = nusc.get_box(instance_ann_token)
    else:
        prev_sample_record = nusc.get('sample', curr_sample_record['prev'])
        curr_ann_rec = nusc.get('sample_annotation', instance_ann_token)
        prev_ann_recs = [nusc.get('sample_annotation', token) for token in prev_sample_record['anns']]
        prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

        if instance_token in prev_inst_map:
            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']
            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))
            prev_ann_rec = prev_inst_map[instance_token]
            # Interpolate center.
            center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                            curr_ann_rec['translation'])]

            # Interpolate orientation.
            rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                        q1=Quaternion(curr_ann_rec['rotation']),
                                        amount=(t - t0) / (t1 - t0))

            box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                        token=curr_ann_rec['token'])
        else:
            box = nusc.get_box(curr_ann_rec['token'])

    return box, attr, cat


def gen_2d_grid_gt(past_pcs, grid_size, extents, instance_infos, proportion_thresh=0.5, future_frame_num=2):
    # -- Reorg instance boxes
    past_frame_num = len(past_pcs)
    num_instances = len(instance_infos)
    all_boxes_list = []
    ref_boxes_list = []
    for i, instance in enumerate(instance_infos):
        all_boxes_list.append(instance_infos[instance]['boxes'])
        ref_boxes_list.append(instance_infos[instance]['ref_box'])
        
    # future_frame_num = len(all_boxes_list[0]) - past_frame_num

    # ----------------------------------------------------
    # Filter and sort the reference point cloud
    refer_pc = past_pcs[-1]
    refer_pc = refer_pc[:, 0:3]
    filter_idx = np.where((extents[0, 0] < refer_pc[:, 0]) & (refer_pc[:, 0] < extents[0, 1]) &
                        (extents[1, 0] < refer_pc[:, 1]) & (refer_pc[:, 1] < extents[1, 1]) &
                        (extents[2, 0] < refer_pc[:, 2]) & (refer_pc[:, 2] < extents[2, 1]))[0]
    refer_pc = refer_pc[filter_idx]
    discrete_pts = np.floor(refer_pc[:, 0:2] / grid_size).astype(np.int32)

    # -- Use Lex Sort, sort by x, then y
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    sorted_order = np.lexsort((y_col, x_col))
    refer_pc = refer_pc[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    contiguous_array = np.ascontiguousarray(discrete_pts).view(
    np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # -- The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # -- Sort unique indices to preserve order
    unique_indices.sort()
    pixel_coords = discrete_pts[unique_indices]

    # -- Number of points per voxel, last voxel calculated separately
    num_points_in_pixel = np.diff(unique_indices)
    num_points_in_pixel = np.append(num_points_in_pixel, discrete_pts.shape[0] - unique_indices[-1])

    # -- Compute the minimum and maximum voxel coordinates
    min_pixel_coord = np.floor(extents.T[0, 0:2] / grid_size)
    max_pixel_coord = np.ceil(extents.T[1, 0:2] / grid_size) - 1

    # -- Get the voxel grid dimensions
    num_divisions = ((max_pixel_coord - min_pixel_coord) + 1).astype(np.int32)

    # -- Bring the min voxel to the origin
    pixel_indices = (pixel_coords - min_pixel_coord).astype(int)
    # ----------------------------------------------------
    
    # ----------------------------------------------------
    # Get the point cloud subsets, which are inside different instance bounding boxes
    refer_pc_idx_per_bbox = list()
    pixel_instance_id = np.zeros(pixel_indices.shape[0], dtype=np.uint8)
    points_instance_id = np.zeros(refer_pc.shape[0], dtype=np.int)
    for i in range(num_instances):
        instance_ref_box = ref_boxes_list[i]
        instance_boxes = all_boxes_list[i]
        idx = point_in_hull_fast(refer_pc[:, 0:3], instance_ref_box)
        # print(idx.shape)
        refer_pc_idx_per_bbox.append(idx)
        points_instance_id[idx] = i + 1  # object id starts from 1, background has id 0
    
    if len(refer_pc_idx_per_bbox) > 0:
        refer_pc_idx_inside_box = np.concatenate(refer_pc_idx_per_bbox).tolist()
    else:
        refer_pc_idx_inside_box = []

    refer_pc_idx_outside_box = set(range(refer_pc.shape[0])) - set(refer_pc_idx_inside_box)
    refer_pc_idx_outside_box = list(refer_pc_idx_outside_box)

    # Compute pixel (cell) instance
    most_freq_info = []
    for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):
        pixel_elements_instances = points_instance_id[v[0]:v[0] + v[1]]
        instance_freq = np.bincount(pixel_elements_instances, minlength=num_instances)
        # print(instance_freq)
        instance_freq = instance_freq / float(v[1])
        most_freq_instance_id, most_freq = np.argmax(instance_freq), np.max(instance_freq)
        most_freq_info.append([most_freq_instance_id, most_freq])

        pixel_instance_id[h] = most_freq_instance_id
        # break
    # print(pixel_instance_id.shape)  # (1501,)

    pixel_instance_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.uint8)
    pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]] = pixel_instance_id[:]
    # print(most_freq_info)
    # ----------------------------------------------------

    # ----------------------------------------------------
    # Compute the displacement vectors (flow)
    all_disp_field_gt_list = list()
    all_valid_pixel_maps_list = list()

    # print(past_frame)
    # print(past_frame_num)
    frame_considered = range(past_frame_num, past_frame_num + future_frame_num)

    for i in frame_considered:
        curr_disp_vectors = np.zeros_like(refer_pc, dtype=np.float32)
        curr_disp_vectors.fill(np.nan)
        curr_disp_vectors[refer_pc_idx_outside_box,] = 0.0

        # First, for each instance, compute the corresponding points displacement.
        for j in range(num_instances):
            pc_in_bbox_idx = refer_pc_idx_per_bbox[j]
            instance_boxes = all_boxes_list[j]
            tmp_box = instance_boxes[i]
            ref_box = ref_boxes_list[j]

            if tmp_box is None:  # It is possible that in this sweep there is no annotation
                continue

            disp_vectors = calc_displace_vector(refer_pc[pc_in_bbox_idx][:, :3], ref_box, tmp_box)

            curr_disp_vectors[pc_in_bbox_idx] = disp_vectors[:]

        # Second, compute the mean displacement vector and category for each non-empty pixel
        disp_field = np.zeros([unique_indices.shape[0], 2], dtype=np.float32)  # we only consider the 2D field
        # We only compute loss for valid pixels where there are corresponding box annotations between two frames
        valid_pixels = np.zeros(unique_indices.shape[0], dtype=np.bool)

        for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):
            pixel_elements_instances = points_instance_id[v[0]:v[0] + v[1]]
            most_freq_instance_id, most_freq = most_freq_info[h]

            if most_freq >= proportion_thresh:
                most_freq_instance_idx = np.where(pixel_elements_instances == most_freq_instance_id)[0]
                most_freq_instance_disp_vectors = curr_disp_vectors[v[0]:v[0] + v[1], :3]
                most_freq_instance_disp_vectors = most_freq_instance_disp_vectors[most_freq_instance_idx]

                if np.isnan(most_freq_instance_disp_vectors).any():  # contains invalid disp vectors
                    valid_pixels[h] = 0.0
                else:
                    mean_disp_vector = np.mean(most_freq_instance_disp_vectors, axis=0)
                    disp_field[h] = mean_disp_vector[0:2]  # ignore the z direction

                    valid_pixels[h] = 1.0
        
        # Finally, assemble to a 2D image
        disp_field_sparse = np.zeros((num_divisions[0], num_divisions[1], 2), dtype=np.float32)
        disp_field_sparse[pixel_indices[:, 0], pixel_indices[:, 1]] = disp_field[:]

        valid_pixel_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
        valid_pixel_map[pixel_indices[:, 0], pixel_indices[:, 1]] = valid_pixels[:]

        all_disp_field_gt_list.append(disp_field_sparse)
        all_valid_pixel_maps_list.append(valid_pixel_map)

    all_disp_field_gt_list = np.stack(all_disp_field_gt_list, axis=0)
    all_valid_pixel_maps_list = np.stack(all_valid_pixel_maps_list, axis=0)
    
    return all_disp_field_gt_list, all_valid_pixel_maps_list, pixel_indices, pixel_instance_map


def create_seq_data_by_sample(nusc_path,
                              nusc_version,
                              target_dir,
                              split_name='train',
                              past_frame=3,
                              future_frame=2,
                              num_keyframe_skipped=0,
                              use_cam=True
                              ):
    """
    consider sample_data from key sample, i.e. 2Hz
    """
    nusc = NuScenes(version=nusc_version, dataroot=nusc_path, verbose=True)
    nusc_root = nusc.dataroot

    area_extents = np.array([[-32., 32.], [-32., 32.], [-3.0, 2.0]])
    voxel_size = [0.25, 0.25, 5.0]
    lidar_range = np.array([-32.0, -32.0, -3.0, 32.0, 32.0, 2.0])
    
    # === motionnet data split ===
    split_list = np.load('data_prepare/split_nusc_motionnet.npy', allow_pickle=True).item()  # 500/100/250
    scenes = split_list.get(split_name)
    print("Split: {}, which contains {} scenes.".format(split_name, len(scenes)))
    res_scenes = list()
    for s in scenes:
        s_id = s.split('_')[1]
        res_scenes.append(int(s_id))
    # ============================

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    save_folder = os.path.join(target_dir, split_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    count = 0

    all_samples_file = []

    for scene_i in res_scenes:
        scene = nusc.scene[scene_i]
        first_sample_token = scene['first_sample_token']
        curr_sample = nusc.get('sample', first_sample_token)
        curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
        
        # print("Processing scene {} ...".format(scene['name']))
        scene_count = 0
        # print(curr_sample)
        while curr_sample['next'] != '':

            # === get pc seq ===
            all_sample_data_tokens = []

            # past frame
            next_sample = curr_sample
            # curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])  # no need
            all_sample_data_tokens.append(curr_sample_data['token'])
            flag = False
            for _ in range(past_frame-1):
                if next_sample['prev'] != '':
                    next_sample = nusc.get('sample', next_sample['prev'])
                    next_sample_data = nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])
                    all_sample_data_tokens.append(next_sample_data['token'])
                else:
                    flag=True
                    break
            if flag:
                curr_sample = nusc.get('sample', curr_sample['next'])
                curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
                continue

            all_sample_data_tokens = all_sample_data_tokens[::-1]

            # future frame
            next_sample = curr_sample
            for _ in range(future_frame):
                if next_sample['next'] != '':
                    next_sample = nusc.get('sample', next_sample['next'])
                    next_sample_data = nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])
                    all_sample_data_tokens.append(next_sample_data['token'])
                else:  # no enough sample in this scene, break to next scene
                    flag=True
                    break
            if flag:
                break

            # To align with sample numbers in motionnet, skip the first sample in each scene
            if scene_count == 0:
                for _ in range(num_keyframe_skipped + 1):
                    curr_sample = nusc.get('sample', curr_sample['next'])
                    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
               
                scene_count += 1
                continue

            # skip the last sample in each scene
            last_sd_token = all_sample_data_tokens[-1]
            next_last_sample_token = nusc.get('sample', nusc.get('sample_data', last_sd_token)['sample_token'])['next']
            if next_last_sample_token == '':
                break

            # ==================

            # === check whether enough camera sample_data ===
            last_sample_data_token = all_sample_data_tokens[-1]
            last_sample = nusc.get('sample', nusc.get('sample_data', last_sample_data_token)['sample_token'])

            cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            for cam_i, cam_name in enumerate(cam_names):
                last_cam_sd_token = last_sample['data'][cam_name]
                
                # check this cam sd
                last_cam_sd = nusc.get('sample_data', last_cam_sd_token)
                next_cam_sd_token = last_cam_sd['next']

                if next_cam_sd_token == '':
                    break
                next_cam_sd = nusc.get('sample_data', next_cam_sd_token)
                if next_cam_sd['next'] == '':
                    break
            # ===============================================

            count += 1

            target_file = os.path.join(save_folder, curr_sample_data['token']+'.npy')
            if not os.path.exists(target_file):
                # === retrieve pc seq files & get pc seq ===
                ref_sd_token = all_sample_data_tokens[past_frame-1]
                # get_global_pose(): transform pc to lidar pose; 
                ref_from_global = get_global_pose(nusc, ref_sd_token, inverse=True)

                past_files_list = []
                past_matrix_list = []
                future_files_list = []
                future_matrix_list = []

                for idx in range(past_frame):
                    curr_sd_token = all_sample_data_tokens[idx]
                    curr_sd = nusc.get("sample_data", curr_sd_token)
                    lidar_path = os.path.join(nusc_root, curr_sd['filename'])
                    past_files_list.append(lidar_path)
                    global_from_curr = get_global_pose(nusc, curr_sd_token, inverse=False)
                    ref_from_curr = ref_from_global.dot(global_from_curr)
                    past_matrix_list.append(ref_from_curr)

                for idx in range(future_frame):
                    curr_sd_token = all_sample_data_tokens[idx+past_frame]
                    curr_sd = nusc.get("sample_data", curr_sd_token)
                    lidar_path = os.path.join(nusc_root, curr_sd['filename'])
                    future_files_list.append(lidar_path)
                    global_from_curr = get_global_pose(nusc, curr_sd_token, inverse=False)
                    ref_from_curr = ref_from_global.dot(global_from_curr)
                    future_matrix_list.append(ref_from_curr)

                # === get point cloud data ===
                past_pc_list = []
                for i, pc_file in enumerate(past_files_list):
                    pcd = get_nusc_pc(pc_file, transform_mat=past_matrix_list[i])
                    pcd = mask_points_by_range(pcd, limit_range=lidar_range)
                    past_pc_list.append(pcd)
                future_pc_list = []
                for i, pc_file in enumerate(future_files_list):
                    break
                    pcd = get_nusc_pc(pc_file, transform_mat=future_matrix_list[i])
                    pcd = mask_points_by_range(pcd, limit_range=lidar_range)
                    future_pc_list.append(pcd)
                
                # === get instance boxes === 
                ego_instance_tokens = []
                ego_sd_token = all_sample_data_tokens[past_frame-1]
                ego_sd = nusc.get('sample_data', ego_sd_token)

                # Get reference pose and timestamp
                ref_pose_rec = nusc.get('ego_pose', ego_sd['ego_pose_token'])
                ref_cs_rec = nusc.get('calibrated_sensor', ego_sd['calibrated_sensor_token'])
                ref_time = 1e-6 * ego_sd['timestamp']

                ego_sample = nusc.get('sample', ego_sd['sample_token'])
                ego_annos = ego_sample['anns']
                for ann_token in ego_annos:
                    ann_rec = nusc.get('sample_annotation', ann_token)
                    instance_token = ann_rec['instance_token']
                    ego_instance_tokens.append(instance_token)
                
                all_instance_infos = dict()
                for instance_token in ego_instance_tokens:
                    box_list = list()
                    all_times = list()
                    attr_list = list()  # attribute list
                    cat_list = list()  # category list
                    boxes = list()
                    for idx in range(len(all_sample_data_tokens)):
                        curr_sd_token = all_sample_data_tokens[idx]
                        curr_sd = nusc.get("sample_data", curr_sd_token)
                        box, attr, cat = get_nusc_instance_box(nusc, curr_sd, instance_token)
                        boxes.append(box)  # It is possible the returned box is None
                        attr_list.append(attr)
                        cat_list.append(cat)
                    for box in boxes:
                        if box is not None:
                            # Move box to ego vehicle coord system
                            box.translate(-np.array(ref_pose_rec['translation']))
                            box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)

                            # Move box to sensor coord system
                            box.translate(-np.array(ref_cs_rec['translation']))
                            box.rotate(Quaternion(ref_cs_rec['rotation']).inverse)
                        box_list.append(box)
                    all_instance_infos[instance_token] = {'boxes': box_list,
                                                        'ref_box': box_list[past_frame-1],
                                                        'attr_list': attr_list,
                                                        'cat_list': cat_list}
                    
                # print(len(all_instance_infos))  # 94
                
                # === generate dense flow gt ===
                sample_flow_gt = None
                # - all_disp_field_gt: the ground-truth displacement vectors for each grid cell
                # - all_valid_pixel_maps: the masking map for valid pixels, used for loss computation
                # - pixel_indices: the indices of non-empty grid cells, used to generate sparse BEV maps
                # - pixel_instance_map: the map specifying the instance id for each grid cell, used for loss computation
                all_disp_field_gt, all_valid_pixel_maps, pixel_indices, pixel_instance_map \
                        = gen_2d_grid_gt(past_pcs=past_pc_list,
                        grid_size=voxel_size[0:2],
                        extents=area_extents,
                        instance_infos=all_instance_infos,
                        future_frame_num=2)

                # === convert dense bev map to sparse data ===
                all_valid_pixel_maps = all_valid_pixel_maps.astype(bool)

                sparse_disp_field_gt = all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :]
                sparse_valid_pixel_maps = all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]]
                sparse_pixel_instance_maps = pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]]

                sample_flow_gt = {'disp_field_gt': sparse_disp_field_gt,
                                'valid_pixel_maps': sparse_valid_pixel_maps,
                                'pixel_indices': pixel_indices,
                                'pixel_instance_maps': sparse_pixel_instance_maps}

                # === get related camera info ===
                cam_info = {}
                # nuscenes camera names
                cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                if use_cam:
                    # print(len(all_sample_data_tokens))  # 5, sample_data token of LIDAR_TOP
                    for idx in range(len(all_sample_data_tokens)):
                        frame_info = {}
                        curr_sd_token = all_sample_data_tokens[idx]
                        curr_sd = nusc.get("sample_data", curr_sd_token)  # lidar_top sample_data
                        cs_record = nusc.get('calibrated_sensor', curr_sd['calibrated_sensor_token'])  # lidar pose
                        lidar_pose = nusc.get('ego_pose', curr_sd['ego_pose_token'])  # lidar ego pose
                        l2e_r = cs_record['rotation']
                        l2e_t = cs_record['translation']
                        e2g_r = lidar_pose['rotation']
                        e2g_t = lidar_pose['translation']
                        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                        frame_info['lidar2ego_translation'] = l2e_t
                        frame_info['lidar2ego_rotation'] = l2e_r
                        frame_info['ego2global_translation'] = e2g_t
                        frame_info['ego2global_rotation'] = e2g_r

                        curr_cam_sample = nusc.get('sample', curr_sd['sample_token'])

                        for cam in cam_names:
                            single_cam_info = {}
                            cam_token = curr_cam_sample['data'][cam]
                            camera_sample = nusc.get('sample_data', cam_token)
                            cs_record = nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])  # camera pose
                            pose_record = nusc.get('ego_pose', camera_sample['ego_pose_token'])  # camera lidar pose

                            l2e_r_s = cs_record['rotation']
                            l2e_t_s = cs_record['translation']
                            e2g_r_s = pose_record['rotation']
                            e2g_t_s = pose_record['translation']

                            single_cam_info['sensor2ego_translation'] = l2e_t_s
                            single_cam_info['sensor2ego_rotation'] = l2e_r_s
                            single_cam_info['ego2global_translation'] = e2g_t_s
                            single_cam_info['ego2global_rotation'] = e2g_r_s

                            # obtain the RT from sensor to Top LiDAR
                            # sweep(cam)->ego->global->ego'->lidar
                            l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                            e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
                            R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                            T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                            T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                                        ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                            single_cam_info['sensor2lidar_rotation'] = R.T  # points @ R.T + T
                            single_cam_info['sensor2lidar_translation'] = T

                            # image path
                            # image_path = os.path.join(nusc_root, camera_sample['filename'])
                            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)

                            single_cam_info['cam_intrinsic'] = cam_intrinsic
                            single_cam_info['cam_path'] = cam_path

                            frame_info[cam] = single_cam_info

                        cam_info[idx] = frame_info
                
                # === save data info ===
                data = {"past_frame": past_frame,
                        "future_frame": future_frame,
                        "all_sample_data_tokens": all_sample_data_tokens,
                        "past_pc_files": past_files_list,
                        "future_pc_files": future_files_list,
                        "past_transform": past_matrix_list,
                        "future_transform": future_matrix_list,
                        "flow_gt": sample_flow_gt,
                        "future_flow": None,
                        "future_flow_mask": None,
                        "bboxes": None,
                        "cam_data": cam_info,
                        }

                # save to target_dir
                np.save(os.path.join(save_folder, curr_sample_data['token']+'.npy'), data)
                print("  >> Finish sample {}".format(count))
                        
            # === move to next sample & sample_data ===
            flag = False
            for _ in range(num_keyframe_skipped + 1):
                if curr_sample['next'] != '':
                    curr_sample = nusc.get('sample', curr_sample['next'])
                else:
                    flag = True
                    break
            if flag:  # No more keyframes
                break
            else:
                curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])

    print(count)
    print(len(all_samples_file))

    # train 17587
    # val 1729
    # test 4327


if __name__=='__main__':
    nusc_path = '/path/to/nuScenes/data/'
    nusc_version = 'v1.0-trainval'
    save_dir = '/path/to/save/processed/dataset'
    split_name = 'train'
    skip_frame = 0
    past_frame = 3
    future_frame = 2
    create_seq_data_by_sample(nusc_path=nusc_path,
                              nusc_version=nusc_version,
                              target_dir=save_dir,
                              split_name=split_name,
                              past_frame=3,
                              future_frame=2,
                              num_keyframe_skipped=skip_frame)
