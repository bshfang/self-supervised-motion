import sys, os
import os.path as osp
import numpy as np
import yaml
from collections import OrderedDict

import torch
import torch.utils.data as data

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
import nuscenes.utils.splits as nusc_splits

try:
    from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
except:
    from spconv.utils import VoxelGenerator
from datasets.data_utils import read_pcd, mask_points_by_range, get_nusc_pc


class MyLidarPointCloud(LidarPointCloud):
    def get_ego_mask(self):
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5),
        )
        return ego_mask


class NuScenes_sequence(data.Dataset):
    def __init__(self,
                 dataset_cfg,
                 split,
                 data_root,
                 static_mask_root=None,
                 rigid_piece_root=None,):
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.load_gt = True
        self.root = data_root
        self.mask_root = static_mask_root
        self.rigid_piece_root = rigid_piece_root

        if 'restrict_loss_range' in dataset_cfg:
            self.restrict_loss_range = dataset_cfg["restrict_loss_range"]
        else:
            self.restrict_loss_range = False

        self.samples=[]
        if split == 'train':
            self.train = True
        else:
            self.train = False

        tmp_path = split
        sample_path = osp.join(self.root, tmp_path)
        for d in os.listdir(sample_path):
            self.samples.append(osp.join(tmp_path,d))

        self.voxel_generator = VoxelGenerator(
            voxel_size=dataset_cfg['voxel_size'],
            point_cloud_range=dataset_cfg['point_cloud_range'],
            max_num_points=dataset_cfg['max_points_in_voxel'],
            max_voxels=dataset_cfg['max_voxel_num']
        )

    def __len__(self):
        return len(self.samples)
    
    def collate_batch_train(self, batch):

        # merge batch*seq_n pc_voxelization features
        merge_pc_voxelization = OrderedDict()
        for i in range(len(batch)):
            for j in range(len(batch[i]['pc_voxelization'])):
                for feature_name, feature in batch[i]['pc_voxelization'][j].items():
                    if feature_name not in merge_pc_voxelization:
                        merge_pc_voxelization[feature_name] = []
                    if isinstance(feature, list):
                        merge_pc_voxelization[feature_name] += feature
                    else:
                        merge_pc_voxelization[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]

        voxel_features = \
            torch.from_numpy(np.concatenate(merge_pc_voxelization['voxel_features']))
        voxel_num_points = \
            torch.from_numpy(np.concatenate(merge_pc_voxelization['voxel_num_points']))
        coords = merge_pc_voxelization['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        past_pcs = OrderedDict()
        future_pcs = OrderedDict()

        static_masks = OrderedDict()
        non_static_rigid_pieces = OrderedDict()

        gt_flows = []
        gt_valid_maps = []
        gt_instance_maps = []
        
        for i in range(len(batch)):
            batch_i = batch[i]
            past_pcs[i] = batch_i['past_pc']
            future_pcs[i] = batch_i['future_pc']
            if self.mask_root and self.train:
                static_masks[i] = batch_i['static_mask']

            if self.rigid_piece_root and self.train:
                non_static_rigid_pieces[i] = batch_i['non_static_rigid_piece']

            if self.load_gt or not self.train:
                if 'all_disp_field_gt' in batch_i:
                    gt_flows.append(batch_i['all_disp_field_gt'])
                    gt_valid_maps.append(batch_i['all_valid_pixel_maps'])
                    gt_instance_maps.append(batch_i['all_pixel_instance_maps'])

        if self.load_gt or not self.train:
            gt_flows = torch.from_numpy(np.stack(gt_flows, axis=0))
            gt_valid_maps = torch.from_numpy(np.stack(gt_valid_maps, axis=0))
            gt_instance_maps = torch.from_numpy(np.stack(gt_instance_maps, axis=0))

        output_dict = {
            'past_pc': past_pcs,
            'future_pc': future_pcs,
            'static_mask': static_masks,
            'non_static_rigid_piece': non_static_rigid_pieces,
            'voxel_features': voxel_features,
            'voxel_num_points': voxel_num_points,
            'voxel_coords': voxel_coords,
            'gt_flows': gt_flows,
            'gt_valid_maps': gt_valid_maps,
            'gt_instance_maps': gt_instance_maps,
        }

        return output_dict
    
    def __getitem__(self, index):
        sample_file_path = self.samples[index]
        filename = osp.join(self.root, sample_file_path)
        with open(filename, 'rb') as fp:
            data = np.load(fp, allow_pickle=True)
            data = data.item()

        sample_data = OrderedDict()

        # === pcs ===
        past_pc = []
        future_pc = []
        past_pc_voxelization = []

        static_masks = []  # -1,0,1,2

        all_sample_data_tokens = data['all_sample_data_tokens']

        for i, pc_file in enumerate(data['past_pc_files']):
            data_dict = OrderedDict()
            if i == 2 and self.restrict_loss_range:
                pcd = get_nusc_pc(pc_file, transform_mat=data['past_transform'][i])
                pcd,mask = mask_points_by_range(pcd, limit_range=self.dataset_cfg['point_cloud_range'])

                voxel_output = self.voxel_generator.generate(pcd)
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], \
                    voxel_output['num_points_per_voxel']
                data_dict['voxel_features'] = voxels
                data_dict['voxel_coords'] = coordinates
                data_dict['voxel_num_points'] = num_points
                past_pc_voxelization.append(data_dict)

                if self.mask_root and self.train:
                    sd_token = all_sample_data_tokens[i]

                    static_mask_file = os.path.join(self.mask_root, sd_token+'.npz')
                    static_mask_data = np.load(static_mask_file)
                    static_mask = static_mask_data['static_mask']

                    static_mask = static_mask[mask]
                
                pcd_2,mask_2 = mask_points_by_range(pcd, limit_range=[-30.0, -30.0, -3.0, 30.0, 30.0, 2.0])
                past_pc.append(torch.from_numpy(pcd_2))
                if self.mask_root and self.train:
                    static_masks.append(torch.from_numpy(static_mask[mask_2]))

                # rigid piece
                if i == 2:
                    if self.rigid_piece_root and self.train:
                        sd_token = all_sample_data_tokens[2]

                        rigid_piece_file = os.path.join(self.rigid_piece_root, sd_token+'.npz')
                        rigid_piece_data = np.load(rigid_piece_file)
                        non_static_aggregated_slic_label = rigid_piece_data['non_static_aggregated_slic_label']

                        sample_data['non_static_rigid_piece'] = torch.from_numpy(non_static_aggregated_slic_label[mask_2])

            else:  # i == 0 or 1 or (i==2 and restrict_loss_range)
                pcd = get_nusc_pc(pc_file, transform_mat=data['past_transform'][i])
                pcd,mask = mask_points_by_range(pcd, limit_range=self.dataset_cfg['point_cloud_range'])
                past_pc.append(torch.from_numpy(pcd))
                if i != 0:
                    if self.mask_root and self.train:
                        sd_token = all_sample_data_tokens[i]

                        # ver. 0722
                        static_mask_file = os.path.join(self.mask_root, sd_token+'.npz')
                        static_mask_data = np.load(static_mask_file)
                        static_mask = static_mask_data['static_mask']

                        static_masks.append(torch.from_numpy(static_mask[mask]))

                if i == 2:
                    if self.rigid_piece_root and self.train:
                        sd_token = all_sample_data_tokens[2]

                        rigid_piece_file = os.path.join(self.rigid_piece_root, sd_token+'.npz')
                        rigid_piece_data = np.load(rigid_piece_file)
                        non_static_aggregated_slic_label = rigid_piece_data['non_static_aggregated_slic_label']

                        sample_data['non_static_rigid_piece'] = torch.from_numpy(non_static_aggregated_slic_label)

                # print(torch.from_numpy(pcd).shape)
                voxel_output = self.voxel_generator.generate(pcd)
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], \
                    voxel_output['num_points_per_voxel']
                data_dict['voxel_features'] = voxels
                data_dict['voxel_coords'] = coordinates
                data_dict['voxel_num_points'] = num_points
                past_pc_voxelization.append(data_dict)

        for i, pc_file in enumerate(data['future_pc_files']):
            pcd = get_nusc_pc(pc_file, transform_mat=data['future_transform'][i])
            pcd,mask = mask_points_by_range(pcd, limit_range=self.dataset_cfg['point_cloud_range'])
            future_pc.append(torch.from_numpy(pcd))
            if self.mask_root and self.train:
                sd_token = all_sample_data_tokens[i+len(data['past_pc_files'])]

                # ver. 0722
                static_mask_file = os.path.join(self.mask_root, sd_token+'.npz')
                static_mask_data = np.load(static_mask_file)
                static_mask = static_mask_data['static_mask']

                static_masks.append(torch.from_numpy(static_mask[mask]))

        sample_data['past_pc'] = past_pc
        sample_data['future_pc'] = future_pc
        sample_data['pc_voxelization'] = past_pc_voxelization

        # === static mask ===
        sample_data['static_mask'] = static_masks

        # === ground truths ===
        if self.load_gt or not self.train:
            gt_dict = data['flow_gt']
            # dims = gt_dict['dims']
            dims = (256, 256, 1)
            num_future_pcs = data['future_frame']
            pixel_indices = gt_dict['pixel_indices']

            # flow gt: disp_field_gt
            sparse_disp_field_gt = gt_dict['disp_field_gt']
            all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
            all_disp_field_gt[:, pixel_indices[:, 1], pixel_indices[:, 0], :] = sparse_disp_field_gt[:]  # y,x

            # valid pixel maps: valid_pixel_maps
            sparse_valid_pixel_maps = gt_dict['valid_pixel_maps']
            all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
            all_valid_pixel_maps[:, pixel_indices[:, 1], pixel_indices[:, 0]] = sparse_valid_pixel_maps[:]  # y,x

            # instance maps: pixel_instance_maps
            sparse_pixel_instance_maps = gt_dict['pixel_instance_maps']
            all_pixel_instance_maps = np.zeros((dims[0], dims[1]), dtype=np.uint8)
            all_pixel_instance_maps[pixel_indices[:, 1], pixel_indices[:, 0]] = sparse_pixel_instance_maps[:]  # y,x

            sample_data['all_disp_field_gt'] = all_disp_field_gt
            sample_data['all_valid_pixel_maps'] = all_valid_pixel_maps
            sample_data['all_pixel_instance_maps'] = all_pixel_instance_maps

        return sample_data