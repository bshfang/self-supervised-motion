import torch
from torch.nn import Module

from losses.chamfer import chamfer_distance

from losses.common_losses import SmoothnessLossSeq, TemporalConsistencyLoss, InstanceConsistencyLoss, RegLoss

#from vis.data_check import visualize_pc

class UnSupervisedPredLoss(Module):
    def __init__(self, 
                 range,
                 voxel_size, 
                 w_chamfer,
                 smoothness_loss_param, 
                 temporal_consistency_loss_param, 
                 **kwargs):
        super(UnSupervisedPredLoss, self).__init__()

        self.lidar_range = range
        self.voxel_size = voxel_size

        # self.reconstruct_loss = ChamferLoss2(**chamfer_loss_params)
        self.chamfer_norm = kwargs['chamfer_norm']
        self.chamfer_weight = w_chamfer
        self.chamfer_loss = chamfer_distance

        # smoothness loss (for ablation)
        self.smoothness_flag = smoothness_loss_param['flag']
        if self.smoothness_flag:
            self.smoothness_loss = SmoothnessLossSeq(**smoothness_loss_param)
            self.smoothness_weight = kwargs['w_smoothness']

        self.use_generated_mask = kwargs['weighted_loss']['use_static_mask']

        # reg loss weight
        self.reg_loss_weight = kwargs['reg_loss_weight']

        # backward flow
        self.use_backward_loss = kwargs['use_backward']

        self.single_predict = kwargs['single_predict']

        # chamfer loss remove static
        self.chamfer_remove_static = kwargs['chamfer_remove_static']

        # temporal_consistency_loss
        self.temporal_consistency_flag = temporal_consistency_loss_param['flag']
        if self.temporal_consistency_flag:
            self.temporal_consistency_loss = TemporalConsistencyLoss(temporal_consistency_loss_param['loss_norm'], self.use_backward_loss)
            self.temporal_consistency_weight = kwargs['w_temporal_consistency']
        
        # instance_consistency_loss
        self.instance_consistency_flag = kwargs['instance_consistency_loss_param']['flag']
        if self.instance_consistency_flag:
            self.instance_consistency_loss = InstanceConsistencyLoss(**kwargs['instance_consistency_loss_param'])
            self.instance_consistency_weight = kwargs['w_instance_consistency']

        if self.use_generated_mask:
            self.point_reg_loss = RegLoss()
    
    def forward(self, data_batch, output_dict):
        batch_size = len(data_batch['past_pc'])

        pillar_flow_output = output_dict['pillar_flow_output']

        nx = int((self.lidar_range[3] - self.lidar_range[0]) / self.voxel_size[0])
        ny = int((self.lidar_range[4] - self.lidar_range[1]) / self.voxel_size[1])

        chamfer_loss = 0
        smoothness_loss = 0
        temporal_consistency_loss = 0
        instance_consistency_loss = 0

        output_loss = {}
        for batch_i in range(batch_size):

            current_pc = data_batch['past_pc'][batch_i][-1][:, :3]
            coord_x = torch.floor((current_pc[:, 0:1] - self.lidar_range[0]) / self.voxel_size[0])
            coord_y = torch.floor((current_pc[:, 1:2] - self.lidar_range[1]) / self.voxel_size[1])
            coord = torch.cat([coord_x, coord_y], dim=1)
            pidx = coord[:, 1] * nx + coord[:, 0]
            pidx = pidx.long()

            future_pc = data_batch['future_pc'][batch_i]
            pillar_flow_seq = pillar_flow_output[batch_i]

            if self.use_generated_mask:
                generated_mask = data_batch['static_mask'][batch_i][1]  # static_mask of current pc
                generated_motion_weight = generated_mask.float()
            
            if self.single_predict:
                future_seq_len = 1
            else:
                future_seq_len = len(future_pc)
            if self.smoothness_flag or self.temporal_consistency_flag or self.instance_consistency_flag:
                if self.use_backward_loss:
                    all_future_flow = torch.zeros((future_seq_len+1, current_pc.shape[0], 3), device=current_pc.device)
                else:
                    all_future_flow = torch.zeros((future_seq_len, current_pc.shape[0], 3), device=current_pc.device)
            
            # backward flow loss
            if self.use_backward_loss:
                pillar_flow = pillar_flow_seq[:, :, 2*2:2*2+2]  # the last two channel of the output is backward flow
                pred_motion = pillar_flow.view(-1,2)
                selected_motion = pred_motion[pidx, :]

                pred_points = torch.zeros_like(current_pc, device=current_pc.device)
                pred_points[:, :2] = current_pc[:, :2] + selected_motion

                if self.smoothness_flag or self.temporal_consistency_flag or self.instance_consistency_flag:
                    all_future_flow[future_seq_len, :, :2] = selected_motion

                past_pc = data_batch['past_pc'][batch_i]
                target_points = past_pc[-2][:, :3]  # the past one frame

                if self.use_generated_mask:
                    if self.chamfer_remove_static:
                        new_pred_points = pred_points[generated_mask].contiguous()
                        point_mask = data_batch['static_mask'][batch_i][0]
                        new_target_points = target_points[point_mask].contiguous()

                        chamfer_loss_i = self.chamfer_loss(new_pred_points.unsqueeze(0), 
                                                        new_target_points.unsqueeze(0), 
                                                        norm=self.chamfer_norm)
                    else:
                        chamfer_loss_i = self.chamfer_loss(pred_points.unsqueeze(0), 
                                                        target_points.unsqueeze(0), 
                                                        weights=generated_motion_weight.unsqueeze(0), 
                                                        norm=self.chamfer_norm)
                    # restrict flow of static points/grids to zero
                    reg_loss = self.reg_loss_weight * self.point_reg_loss(selected_motion, (1.0-generated_motion_weight))
                    chamfer_loss += (chamfer_loss_i + reg_loss) / 2
                else:
                    chamfer_loss += self.chamfer_loss(pred_points.unsqueeze(0), 
                                                      target_points.unsqueeze(0),
                                                      norm=self.chamfer_norm)

            # predicted forward flow loss
            for seq_i in range(future_seq_len):

                pillar_flow = pillar_flow_seq[:, :, seq_i*2:seq_i*2+2]
                pred_motion = pillar_flow.view(-1,2)
                selected_motion = pred_motion[pidx, :]

                pred_points = torch.zeros_like(current_pc, device=current_pc.device)
                pred_points[:, :2] = current_pc[:, :2] + selected_motion
                if self.smoothness_flag or self.temporal_consistency_flag or self.instance_consistency_flag:
                    all_future_flow[seq_i, :, :2] = selected_motion
                target_points = future_pc[seq_i][:, :3]

                if self.use_generated_mask:
                    if self.chamfer_remove_static:
                        new_pred_points = pred_points[generated_mask].contiguous()
                        point_mask = data_batch['static_mask'][batch_i][seq_i+2]
                        new_target_points = target_points[point_mask].contiguous()
                        
                        chamfer_loss_i = self.chamfer_loss(new_pred_points.unsqueeze(0), 
                                                        new_target_points.unsqueeze(0), 
                                                        norm=self.chamfer_norm)
                    else:
                        chamfer_loss_i = self.chamfer_loss(pred_points.unsqueeze(0), 
                                                        target_points.unsqueeze(0), 
                                                        weights=generated_motion_weight.unsqueeze(0), 
                                                        norm=self.chamfer_norm)
                    reg_loss = self.reg_loss_weight * self.point_reg_loss(selected_motion, (1.0-generated_motion_weight))
                    chamfer_loss += (chamfer_loss_i + reg_loss) / 2
                else:
                    chamfer_loss += self.chamfer_loss(pred_points.unsqueeze(0), 
                                                      target_points.unsqueeze(0),
                                                      norm=self.chamfer_norm)
            
            if self.smoothness_flag:
                current_pc = current_pc.unsqueeze(0).contiguous()
                smoothness_loss += self.smoothness_loss(current_pc, all_future_flow)
                smoothness_loss += self.smoothness_loss(current_pc, all_future_flow)
                
            if self.temporal_consistency_flag:
                temporal_consistency_loss += self.temporal_consistency_loss(all_future_flow)
                temporal_consistency_loss += self.temporal_consistency_loss(all_future_flow)
            
            if self.instance_consistency_flag:
                instance_mask = data_batch['non_static_rigid_piece'][batch_i]
                instance_consistency_loss += self.instance_consistency_loss(all_future_flow, instance_mask)

        smoothness_loss /= batch_size
        temporal_consistency_loss /= batch_size
        instance_consistency_loss /= batch_size

        if self.use_backward_loss:
            chamfer_loss /= (future_seq_len + 1)
        else:
            chamfer_loss /= future_seq_len

        chamfer_loss /= batch_size

        if self.smoothness_flag:
            output_loss['smoothness_loss'] = self.smoothness_weight * smoothness_loss
        if self.temporal_consistency_flag:
            output_loss['temporal_consistency_loss'] = self.temporal_consistency_weight * temporal_consistency_loss

        if self.instance_consistency_flag:
            output_loss['instance_consistency_loss'] = self.instance_consistency_weight * instance_consistency_loss

        output_loss['chamfer_loss'] = self.chamfer_weight * chamfer_loss

        return output_loss
