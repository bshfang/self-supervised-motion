import torch
import torch.nn as nn
from torch.nn import Module, MSELoss, L1Loss, SmoothL1Loss

class SupervisedPredLoss(Module):
    def __init__(self,
                 use_weighted_loss,
                 **kwargs):
        super(SupervisedPredLoss, self).__init__()

        self.use_weighted_loss = use_weighted_loss

        if use_weighted_loss:
            self.criterion = SmoothL1Loss(reduction='none')
        else:
            self.criterion = SmoothL1Loss(reduction='sum')

    def forward(self, data_batch, output_dict):
        pillar_flow_output = output_dict['pillar_flow_output']
        b, h, w, _ = pillar_flow_output.shape
        pillar_flow_output = pillar_flow_output.view(b, h, w, -1, 2)
        pillar_flow_output = pillar_flow_output.permute(0, 3, 4, 1, 2).contiguous()
        pillar_flow_output = pillar_flow_output.view(-1, 2, h, w)

        gt_flow = data_batch['gt_flows']
        gt_flow = gt_flow.view(-1, gt_flow.size(2), gt_flow.size(3), gt_flow.size(4))
        gt_flow = gt_flow.permute(0, 3, 1, 2).contiguous()
        
        gt_valid_maps =data_batch['gt_valid_maps']

        gt_valid_maps = gt_valid_maps.view(-1, gt_valid_maps.size(2), gt_valid_maps.size(3))
        gt_valid_maps = torch.unsqueeze(gt_valid_maps, 1)
        valid_pixel_num = torch.nonzero(gt_valid_maps).size(0)

        output_loss = {}

        # Compute flow loss
        if self.use_weighted_loss:
            gt_instance_maps = data_batch['gt_instance_maps']  # b, 256, 256

            weight_map = torch.zeros_like(gt_instance_maps).to(gt_instance_maps.device)
            weight_vector = [0.005, 1.0]  # background: 0.005, foreground: 1.0

            background_mask = gt_instance_maps == 0  # all bg grids and empty grids
            foreground_mask = gt_instance_maps > 0
            weight_map[background_mask] = weight_vector[0]
            weight_map[foreground_mask] = weight_vector[1]

            weight_map = weight_map.unsqueeze(1).unsqueeze(1)
            map_shape = weight_map.size()

            loss_flow = self.criterion(gt_flow * gt_valid_maps, pillar_flow_output * gt_valid_maps)
            loss_flow = loss_flow.view(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])
            loss_flow = torch.sum(loss_flow * weight_map) / valid_pixel_num

        else:
            loss_flow = self.criterion(gt_flow * gt_valid_maps, pillar_flow_output * gt_valid_maps) / valid_pixel_num

        output_loss['flow_loss'] = loss_flow
        return output_loss