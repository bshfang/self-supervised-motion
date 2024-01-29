import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointpillars.pillar_encoder import PillarVFE, PointPillarScatter
from models.temporal_model.stpn import STPN


class MotionHead(nn.Module):
    def __init__(self, in_channels, output_frame, channels=32):
        super(MotionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, 2*output_frame, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class FlowPred(nn.Module):
    def __init__(self, model_cfg, input_frame, output_frame):
        super(FlowPred, self).__init__()

        self.input_frame = input_frame
        self.output_frame = output_frame

        self.pillar_vfe = PillarVFE(model_cfg=model_cfg['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=model_cfg['voxel_size'],
                                    point_cloud_range=model_cfg['point_cloud_range'])
        self.scatter = PointPillarScatter(model_cfg['point_pillar_scatter'])

        # To simplify, use motionnet backbone
        self.temporal_modal = STPN(model_cfg['temporal_model']['height_feat_size'])

        self.motion_head = MotionHead(in_channels=32, output_frame=output_frame)

    def forward(self, batch):
        data_dict = {'voxel_features': batch['voxel_features'],
                      'voxel_coords': batch['voxel_coords'],
                      'voxel_num_points': batch['voxel_num_points']}
        data_dict = self.pillar_vfe(data_dict)
        data_dict = self.scatter(data_dict)
        x = data_dict['spatial_features']
        x = x.view(-1, self.input_frame, *x.shape[-3:])
        x = x.permute(0,2,1,3,4)

        x = self.temporal_modal(x)
        flow_output = self.motion_head(x)

        output_dict = {'pillar_flow_output': flow_output}
        return output_dict