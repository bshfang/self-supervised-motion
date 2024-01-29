import torch
from torch.nn import Module, MSELoss, L1Loss
from lib.pointnet2 import pointnet2_utils as pointutils


class KnnLoss(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(KnnLoss, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        flow = pred_flow.permute(0, 2, 1).contiguous()
        dist, idx = pointutils.knn(self.k, pc_source, pc_source)
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.k).to(idx.device)
        idx[dist > self.radius] = tmp_idx[dist > self.radius]  # set idx that is > radius to itself
        nn_flow = pointutils.grouping_operation(flow, idx.detach())
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean( dim=-1)
        return loss.mean()


class BallQLoss(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(BallQLoss, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        flow = pred_flow.permute(0, 2, 1).contiguous()
        idx = pointutils.ball_query(self.radius, self.k, pc_source, pc_source)
        nn_flow = pointutils.grouping_operation(flow, idx.detach())  # retrieve flow of nn
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean( dim=-1)
        return loss.mean()


class KnnLossSeq(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(KnnLossSeq, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor, weights=None) -> torch.Tensor:
        """
        pc_source: source current pc for 1 batch, (1, n, 3)
        pred_flow: flow of seq_len, (seq_len, n, 3)
        weights: (n)
        """
        flow = pred_flow.permute(0, 2, 1).contiguous()
        dist, idx = pointutils.knn(self.k, pc_source, pc_source)  # distance: near -> far
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.k).to(idx.device)
        idx[dist > self.radius] = tmp_idx[dist > self.radius]  # set idx that is > radius to itself
        seq_len = flow.shape[0]
        idx = idx.repeat(seq_len, 1, 1)
        nn_flow = pointutils.grouping_operation(flow, idx.detach())
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean( dim=-1)
        
        if weights is not None:
            loss = loss * weights.unsqueeze(0)
            loss = loss.sum(1)
            if weights.sum() > 0:
                loss /= weights.sum()
            else:
                loss = loss
            return loss.mean()
        else:
            loss = loss.mean(dim=-1)
            return loss.mean()


class BallQLossSeq(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(BallQLossSeq, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor, weights=None) -> torch.Tensor:
        flow = pred_flow.permute(0, 2, 1).contiguous()
        idx = pointutils.ball_query(self.radius, self.k, pc_source, pc_source)
        seq_len = flow.shape[0]
        idx = idx.repeat(seq_len, 1, 1)
        nn_flow = pointutils.grouping_operation(flow, idx.detach())  # retrieve flow of nn
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean(dim=-1)

        if weights is not None:
            loss = loss * weights.unsqueeze(0)
            loss = loss.sum(1)
            if weights.sum() > 0:
                loss /= weights.sum()
            else:
                loss = loss
            return loss.mean()
        else:
            loss = loss.mean(dim=-1)
            return loss.mean()


class SmoothnessLoss(Module):
    def __init__(self, w_knn, w_ball_q, knn_loss_params, ball_q_loss_params, **kwargs):
        super(SmoothnessLoss, self).__init__()
        self.knn_loss = KnnLoss(**knn_loss_params)
        self.ball_q_loss = BallQLoss(**ball_q_loss_params)
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        loss = (self.w_knn * self.knn_loss(pc_source, pred_flow)) + (self.w_ball_q * self.ball_q_loss(pc_source, pred_flow))
        return loss
    

class SmoothnessLossSeq(Module):
    def __init__(self, w_knn, w_ball_q, knn_loss_params, ball_q_loss_params, **kwargs):
        super(SmoothnessLossSeq, self).__init__()
        self.knn_loss = KnnLossSeq(**knn_loss_params)
        self.ball_q_loss = BallQLossSeq(**ball_q_loss_params)
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor, weights=None) -> torch.Tensor:
        """
        pc_source: source current pc for 1 batch, (1, n, 3)
        pred_flow: flow of seq_len, (seq_len, n, 3)
        weight: 
        """
        knn_loss = self.knn_loss(pc_source, pred_flow, weights=weights)
        ball_q_loss = self.ball_q_loss(pc_source, pred_flow, weights=weights)
        loss = self.w_knn * knn_loss + self.w_ball_q * ball_q_loss
        return loss


class TemporalConsistencyLoss(Module):
    def __init__(self, loss_norm, use_backward, **kwargs):
        super(TemporalConsistencyLoss, self).__init__()
        self.loss_norm = loss_norm
        self.use_backward = use_backward

    def forward(self, pred_flow: torch.Tensor, weights=None) -> torch.Tensor:
        """
        pred_flow: flow of seq_len, (seq_len, n, 3)
        weights: n
        """
        rescale_pred = torch.zeros_like(pred_flow, device=pred_flow.device)
        if self.use_backward:
            pred_len = pred_flow.shape[0] - 1
        else:
            pred_len = pred_flow.shape[0]
        for seq_i in range(pred_len):
            rescale_pred[seq_i, :, :] = pred_flow[seq_i, :, :] / (seq_i+1)
        if self.use_backward:
            rescale_pred[pred_len, :, :] = pred_flow[pred_len, :, :] * (-1.)
        mean_flow = rescale_pred.mean(dim=0)
        loss = (mean_flow.unsqueeze(0) - rescale_pred).norm(p=self.loss_norm, dim=2)
        
        if weights is not None:
            loss = loss * weights.unsqueeze(0)
            loss = loss.sum(1)
            if weights.sum() > 0:
                loss /= weights.sum()
            else:
                loss = loss
            return loss.mean()
        else:
            loss = loss.mean(dim=-1)
            return loss.mean()


class InstanceConsistencyLoss(Module):
    def __init__(self, loss_norm, **kwargs):
        super(InstanceConsistencyLoss, self).__init__()
        self.loss_norm = loss_norm

    def forward(self, pred_flow: torch.Tensor, instance_map: torch.Tensor) -> torch.Tensor:
        """
        pred_flow: flow of seq_len, (seq_len, n, 3)
        instance_map: (n)
        """
        seq_len = pred_flow.shape[0]
        instances_idx = torch.unique(instance_map)
        
        instance_num = instances_idx.shape[0] - 1

        all_loss = 0
        for idx in instances_idx:
            if idx == -1:  # idx == 0 background: 0 in gt, -1 in generated mask
                pass
            else:
                instance_mask = instance_map == idx
                instance_flow = pred_flow[:, instance_mask]

                instance_mean_flow = instance_flow.mean(dim=1)  # seq_len, 3
                loss = (instance_mean_flow.unsqueeze(1) - instance_flow).norm(p=self.loss_norm, dim=2)
                loss = loss.mean()
                all_loss += loss
        if instance_num:
            all_loss /= instance_num
            return all_loss
        else:
            return 0


class RegLoss(Module):
    """
    restrict all flow to zero
    """
    def __init__(self, loss_norm=1):
        super(RegLoss, self).__init__()
        self.loss_norm = loss_norm

    def forward(self, pred_flow, weights):
        """
        pred_flow: n, 2
        weights: n
        """
        loss = pred_flow.norm(p=self.loss_norm, dim=1) * weights
        loss = loss.sum(dim=0)
        loss /= weights.sum()

        return loss