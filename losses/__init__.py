from losses.unsup_pred_losses import UnSupervisedPredLoss
from losses.sup_pred_losses import SupervisedPredLoss

__all__ = {
    'unsup_pred': UnSupervisedPredLoss,
    'sup_pred': SupervisedPredLoss,
}


def build_criterion(loss_config):
    criterion = __all__[loss_config['loss_type']](**loss_config)

    return criterion