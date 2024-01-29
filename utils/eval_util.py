import numpy as np
from collections import OrderedDict


class AverageMeter:
    def __init__(self):
        self.loss_dict = OrderedDict()

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: [loss_val, 1]})
            else:
                self.loss_dict[loss_name][0] += loss_val
                self.loss_dict[loss_name][1] += 1

    def get_mean_loss(self):
        all_loss_val = 0.0
        all_loss_count = 0
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            all_loss_val += loss_val
            all_loss_count += loss_count
        return all_loss_val / (all_loss_count / len(self.loss_dict))

    def get_mean_loss_dict(self):
        loss_dict = {}
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            loss_dict[loss_name] = loss_val / loss_count
        return loss_dict

    def get_printable(self):
        text = ""
        all_loss_sum = 0.0
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            all_loss_sum += loss_val / loss_count
            text += "(%s:%.4f) " % (loss_name, loss_val / loss_count)
        text += " sum = %.4f" % all_loss_sum
        return text