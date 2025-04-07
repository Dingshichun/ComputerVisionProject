# 计算损失

import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, neg_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.neg_ratio = neg_ratio

    def forward(self, pred_loc, pred_conf, gt_loc, gt_label):
        pos_mask = gt_label > 0
        loc_loss = F.smooth_l1_loss(
            pred_loc[pos_mask], gt_loc[pos_mask], reduction="sum"
        )

        # 分类损失（含难例挖掘）
        conf_loss = F.cross_entropy(
            pred_conf.view(-1, self.num_classes), gt_label.view(-1), reduction="none"
        )
        pos_conf_loss = conf_loss[pos_mask.view(-1)]
        neg_conf_loss = conf_loss[~pos_mask.view(-1)]
        num_neg = min(self.neg_ratio * pos_conf_loss.size(0), neg_conf_loss.size(0))
        _, idx = neg_conf_loss.topk(num_neg)

        total_loss = (
            loc_loss + pos_conf_loss.sum() + neg_conf_loss[idx].sum()
        ) / pos_mask.sum()
        return total_loss
