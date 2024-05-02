#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..utils import logging
# from pytorch_metric_learning import losses
logger = logging.get_logger("visual_prompt")


class SigmoidLoss(nn.Module):
    def __init__(self):
        super(SigmoidLoss, self).__init__()

    def is_single(self):
        return True

    def is_local(self):
        return False

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        # (batch_size, num_classes)
        return target

    def loss(
        self, logits, targets, per_cls_weights,
        multihot_targets: Optional[bool] = False
    ):
        # targets: 1d-tensor of integer
        # Only support single label at this moment
        # if len(targets.shape) != 2:
        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # logger.info(f"loss shape: {loss.shape}")
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        ).unsqueeze(0)
        # logger.info(f"weight shape: {weight.shape}")
        loss = torch.mul(loss.to(torch.float32), weight.to(torch.float32))
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, targets, per_cls_weights=None, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  per_cls_weights, multihot_targets)
        return loss


class SoftmaxLoss(SigmoidLoss):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights=None, kwargs=None):
        #weight = torch.tensor(
            #per_cls_weights, device=logits.device
        #)
        loss = F.cross_entropy(logits, targets, reduction="none")

        return torch.sum(loss) / targets.shape[0]


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()
        print('using focal loss', self.gamma)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# class Arc_loss(nn.Module):
#     def __init__(self, class_emb=345, embed=768*2):
#         super(Arc_loss, self).__init__()
#         self.loss = losses.ArcFaceLoss(class_emb, embed, margin=28.6, scale=5)
#     def forward(self, embed, target):
#         return self.loss(embeddings=embed, labels=target)
#
#     def get_logits(self, embed):
#         return self.loss.get_logits(embeddings=embed)

LOSS = {
    "softmax": SoftmaxLoss,
    "MSE": torch.nn.MSELoss
}


def build_loss(loss_name):
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    else:
        return loss_fn()
