#!/usr/bin/env python

"""

Purpose :

"""

import torch
import torch.nn as nn
import torch.utils.data

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, gamma=0.75, alpha=0.7):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        print("inside ft loss")
        print(y_pred.shape, y_true.shape)
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        pt_1 = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        # return pow((1 - pt_1), self.gamma)
        return pow(abs(1 - pt_1), self.gamma)


class SegmentationLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SegmentationLoss, self).__init__()
        self.smooth = smooth
        # Binary cross entropy
        self.BCE_logits_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true, batch_index=0):
        y_pred_dice = torch.sigmoid(y_pred)
        iflat = torch.flatten(y_pred_dice, 1)
        tflat = torch.flatten(y_true, 1)
        intersection = torch.sum(iflat * tflat, 1)
        union = torch.sum(iflat + tflat, 1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        # if torch.any(torch.isnan(dice_loss)):
        # print("nan value found in dice_loss")
        # torch.save(y_pred, f'y_pred_{batch_index}.pt')
        # torch.save(y_true, f"y_true_{batch_index}.pt")

        # Binary cross entropy
        BCE_loss = self.BCE_logits_loss(y_pred, y_true)

        seg_loss = dice_loss + (1 * BCE_loss)
        return seg_loss, dice_score


class DiceScore(nn.Module):
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        iflat = torch.flatten(y_pred, 1)
        tflat = torch.flatten(y_true, 1)
        intersection = torch.sum(iflat * tflat, 1)
        union = torch.sum(iflat + tflat, 1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        return dice_score


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        # MSE Loss
        self.mseloss = torch.nn.MSELoss(reduction="mean")

    def forward(self, inputs, targets):
        return self.mseloss(inputs, targets)
