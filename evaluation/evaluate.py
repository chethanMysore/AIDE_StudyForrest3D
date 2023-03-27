#!/usr/bin/env python

# from __future__ import print_function, division
'''

Purpose : 

'''

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

class Dice(nn.Module):
    """
    Class used to get dice_loss and dice_score
    """

    def __init__(self, smooth=1):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss, dice_score


class IOU(nn.Module):
    def __init__(self, smooth=1):
        super(IOU, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f) - intersection
        score = (intersection + self.smooth) / (union + self.smooth)
        return score


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    # def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
    #     # comment out if your model contains a sigmoid or equivalent activation layer
    #     inputs = F.sigmoid(inputs)
    #
    #     # flatten label and prediction tensors
    #     inputs = inputs.view(-1)
    #     targets = targets.view(-1)
    #
    #     # True Positives, False Positives & False Negatives
    #     TP = (inputs * targets).sum()
    #     FP = ((1 - targets) * inputs).sum()
    #     FN = (targets * (1 - inputs)).sum()
    #
    #     Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    #     FocalTversky = (1 - Tversky) ** gamma
    #
    #     return FocalTversky

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky




class FocalTverskyLoss_detailed(nn.Module):
    def __init__(self, smooth=1, gamma=0.75, alpha=0.7):
        super(FocalTverskyLoss_detailed, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logger, y_pred, y_true):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        logger.info("True Positive:" + str(true_pos) + " False_Negative:" + str(false_neg) + " False_Positive:" + str(false_pos))
        pt_1 = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return pow((1 - pt_1), self.gamma)

class MIP_Loss(nn.Module):
    def __init__(self, smooth=1, gamma=0.75, alpha=0.7):
        super(MIP_Loss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, patches_batch, pre_loaded_labels, loss_fn, patch_size):
        mip_loss = 0
        for index, pred in enumerate(y_pred):
            patch_subject_name = patches_batch['subjectname'][index]
            label_3d = [lbl for lbl in pre_loaded_labels if lbl['subjectname'] == patch_subject_name][0]
            label_3d = torch.from_numpy(label_3d['data']).float().cuda()
            patch_width_coord, patch_length_coord, patch_depth_coord = patches_batch["start_coords"][index][0]

            true_mip = torch.amax(label_3d, -1)
            true_mip_patch = true_mip[patch_width_coord:patch_width_coord + patch_size,
                             patch_length_coord:patch_length_coord + patch_size]
            predicted_patch_mip = torch.amax(pred, -1)
            pad = ()
            for dim in range(len(true_mip_patch.shape)):
                target_shape = true_mip_patch.shape[::-1]
                pad_needed = patch_size - target_shape[dim]
                pad_dim = (pad_needed // 2, pad_needed - (pad_needed // 2))
                pad += pad_dim

            true_mip_patch = torch.nn.functional.pad(true_mip_patch, pad[:6], value=np.finfo(np.float).eps)
            mip_loss += loss_fn(predicted_patch_mip, true_mip_patch)
        mip_loss = mip_loss / (len(y_pred) + 0.0001)
        return mip_loss


def getMetric(logger, y_pred, y_true):
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    intersection = true_pos
    union = torch.sum(y_true_pos + y_pred_pos)
    return true_pos, false_neg, false_pos, intersection, union


def getLosses(logger, true_pos, false_neg, false_pos, intersection, union):
    smooth = 1
    gamma = 0.75
    alpha = 0.7

    dice_score = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score
    iou = (intersection + smooth) / (union - intersection + smooth)
    pt_1 = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    floss = pow((1 - pt_1), gamma)

    logger.info("True Positive:" + str(true_pos) + " False_Negative:" + str(false_neg) + " False_Positive:" + str(false_pos))
    logger.info("Floss:" + str(floss) + " diceloss:" + str(dice_loss) + " iou:" + str(iou))
    return floss, dice_loss, iou


class segmentationLossImage(nn.Module):

    def forward(self, inputs, targets):
        N = targets.size(0)
        inputs = inputs.view(N, 256, 256)
        # dice loss
        dice_inputs = inputs[:]
        dice_inputs = torch.sigmoid(dice_inputs)
        # flatten
        iflat = dice_inputs.view(N, -1)
        tflat = targets.view(N, -1)
        intersection = (iflat * tflat).sum(1)
        dice = (2. * intersection + 1) / (iflat.sum(1) + tflat.sum(1) + 1)
        dice_loss = 1 - dice
        # Binary cross entropy
        # BCE_logits_loss = torch.nn.BCEWithLogitsLoss()
        # BCE_loss = BCE_logits_loss(inputs, targets.float())

        fcloss = FocalTverskyLoss()
        fcloss_ = fcloss(inputs, targets.float())
        return dice_loss + (1 * fcloss_)


class consistencyLossImage(nn.Module):

    def forward(self, inputs, targets):
        inputs = torch.nn.functional.sigmoid(inputs)
        mseloss = torch.nn.MSELoss(reduction="mean")
        return mseloss(inputs, targets)


def Dice_fn(inputs, targets, threshold=0.5):
    inputs = torch.nn.functional.sigmoid(inputs)
    dice = 0.
    img_count = 0
    for input_, target_ in zip(inputs, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        dice_single = ((2. * intersection) / (iflat.sum() + tflat.sum()))
        img_count += 1
        dice += dice_single
    return dice