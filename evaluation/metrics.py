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


class SegmentationLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SegmentationLoss, self).__init__()
        self.smooth = smooth
        # Binary cross entropy
        self.BCE_logits_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true,batch_index=0):
        y_pred_dice = torch.sigmoid(y_pred)
        iflat = torch.flatten(y_pred_dice, 1)
        tflat = torch.flatten(y_true, 1)
        intersection = torch.sum(iflat * tflat, 1)
        union = torch.sum(iflat + tflat, 1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        #if torch.any(torch.isnan(dice_loss)):
            #print("nan value found in dice_loss")
            #torch.save(y_pred, f'y_pred_{batch_index}.pt')
            #torch.save(y_true, f"y_true_{batch_index}.pt")

        # Binary cross entropy
        BCE_loss = self.BCE_logits_loss(y_pred, y_true)

        seg_loss = dice_loss + (1 * BCE_loss)
        return seg_loss


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        # MSE Loss
        self.mseloss = torch.nn.MSELoss(reduction="mean")

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return self.mseloss(inputs, targets)



