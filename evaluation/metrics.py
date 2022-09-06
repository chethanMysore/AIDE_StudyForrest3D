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
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        # Binary cross entropy
        self.BCE_logits_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        BCE_loss = self.BCE_logits_loss(y_pred, y_true)
        return BCE_loss


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        # MSE Loss
        self.mseloss = torch.nn.MSELoss(reduction="mean")

    def forward(self, inputs, targets):
        return self.mseloss(inputs, targets)
