# !/usr/bin/env python
"""

"""

import os
import random
import sys
from glob import glob

import nibabel
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchio as tio
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from evaluation.evaluate import (IOU, Dice, FocalTverskyLoss, MIP_Loss, getLosses, segmentationLossImage,
                                 consistencyLossImage, getMetric, Dice_fn)
from utils.elastic_transform import RandomElasticDeformation, warp_image
from utils.result_analyser import *
from utils.vessel_utils import (convert_and_save_tif, create_diff_mask,
                                create_mask, load_model, load_model_with_amp,
                                save_model, write_summary, write_Epoch_summary)
from utils.datasets import prostate_seg
from utils.transforms import Compose, Resize, RandomRotate, RandomHorizontallyFlip, ToTensor, Normalize
from utils.model_manager import getModel

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"


class Pipeline:

    def __init__(self, cmd_args, net1, net2, logger, dir_path, checkpoint_path, writer_training, writer_validating, test_logger,
                 training_set=None, validation_set=None, test_set=None, wandb=None):

        self.logger = logger
        self.wandb = wandb
        self.net1 = net1
        self.net2 = net2
        self.MODEL_NAME = cmd_args.model_name
        self.model_type = cmd_args.model
        self.lr_1 = cmd_args.learning_rate
        self.logger.info("learning rate " + str(self.lr_1))
        self.optimizer1 = torch.optim.Adam(net1.parameters(), lr=cmd_args.learning_rate, amsgrad=True)
        self.optimizer2 = torch.optim.Adam(net2.parameters(), lr=cmd_args.learning_rate, amsgrad=True)
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizer1, step_size=20, gamma=0.1)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.optimizer2, step_size=20, gamma=0.1)
        self.num_epochs = cmd_args.num_epochs
        self.k_folds = cmd_args.k_folds
        self.learning_rate = cmd_args.learning_rate

        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.test_logger = test_logger
        self.checkpoint_path = checkpoint_path
        self.load_path = cmd_args.load_path
        self.DATASET_FOLDER = dir_path
        self.output_path = cmd_args.output_path

        self.model_name = cmd_args.model_name

        self.clip_grads = cmd_args.clip_grads
        self.with_apex = cmd_args.apex
        self.deform = cmd_args.deform

        # image input parameters
        self.patch_size = cmd_args.patch_size
        self.stride_depth = cmd_args.stride_depth
        self.stride_length = cmd_args.stride_length
        self.stride_width = cmd_args.stride_width
        self.samples_per_epoch = cmd_args.samples_per_epoch

        # execution configs
        self.batch_size = cmd_args.batch_size
        self.num_worker = cmd_args.num_worker

        # Losses
        self.floss_param_smooth = cmd_args.floss_param_smooth
        self.floss_param_gamma = cmd_args.floss_param_gamma
        self.floss_param_alpha = cmd_args.floss_param_alpha
        self.mip_loss_param_smooth = cmd_args.mip_loss_param_smooth
        self.mip_loss_param_gamma = cmd_args.mip_loss_param_gamma
        self.mip_loss_param_alpha = cmd_args.mip_loss_param_alpha
        self.dice = Dice()
        self.focalTverskyLoss = FocalTverskyLoss()
        self.mip_loss = FocalTverskyLoss()
        self.seg_loss = segmentationLossImage()
        self.consistency_loss = consistencyLossImage()
        self.floss_coeff = cmd_args.floss_coeff
        self.mip_loss_coeff = cmd_args.mip_loss_coeff
        self.iou = IOU()

        self.BEST_DICE1 = float('inf')
        self.BEST_DICE2 = float('inf')
        self.test_set = test_set
        self.train_root = cmd_args.train_root
        self.test_root = cmd_args.test_root
        self.train_csv = cmd_args.train_csv
        self.traincase_csv = cmd_args.traincase_csv
        self.test_csv = cmd_args.test_csv
        self.testcase_csv = cmd_args.testcase_csv
        self.tempmaskfolder = "train_aug"
        self.temperature = cmd_args.temperature
        self.segcor_weight = cmd_args.segcor_weight
        self.train_aug = Compose([
            Resize(size=(256, 256)),
            RandomRotate(60),
            RandomHorizontallyFlip(),
            ToTensor(),
            Normalize(mean=None,
                      std=None)])
        self.test_aug = Compose([
            Resize(size=(256, 256)),
            ToTensor(),
            Normalize(mean=None,
                      std=None)])
        os.makedirs(os.path.join(self.train_root, self.tempmaskfolder), exist_ok=True)

        if self.with_apex:
            self.scaler = GradScaler()

    def normaliser(self, batch):
        for i in range(batch.shape[0]):
            if batch[i].max() > 0.0:
                batch[i] = batch[i] / batch[i].max()
        return batch

    def load(self, checkpoint_path=None, load_best=True, fold_index=""):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        print('Loading models...')
        checkpoint_net1 = torch.load(os.path.join(self.checkpoint_path + "/net1/", 'checkpointbest' + str(fold_index) + '.pth'))
        checkpoint_net2 = torch.load(os.path.join(self.checkpoint_path + "/net2/", 'checkpointbest' + str(fold_index) + '.pth'))
        self.net1.load_state_dict(checkpoint_net1['state_dict'])
        self.optimizer1.load_state_dict(checkpoint_net1['optimizer'])
        self.scheduler1.load_state_dict(checkpoint_net1['scheduler'])
        self.net2.load_state_dict(checkpoint_net2['state_dict'])
        self.optimizer2.load_state_dict(checkpoint_net2['optimizer'])
        self.scheduler2.load_state_dict(checkpoint_net2['scheduler'])
        self.net1.eval()
        self.net2.eval()

    def reset(self):
        del self.net1
        del self.net2
        self.net1 = torch.nn.DataParallel(getModel(self.model_type, self.output_path + "/" + self.MODEL_NAME))
        self.net1.cuda()
        self.net2 = torch.nn.DataParallel(getModel(self.model_type, self.output_path + "/" + self.MODEL_NAME))
        self.net2.cuda()
        self.optimizer1 = torch.optim.Adam(self.net1.parameters(), lr=self.learning_rate, amsgrad=True)
        self.optimizer2 = torch.optim.Adam(self.net2.parameters(), lr=self.learning_rate, amsgrad=True)
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizer1, step_size=20, gamma=0.1)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.optimizer2, step_size=20, gamma=0.1)
        self.BEST_DICE1 = float('inf')
        self.BEST_DICE2 = float('inf')

    def reverseaug(self, augset, augoutput, classno):
        for batch_idx in range(len(augset['augno'])):
            for aug_idx in range(augset['augno'][batch_idx]):
                imgflip = augset['hflip{}'.format(aug_idx + 1)][batch_idx]
                rotation = 0 - augset['degree{}'.format(aug_idx + 1)][batch_idx]
                for classidx in range(classno):
                    mask = augoutput[aug_idx][batch_idx, classidx, :, :]
                    mask = mask.cpu().numpy()
                    mask = Image.fromarray(mask, mode='F')
                    if imgflip:
                        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.rotate(rotation, Image.BILINEAR)
                    mask = torch.from_numpy(np.array(mask))
                    augoutput[aug_idx][batch_idx, classidx, :, :] = mask
        return augoutput

    def sharpen(self, mask, temperature):
        masktemp = torch.pow(mask, temperature)
        masktempsum = masktemp.sum(dim=1).unsqueeze(dim=1)
        sharpenmask = masktemp / masktempsum
        return sharpenmask

    def train(self):
        self.logger.debug("Training...")
        train_cases = pd.read_csv(self.traincase_csv)['Image'].tolist()
        test_cases = pd.read_csv(self.testcase_csv)['Image'].tolist()
        train_data = pd.read_csv(self.train_csv)
        test_data = pd.read_csv(self.test_csv)
        random.shuffle(train_cases)
        # get k folds for cross validation
        folds = [train_cases[i::self.k_folds] for i in range(self.k_folds)]
        for fold_index in range(self.k_folds):
            train_cases = []
            for idx, fold in enumerate(folds):
                if idx != fold_index:
                    train_cases.extend([*fold])
            validation_cases = [*folds[fold_index]]
            # create training and validation samples from folds
            train_samples = train_data.applymap(lambda row: row if [vol for vol in train_cases
                                                                    if row.startswith(str(vol) + "/Image")
                                                                    or row.startswith(str(vol) + "/Mask")] else None)
            train_samples = train_samples[train_samples['Image'].notnull()]
            train_data_imgs = train_samples['Image'].values.tolist()
            train_data_masks = train_samples['Mask'].values.tolist()

            validation_samples = train_data.applymap(lambda row: row if [vol for vol in validation_cases
                                                                         if row.startswith(str(vol) + "/Image")
                                                                         or row.startswith(
                    str(vol) + "/Mask")] else None)
            validation_samples = validation_samples[validation_samples['Image'].notnull()]
            validation_data_imgs = validation_samples['Image'].values.tolist()
            validation_data_masks = validation_samples['Mask'].values.tolist()

            train_dataset = prostate_seg(root=self.train_root, csv_file=None, tempmaskfolder=self.tempmaskfolder,
                                         transform=self.train_aug, imgs=train_data_imgs, masks=train_data_masks)
            validation_dataset = prostate_seg(root=self.train_root, csv_file=None, tempmaskfolder=self.tempmaskfolder,
                                              transform=self.test_aug, imgs=validation_data_imgs,
                                              masks=validation_data_masks)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                      num_workers=4, shuffle=True, drop_last=True)
            validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size,
                                           num_workers=4, shuffle=False)

            print("Train Fold: " + str(fold_index) + " of " + str(self.k_folds))
            for epoch in range(self.num_epochs):
                print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
                self.net1.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
                self.net2.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
                num_classes = 1

                train_loss1 = 0.
                train_dice1 = 0.
                train_count = 0
                train_loss2 = 0.
                train_dice2 = 0.

                for batch_idx, (inputs, augset, targets, targets1, targets2) in \
                        tqdm(enumerate(train_loader), total=int(len(train_loader.dataset) / self.batch_size)):

                    augoutput1 = []
                    augoutput2 = []

                    for aug_idx in range(augset['augno'][0]):
                        augimg = augset['img{}'.format(aug_idx + 1)].cuda()
                        augimg = augimg.unsqueeze(dim=1)
                        augoutput1.append(self.net1(augimg).detach())
                        augoutput2.append(self.net2(augimg).detach())

                    augoutput1 = self.reverseaug(augset, augoutput1, classno=num_classes)
                    augoutput2 = self.reverseaug(augset, augoutput2, classno=num_classes)

                    for aug_idx in range(len(augoutput1)):
                        augoutput1[aug_idx] = augoutput1[aug_idx].view(targets1.size(0), 256, 256)

                    for aug_idx in range(len(augoutput2)):
                        augoutput2[aug_idx] = augoutput2[aug_idx].view(targets2.size(0), 256, 256)

                    for aug_idx in range(augset['augno'][0]):
                        augmask1 = torch.nn.functional.sigmoid(augoutput1[aug_idx])
                        augmask2 = torch.nn.functional.sigmoid(augoutput2[aug_idx])

                        if aug_idx == 0:
                            pseudo_label1 = augmask1
                            pseudo_label2 = augmask2
                        else:
                            pseudo_label1 += augmask1
                            pseudo_label2 += augmask2

                    pseudo_label1 = pseudo_label1 / float(augset['augno'][0])
                    pseudo_label2 = pseudo_label2 / float(augset['augno'][0])
                    pseudo_label1 = self.sharpen(pseudo_label1, self.temperature)
                    pseudo_label2 = self.sharpen(pseudo_label2, self.temperature)

                    weightmap1 = 1.0 - 4.0 * pseudo_label1
                    weightmap2 = 1.0 - 4.0 * pseudo_label2

                    inputs = inputs.cuda()
                    targets1 = targets1.cuda()
                    targets2 = targets2.cuda()
                    self.optimizer1.zero_grad()
                    self.optimizer2.zero_grad()
                    inputs = inputs.unsqueeze(dim=1)
                    outputs1 = self.net1(inputs)
                    outputs2 = self.net2(inputs)

                    outputs1 = outputs1.view(targets1.size(0), 256, 256)
                    outputs2 = outputs2.view(targets2.size(0), 256, 256)

                    loss1_segpre = self.seg_loss(outputs1, targets2)
                    loss2_segpre = self.seg_loss(outputs2, targets1)

                    _, indx1 = loss1_segpre.sort()
                    _, indx2 = loss2_segpre.sort()

                    loss1_seg1 = self.seg_loss(outputs1[indx2[0:2], :, :], targets2[indx2[0:2], :, :]).mean()
                    loss2_seg1 = self.seg_loss(outputs2[indx1[0:2], :, :], targets1[indx1[0:2], :, :]).mean()
                    loss1_seg2 = self.seg_loss(outputs1[indx2[2:], :, :], targets2[indx2[2:], :, :]).mean()
                    loss2_seg2 = self.seg_loss(outputs2[indx1[2:], :, :], targets1[indx1[2:], :, :]).mean()

                    loss1_cor = weightmap2[indx2[2:], :, :] * self.consistency_loss(outputs1[indx2[2:], :, :],
                                                                               pseudo_label2[indx2[2:], :, :])
                    loss1_cor = loss1_cor.mean()
                    loss1 = self.segcor_weight[0] * (loss1_seg1 + loss1_seg2) + \
                            self.segcor_weight[1] * loss1_cor

                    loss2_cor = weightmap1[indx1[2:], :, :] * self.consistency_loss(outputs2[indx1[2:], :, :],
                                                                               pseudo_label1[indx1[2:], :, :])
                    loss2_cor = loss2_cor.mean()
                    loss2 = self.segcor_weight[0] * (loss2_seg1 + loss2_seg2) + \
                            self.segcor_weight[1] * loss2_cor

                    loss1.backward(retain_graph=True)
                    self.optimizer1.step()
                    loss2.backward()
                    self.optimizer2.step()

                    train_count += inputs.shape[0]
                    train_loss1 += loss1.item() * inputs.shape[0]
                    train_dice1 += Dice_fn(outputs1, targets2).item()
                    train_loss2 += loss2.item() * inputs.shape[0]
                    train_dice2 += Dice_fn(outputs2, targets1).item()

                train_loss1_epoch = train_loss1 / float(train_count)
                train_dice1_epoch = train_dice1 / float(train_count)
                train_loss2_epoch = train_loss2 / float(train_count)
                train_dice2_epoch = train_dice2 / float(train_count)

                self.logger.info("Fold:" + str(fold_index) + " Epoch:" + str(epoch) + " Average Training..." +
                                 "\n train_loss1:" + str(train_loss1_epoch) + " train_dice1: " + str(
                    train_dice1_epoch) + " train_loss2: " + str(train_loss2_epoch) + " train_dice2: " + str(
                    train_dice2_epoch))

                if self.wandb is not None:
                    self.wandb.log({"train_loss1_" + str(fold_index): train_loss1_epoch,
                                    "train_dice1_" + str(fold_index): train_dice1_epoch,
                                    "train_loss2_" + str(fold_index): train_loss2_epoch,
                                    "train_dice2_" + str(fold_index): train_dice2_epoch})

                torch.cuda.empty_cache()  # to avoid memory errors
                self.validate(fold_index, epoch, validation_loader)
                torch.cuda.empty_cache()  # to avoid memory errors

            # Testing for current fold
            torch.cuda.empty_cache()  # to avoid memory errors
            self.load(fold_index=fold_index)
            self.test(self.test_logger, fold_index=fold_index)
            torch.cuda.empty_cache()  # to avoid memory errors

            # Discard the current model and reset training parameters
            self.reset()

        return self.model

    def validate(self, fold_index, epoch, validation_loader=None):
        """
        Method to validate
        :return:
        """
        self.logger.debug('Validating...')
        print("Validate Fold: " + str(fold_index) + " of " + str(self.k_folds) +
              "Validate Epoch: " + str(epoch) + " of " + str(self.num_epochs))

        self.net1.eval()
        self.net2.eval()

        val_loss1 = 0.
        val_dice1 = 0.
        val_loss2 = 0.
        val_dice2 = 0.
        val_count = 0

        output1_list = []
        output2_list = []

        for batch_idx, (inputs, augset, targets, targets1, targets2) in \
                tqdm(enumerate(validation_loader), total=int(len(validation_loader.dataset) / self.batch_size)):
            with torch.no_grad():
                inputs = inputs.cuda()
                inputs = inputs.unsqueeze(dim=1)
                targets1 = targets1.cuda()
                targets2 = targets2.cuda()
                outputs1 = self.net1(inputs)
                outputs2 = self.net2(inputs)
                outputs1 = outputs1.view(targets1.size(0), 256, 256)
                outputs2 = outputs2.view(targets2.size(0), 256, 256)
                if epoch == 15:
                    outputs1 = torch.sigmoid(outputs1).detach().cpu()
                    outputs2 = torch.sigmoid(outputs2).detach().cpu()
                    output1_list.extend(outputs1)
                    output2_list.extend(outputs2)
                loss1 = self.seg_loss(outputs1, targets2).mean()
                loss2 = self.seg_loss(outputs2, targets1).mean()
            val_count += inputs.shape[0]
            val_loss1 += loss1.item() * inputs.shape[0]
            val_dice1 += Dice_fn(outputs1, targets2).item()
            val_loss2 += loss2.item() * inputs.shape[0]
            val_dice2 += Dice_fn(outputs2, targets1).item()

        val_loss1_epoch = val_loss1 / float(val_count)
        val_dice1_epoch = val_dice1 / float(val_count)
        val_loss2_epoch = val_loss2 / float(val_count)
        val_dice2_epoch = val_dice2 / float(val_count)

        self.logger.info("Fold:" + str(fold_index) + " Epoch:" + str(epoch) + ' Validating' + "..." +
                         "\n val_loss1:" + str(val_loss1_epoch) +
                         "\n val_dice1:" + str(val_dice1_epoch) +
                         "\n val_loss2:" + str(val_loss2_epoch) +
                         "\n val_dice2:" + str(val_dice2_epoch))
        # TODO: Not logging to tensorboard currently
        # write_Epoch_summary(writer, fold_index, focalTverskyLoss=floss, mipLoss=mipLoss, diceLoss=dloss, diceScore=0,
        #                     iou=0, total_loss=total_loss)
        if self.wandb is not None:
            self.wandb.log({"val_loss1_" + str(fold_index): val_loss1_epoch, "val_dice1_" + str(fold_index): val_dice1_epoch, "val_loss2_" + str(fold_index): val_loss2_epoch,
                            "val_dice2_" + str(fold_index): val_dice2_epoch, "epoch": epoch, "fold_index": fold_index})

        if self.BEST_DICE1 < val_dice1_epoch:  # Save best metric evaluation weights
            self.BEST_DICE1 = val_dice1_epoch
            self.logger.info(
                'Best metric... @ fold:' + str(fold_index) + ' Current Best Dice for Net1:' + str(self.BEST_DICE1))

            save_model(self.checkpoint_path + '/net1/', {
                'epoch_type': 'best',
                'epoch': fold_index,
                'state_dict': self.net1.state_dict(),
                'optimizer': self.optimizer1.state_dict(),
                'scheduler': self.scheduler1.state_dict()}, fold_index=fold_index)

        if self.BEST_DICE2 < val_dice2_epoch:  # Save best metric evaluation weights
            self.BEST_DICE2 = val_dice2_epoch
            self.logger.info(
                'Best metric... @ fold:' + str(fold_index) + ' Current Best Dice for Net2:' + str(self.BEST_DICE2))

            save_model(self.checkpoint_path + '/net2/', {
                'epoch_type': 'best',
                'epoch': fold_index,
                'state_dict': self.net2.state_dict(),
                'optimizer': self.optimizer2.state_dict(),
                'scheduler': self.scheduler2.state_dict()}, fold_index=fold_index)

    def pseudo_train(self, test_logger):
        test_logger.debug('Testing With MIP...')

        traindataset = self.create_TIOSubDS(vol_path=self.DATASET_FOLDER + '/train/',
                                            label_path=self.DATASET_FOLDER + '/train_label/')
        sampler = torch.utils.data.RandomSampler(data_source=traindataset, replacement=True,
                                                 num_samples=self.samples_per_epoch)
        self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size, shuffle=False,
                                                        num_workers=self.num_worker, pin_memory=True,
                                                        sampler=sampler)
        result_root = os.path.join(self.output_path, self.model_name, "results")
        result_root = os.path.join(result_root, "mips")
        os.makedirs(result_root, exist_ok=True)
        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.model.eval()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_floss = 0
            total_mipLoss = 0
            total_DiceLoss = 0
            total_IOU = 0
            total_DiceScore = 0
            batch_index = 0
            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):

                local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_labels = patches_batch['label'][tio.DATA].float().cuda()

                local_batch = torch.movedim(local_batch, -1, -3)
                local_labels = torch.movedim(local_labels, -1, -3)

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

                # Clear gradients
                # self.optimizer.zero_grad()

                # try:
                with autocast(enabled=self.with_apex):
                    loss_ratios = [1, 0.66, 0.34]  # TODO param

                    floss = torch.tensor(0.001).float().cuda()
                    mip_loss = torch.tensor(0.001).float().cuda()
                    output1 = 0
                    level = 0
                    diceLoss_batch = 0
                    diceScore_batch = 0
                    IOU_batch = 0

                    # -------------------------------------------------------------------------------------------------
                    # First Branch Supervised error
                    if not self.isProb:
                        # Compute DiceLoss using batch labels
                        for output in self.model(local_batch):
                            if level == 0:
                                output1 = output
                            if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))
                            output = torch.sigmoid(output)

                            floss += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                            # Compute MIP loss from the patch on the MIP of the 3D label and the patch prediction
                            mip_loss_patch = torch.tensor(0.001).float().cuda()
                            num_patches = 0
                            for index, op in enumerate(output):
                                op_mip = torch.amax(op, 1)
                                true_mip = patches_batch['ground_truth_mip_patch'][index].float().cuda()
                                mip_loss_patch += self.focalTverskyLoss(op_mip, true_mip)
                                op_mip = op_mip.detach().cpu().squeeze().numpy()
                                true_mip = true_mip.detach().cpu().squeeze().numpy()
                                Image.fromarray((op_mip * 255).astype('uint8'), 'L').save(
                                    os.path.join(result_root,
                                                 "level_" + str(level) + "_patch_" + str(index) + "_op_mip.tif"))
                                Image.fromarray((true_mip * 255).astype('uint8'), 'L').save(
                                    os.path.join(result_root,
                                                 "level_" + str(level) + "_patch_" + str(index) + "_true_mip.tif"))
                                test_logger.info("Testing with mip..." +
                                                 "\n floss:" + str(floss) +
                                                 "\n mip_loss:" + str(mip_loss_patch))
                            if not torch.any(torch.isnan(mip_loss_patch)):
                                mip_loss += mip_loss_patch / len(output)
                            level += 1

                    test_logger.info("Testing with mip..." +
                                     "\n Average mip_loss:" + str(mip_loss))
                break

    def test_with_MIP(self, test_logger, test_subjects=None):
        test_logger.debug('Testing...')

        if test_subjects is None:
            test_folder_path = self.DATASET_FOLDER + '/test/'
            test_label_path = self.DATASET_FOLDER + '/test_label/'

            test_subjects = self.create_TIOSubDS(vol_path=test_folder_path, is_train=False, label_path=test_label_path,
                                                 get_subjects_only=True)

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))

        df = pd.DataFrame(columns=["Subject", "Dice", "IoU"])
        result_root = os.path.join(self.output_path, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)

        self.model.eval()

        with torch.no_grad():
            for test_subject in test_subjects:
                if 'label' in test_subject:
                    label = test_subject['label'][tio.DATA].float().squeeze().numpy()
                    # del test_subject['label']
                else:
                    label = None
                subjectname = test_subject['subjectname']
                del test_subject['subjectname']

                result_root = os.path.join(result_root, subjectname + "_MIPs")
                os.makedirs(result_root, exist_ok=True)

                grid_sampler = tio.inference.GridSampler(
                    test_subject,
                    self.patch_size,
                    overlap,
                )
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=self.num_worker)

                for index, patches_batch in enumerate(tqdm(patch_loader)):
                    local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    locations = patches_batch[tio.LOCATION]
                    local_label = patches_batch['label'][tio.DATA].float()

                    local_batch = torch.movedim(local_batch, -1, -3)

                    with autocast(enabled=self.with_apex):
                        if not self.isProb:
                            output = self.model(local_batch)
                            if type(output) is tuple or type(output) is list:
                                output = output[0]
                            output = torch.sigmoid(output).detach().cpu()
                        else:
                            self.model.forward(local_batch, training=False)
                            output = self.model.sample(
                                testing=True).detach().cpu()  # TODO: need to check whether sigmoid is needed for prob

                    output = torch.movedim(output, -3, -1)
                    for idx, op in enumerate(output):
                        op_mip = torch.amax(op.squeeze().numpy(), 1)
                        label_mip = torch.amax(local_label[idx].squeeze().numpy(), 1)
                        Image.fromarray((op_mip * 255).astype('uint8'), 'L').save(
                            os.path.join(result_root, subjectname + "_patch" + str(idx) + "_pred_MIP.tif"))
                        Image.fromarray((label_mip * 255).astype('uint8'), 'L').save(
                            os.path.join(result_root, subjectname + "_patch" + str(idx) + "_true_MIP.tif"))

                    aggregator.add_batch(output, locations)

        #         predicted = aggregator.get_output_tensor().squeeze().numpy()
        #
        #         try:
        #             thresh = threshold_otsu(predicted)
        #             result = predicted > thresh
        #         except Exception as error:
        #             test_logger.exception(error)
        #             result = predicted > 0.5  # exception will be thrown only if input image seems to have just one color 1.0.
        #         result = result.astype(np.float32)
        #
        #         if label is not None:
        #             datum = {"Subject": subjectname}
        #             dice3D = dice(result, label)
        #             iou3D = IoU(result, label)
        #             datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3D], "IoU": [iou3D]})
        #             df = pd.concat([df, datum], ignore_index=True)
        #
        #         if save_results:
        #             save_nifti(result, os.path.join(result_root, subjectname + ".nii.gz"))
        #
        #             resultMIP = np.max(result, axis=-1)
        #             Image.fromarray((resultMIP * 255).astype('uint8'), 'L').save(
        #                 os.path.join(result_root, subjectname + "_MIP.tif"))
        #
        #             if label is not None:
        #                 overlay = create_diff_mask_binary(result, label)
        #                 save_tifRGB(overlay, os.path.join(result_root, subjectname + "_colour.tif"))
        #
        #                 overlayMIP = create_diff_mask_binary(resultMIP, np.max(label, axis=-1))
        #                 Image.fromarray(overlayMIP.astype('uint8'), 'RGB').save(
        #                     os.path.join(result_root, subjectname + "_colourMIP.tif"))
        #
        #         test_logger.info("Testing " + subjectname + "..." +
        #                          "\n Dice:" + str(dice3D) +
        #                          "\n JacardIndex:" + str(iou3D))
        #
        # df.to_excel(os.path.join(result_root, "Results_Main.xlsx"))

    def test(self, test_logger, save_results=True, test_subjects=None, fold_index=""):
        test_logger.debug('Testing...')

        if test_subjects is None:
            test_folder_path = self.DATASET_FOLDER + '/test/'
            test_label_path = self.DATASET_FOLDER + '/test_label/'

            test_subjects = self.create_TIOSubDS(vol_path=test_folder_path, is_train=False, label_path=test_label_path,
                                                 get_subjects_only=True)

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))

        df = pd.DataFrame(columns=["Subject", "Dice", "IoU"])
        result_root = os.path.join(self.output_path, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)

        self.net1.eval()

        with torch.no_grad():
            for test_subject in test_subjects:
                if 'label' in test_subject:
                    label = test_subject['label'][tio.DATA].float().squeeze().numpy()
                    del test_subject['label']
                else:
                    label = None
                subjectname = test_subject['subjectname']
                del test_subject['subjectname']

                grid_sampler = tio.inference.GridSampler(
                    test_subject,
                    self.patch_size,
                    overlap,
                )
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=self.num_worker)

                for index, patches_batch in enumerate(tqdm(patch_loader)):
                    local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    locations = patches_batch[tio.LOCATION]

                    local_batch = torch.movedim(local_batch, -1, -3)

                    with autocast(enabled=self.with_apex):
                        if not self.isProb:
                            output = self.net1(local_batch)
                            if type(output) is tuple or type(output) is list:
                                output = output[0]
                            output = torch.sigmoid(output).detach().cpu()
                        else:
                            self.net1.forward(local_batch, training=False)
                            output = self.net1.sample(
                                testing=True).detach().cpu()  # TODO: need to check whether sigmoid is needed for prob

                    output = torch.movedim(output, -3, -1).type(local_batch.type())
                    aggregator.add_batch(output, locations)

                predicted = aggregator.get_output_tensor().squeeze().numpy()

                try:
                    thresh = threshold_otsu(predicted)
                    result = predicted > thresh
                except Exception as error:
                    test_logger.exception(error)
                    result = predicted > 0.5  # exception will be thrown only if input image seems to have just one color 1.0.
                result = result.astype(np.float32)

                if label is not None:
                    datum = {"Subject": subjectname}
                    dice3D = dice(result, label)
                    iou3D = IoU(result, label)
                    datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3D], "IoU": [iou3D]})
                    df = pd.concat([df, datum], ignore_index=True)

                if save_results:
                    save_nifti(result, os.path.join(result_root, subjectname + "_fld" + str(fold_index) + ".nii.gz"))

                    resultMIP = np.max(result, axis=-1)
                    Image.fromarray((resultMIP * 255).astype('uint8'), 'L').save(
                        os.path.join(result_root, subjectname + str(fold_index) + "_MIP.tif"))

                    if label is not None:
                        overlay = create_diff_mask_binary(result, label)
                        save_tifRGB(overlay, os.path.join(result_root, subjectname + "_fld" + str(fold_index) + "_colour.tif"))

                        overlayMIP = create_diff_mask_binary(resultMIP, np.max(label, axis=-1))
                        color_mip = Image.fromarray(overlayMIP.astype('uint8'), 'RGB')
                        color_mip.save(
                            os.path.join(result_root, subjectname + "_fld" + str(fold_index) + "_colourMIP.tif"))
                        if self.wandb is not None:
                            self.wandb.log({"" + subjectname + "_fld" + str(fold_index): self.wandb.Image(color_mip)})


                test_logger.info("Testing " + subjectname + "..." +
                                 "\n Dice:" + str(dice3D) +
                                 "\n JacardIndex:" + str(iou3D))

        df.to_excel(os.path.join(result_root, "Results_Main_fld" + str(fold_index) + ".xlsx"))

    def predict(self, image_path, label_path, predict_logger):
        image_name = os.path.basename(image_path).split('.')[0]

        subdict = {
            "img": tio.ScalarImage(image_path),
            "subjectname": image_name,
        }

        if bool(label_path):
            subdict["label"] = tio.LabelMap(label_path)

        subject = tio.Subject(**subdict)

        self.test(predict_logger, save_results=True, test_subjects=[subject])
