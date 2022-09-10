# !/usr/bin/env python
"""

"""

import torch
import torch.utils.data
import torchio as tio
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import numpy as np
from glob import glob
import torchvision as tv

from evaluation.metrics import (SegmentationLoss, ConsistencyLoss)
from utils.vessel_utils import (load_model, load_model_with_amp, save_model, write_epoch_summary)

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class Pipeline:

    def __init__(self, cmd_args, UNet1, UNet2, logger, dir_path, checkpoint_path, writer_training, writer_validating, wandb=None):

        self.UNet1 = UNet1
        self.UNet2 = UNet2
        self.logger = logger
        self.wandb = wandb
        self.learning_rate = cmd_args.learning_rate
        self.optimizer1 = torch.optim.Adam(UNet1.parameters(), lr=cmd_args.learning_rate)
        self.optimizer2 = torch.optim.Adam(UNet2.parameters(), lr=cmd_args.learning_rate)
        self.num_epochs = cmd_args.num_epochs
        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.CHECKPOINT_PATH = checkpoint_path
        self.DATASET_PATH = dir_path
        self.OUTPUT_PATH = cmd_args.output_path

        self.model_name = cmd_args.model_name
        self.clip_grads = cmd_args.clip_grads
        self.with_apex = cmd_args.apex

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
        self.random_transforms = [tv.transforms.RandomAffine(degrees=(30, 70), scale=(0.5, 0.75))]
        self.segcor_weight1 = cmd_args.segcor_weight1
        self.segcor_weight2 = cmd_args.segcor_weight2
        self.segmentation_loss = SegmentationLoss()
        self.consistency_loss = ConsistencyLoss()

        # Following metrics can be used to evaluate
        # self.dice = Dice()
        # self.focalTverskyLoss = FocalTverskyLoss()
        # self.iou = IOU()

        self.LOWEST_LOSS = float('inf')

        if self.with_apex:
            self.scaler = GradScaler()

        self.logger.info("Model Hyper Params: ")
        self.logger.info("\nLearning Rate: " + str(self.learning_rate))

        if cmd_args.train:  # Only if training is to be performed
            training_set = Pipeline.create_tio_sub_ds(vol_path=self.DATASET_PATH + '/train/',
                                                      label_path=self.DATASET_PATH + '/train_label/',
                                                      patch_size=self.patch_size,
                                                      samples_per_epoch=self.samples_per_epoch,
                                                      stride_length=self.stride_length, stride_width=self.stride_width,
                                                      stride_depth=self.stride_depth, num_worker=self.num_worker)
            self.train_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=0)
            validation_set, num_subjects = Pipeline.create_tio_sub_ds(vol_path=self.DATASET_PATH + '/validate/',
                                                                      label_path=self.DATASET_PATH + '/validate_label/',
                                                                      patch_size=self.patch_size,
                                                                      samples_per_epoch=self.samples_per_epoch,
                                                                      stride_length=self.stride_length,
                                                                      stride_width=self.stride_width,
                                                                      stride_depth=self.stride_depth,
                                                                      num_worker=self.num_worker,
                                                                      is_train=False)
            self.validate_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size,
                                                               shuffle=False, num_workers=self.num_worker)

    @staticmethod
    def create_tio_sub_ds(vol_path, label_path, patch_size, samples_per_epoch, stride_length, stride_width, stride_depth, num_worker,
                          is_train=True, get_subjects_only=False):

        vols = glob(vol_path + "*.nii") + glob(vol_path + "*.nii.gz")
        labels = glob(label_path + "*.nii") + glob(label_path + "*.nii.gz")
        subjects = []
        for i in range(len(vols)):
            v = vols[i]
            filename = os.path.basename(v).split('.')[0]
            l = [s for s in labels if filename in s][0]
            subject = tio.Subject(
                img=tio.ScalarImage(v),
                label=tio.LabelMap(l),
                subjectname=filename,
            )

            vol_transforms = tio.ToCanonical(), tio.Resample('label')
            transform = tio.Compose(vol_transforms)
            subject = transform(subject)
            subjects.append(subject)

        if get_subjects_only:
            return subjects

        if is_train:
            subjects_dataset = tio.SubjectsDataset(subjects)
            sampler = tio.data.UniformSampler(patch_size)
            patches_queue = tio.Queue(
                subjects_dataset,
                max_length=(samples_per_epoch // len(subjects)) * 4,
                samples_per_volume=(samples_per_epoch // len(subjects)),
                sampler=sampler,
                num_workers=num_worker,
                start_background=True
            )
            return patches_queue
        else:
            overlap = np.subtract(patch_size, (stride_length, stride_width, stride_depth))
            grid_samplers = []
            for i in range(len(subjects)):
                grid_sampler = tio.inference.GridSampler(
                    subjects[i],
                    patch_size,
                    overlap,
                )
                grid_samplers.append(grid_sampler)
            return torch.utils.data.ConcatDataset(grid_samplers), len(grid_samplers)

    @staticmethod
    def normaliser(batch):
        for i in range(batch.shape[0]):
            if batch[i].max() > 0.0:
                batch[i] = batch[i] / batch[i].max()
        return batch

    @staticmethod
    def apply_transformation(transforms, ip):
        list_tensor = []
        for idx, batch in enumerate(ip):
            transformed_batch = batch
            for transform in transforms:
                transformed_batch = transform(batch)
            list_tensor.append(transformed_batch)
        return torch.stack(list_tensor, dim=0)

    def load(self, checkpoint_path=None, load_best=True):
        if checkpoint_path is None:
            checkpoint_path = self.CHECKPOINT_PATH

        if self.with_apex:
            self.UNet1, self.UNet2, self.optimizer, self.scaler = load_model_with_amp(self.UNet1, self.UNet2, self.optimizer1, self.optimizer2, checkpoint_path,
                                                                          batch_index="best" if load_best else "last")
        else:
            self.UNet1, self.UNet2, self.optimizer = load_model(self.UNet1, self.UNet2, self.optimizer1, self.optimizer2, checkpoint_path,
                                                    batch_index="best" if load_best else "last")

    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.UNet1.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            self.UNet2.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_loss1 = 0
            total_loss2 = 0
            total_loss = 0
            batch_index = 0

            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):
                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_labels = Pipeline.normaliser(patches_batch['label'][tio.DATA].float().cuda())

                local_batch_aug = Pipeline.apply_transformation(self.random_transforms, local_batch)
                local_labels_aug = Pipeline.apply_transformation(self.random_transforms, local_labels)

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

                # Clear gradients
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()

                # try:
                with autocast(enabled=self.with_apex):
                    # Get the classification response map(normalized) and respective class assignments after argmax
                    model_output = self.UNet1(local_batch)
                    model_output = torch.sigmoid(model_output)
                    model_output_aug = self.UNet2(local_batch_aug)
                    model_output_aug = torch.sigmoid(model_output_aug)

                    seg_loss = self.segmentation_loss(model_output, local_labels)
                    seg_loss_aug = self.segmentation_loss(model_output_aug, local_labels_aug)

                    _, indx = seg_loss.sort()
                    _, indx_aug = seg_loss_aug.sort()

                    loss1_seg1 = self.segmentation_loss(Pipeline.apply_transformation(self.random_transforms,
                                                        model_output[indx_aug[0:2], :, :, :, :]),
                                                        local_labels_aug[indx_aug[0:2], :, :, :, :]).mean()
                    loss2_seg1 = self.segmentation_loss(model_output_aug[indx[0:2], :, :, :, :],
                                                        Pipeline.apply_transformation(self.random_transforms,
                                                        local_labels[indx[0:2], :, :, :, :])).mean()
                    loss1_seg2 = self.segmentation_loss(Pipeline.apply_transformation(self.random_transforms,
                                                        model_output[indx_aug[2:], :, :, :, :]),
                                                        local_labels_aug[indx_aug[2:], :, :, :, :]).mean()
                    loss2_seg2 = self.segmentation_loss(model_output_aug[indx[2:], :, :, :, :],
                                                        Pipeline.apply_transformation(self.random_transforms,
                                                        local_labels[indx[2:], :, :, :, :])).mean()

                    consistency_loss1 = self.consistency_loss(Pipeline.apply_transformation(self.random_transforms,
                                                              model_output[indx_aug[2:], :, :, :, :]),
                                                              local_labels_aug[indx_aug[2:], :, :, :, :]).mean()

                    loss1 = (self.segcor_weight1 * (loss1_seg1 + loss1_seg2)) + (self.segcor_weight2 * consistency_loss1)

                    consistency_loss2 = self.consistency_loss(model_output_aug[indx[2:], :, :, :, :],
                                                              Pipeline.apply_transformation(self.random_transforms,
                                                              local_labels[indx[2:], :, :, :, :])).mean()
                    loss2 = (self.segcor_weight1 * (loss2_seg1 + loss2_seg2)) + (self.segcor_weight2 * consistency_loss2)

                    total_loss = loss1 + loss2

                # except Exception as error:
                #     self.logger.exception(error)
                #     sys.exit()

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + " Training..." +
                                 "\n loss1: " + str(loss1) + " loss2: " +
                                 str(loss2) + " total_loss: " + str(total_loss))

                # Calculating gradients for UNet1
                if self.with_apex:
                    self.scaler.scale(loss1).backward()

                    if self.clip_grads:
                        self.scaler.unscale_(self.optimizer1)
                        torch.nn.utils.clip_grad_norm_(self.UNet1.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.scaler.step(self.optimizer1)
                    self.scaler.update()
                else:
                    if not torch.any(torch.isnan(loss1)):
                        loss1.backward()
                    else:
                        self.logger.info("nan found in floss.... no backpropagation!!")
                    if self.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.UNet1.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.optimizer1.step()

                # Calculating gradients for UNet2
                if self.with_apex:
                    self.scaler.scale(loss2).backward()
                    if self.clip_grads:
                        self.scaler.unscale_(self.optimizer2)
                        torch.nn.utils.clip_grad_norm_(self.UNet2.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
                    self.scaler.step(self.optimizer2)
                    self.scaler.update()
                else:
                    if not torch.any(torch.isnan(loss2)):
                        loss2.backward()
                    else:
                        self.logger.info("nan found in floss.... no backpropagation!!")
                    if self.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.UNet2.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
                    self.optimizer2.step()

                training_batch_index += 1

                # Initialising the average loss metrics
                total_loss1 += loss1.detach().item()
                total_loss2 += loss2.detach().item()
                total_loss += total_loss.detach().item()

                # To avoid memory errors
                torch.cuda.empty_cache()

            # Calculate the average loss per batch in one epoch
            total_loss1 /= (batch_index + 1.0)
            total_loss2 /= (batch_index + 1.0)
            total_loss /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n loss1: " + str(total_loss1) + " loss2: " +
                             str(total_loss2) + " total_loss: " + str(total_loss))
            write_epoch_summary(writer=self.writer_training, index=epoch,
                                loss1=total_loss1,
                                loss2=total_loss2,
                                total_loss=total_loss)

            if self.wandb is not None:
                self.wandb.log({"loss1_train": total_loss1,
                                "loss2_train": total_loss2,
                                "total_loss_train": total_loss})

            save_model(self.CHECKPOINT_PATH, {
                'epoch_type': 'last',
                'epoch': epoch,
                # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved after validate)
                'state_dict': [self.UNet1.state_dict(), self.UNet2.state_dict()],
                'optimizer': [self.optimizer1.state_dict(), self.optimizer2.state_dict()],
                'amp': self.scaler.state_dict()
            })

            torch.cuda.empty_cache()  # to avoid memory errors
            self.validate(training_batch_index, epoch)
            torch.cuda.empty_cache()  # to avoid memory errors

        return self.UNet1, self.UNet2

    def validate(self, training_index, epoch):
        """
        Method to validate
        :param training_index: Epoch after which validation is performed(can be anything for test)
        :param epoch: Current training epoch
        :return:
        """
        self.logger.debug('Validating...')
        print("Validate Epoch: " + str(epoch) + " of " + str(self.num_epochs))
        writer = self.writer_validating
        self.UNet1.eval()
        self.UNet2.eval()

        total_loss1 = 0
        total_loss2 = 0
        total_loss = 0
        no_patches = 0

        for batch_index, patches_batch in enumerate(tqdm(self.validate_loader)):
            self.logger.info("loading" + str(batch_index))
            no_patches += 1
            local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
            local_labels = Pipeline.normaliser(patches_batch['label'][tio.DATA].float().cuda())

            local_batch_aug = Pipeline.apply_transformation(self.random_transforms, local_batch)
            local_labels_aug = Pipeline.apply_transformation(self.random_transforms, local_labels)

            # Transfer to GPU
            self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

            # try:
            with autocast(enabled=self.with_apex):
                # Get the classification response map(normalized) and respective class assignments after argmax
                model_output = self.UNet1(local_batch)
                model_output = torch.sigmoid(model_output)
                model_output_aug = self.UNet2(local_batch_aug)
                model_output_aug = torch.sigmoid(model_output_aug)

                seg_loss = self.segmentation_loss(model_output, local_labels)
                seg_loss_aug = self.segmentation_loss(model_output_aug, local_labels_aug)

                _, indx = seg_loss.sort()
                _, indx_aug = seg_loss_aug.sort()

                loss1_seg1 = self.segmentation_loss(Pipeline.apply_transformation(self.random_transforms,
                                                    model_output[indx_aug[0:2], :, :, :, :]),
                                                    local_labels_aug[indx_aug[0:2], :, :, :, :]).mean()
                loss2_seg1 = self.segmentation_loss(model_output_aug[indx[0:2], :, :, :, :],
                                                    Pipeline.apply_transformation(self.random_transforms,
                                                    local_labels[indx[0:2], :, :, :, :])).mean()
                loss1_seg2 = self.segmentation_loss(Pipeline.apply_transformation(self.random_transforms,
                                                    model_output[indx_aug[2:], :, :, :, :]),
                                                    local_labels_aug[indx_aug[2:], :, :, :, :]).mean()
                loss2_seg2 = self.segmentation_loss(model_output_aug[indx[2:], :, :, :, :],
                                                    Pipeline.apply_transformation(self.random_transforms,
                                                    local_labels[indx[2:], :, :, :, :])).mean()

                consistency_loss1 = self.consistency_loss(Pipeline.apply_transformation(self.random_transforms,
                                                          model_output[indx_aug[2:], :, :, :, :]),
                                                          local_labels_aug[indx_aug[2:], :, :, :, :]).mean()

                loss1 = (self.segcor_weight1 * (loss1_seg1 + loss1_seg2)) + (self.segcor_weight2 * consistency_loss1)

                consistency_loss2 = self.consistency_loss(model_output_aug[indx[2:], :, :, :, :],
                                                          Pipeline.apply_transformation(self.random_transforms,
                                                          local_labels[indx[2:], :, :, :, :])).mean()
                loss2 = (self.segcor_weight1 * (loss2_seg1 + loss2_seg2)) + (self.segcor_weight2 * consistency_loss2)

                total_loss = loss1 + loss2

                # Log validation losses
                self.logger.info("Batch_Index:" + str(batch_index) + " Validation..." +
                                 "\n loss1: " + str(loss1) + " loss2: " +
                                 str(loss2) + " total_loss: " + str(total_loss))

                total_loss1 += loss1.detach().cpu()
                total_loss2 += loss2.detach().cpu()
                total_loss += total_loss.detach().cpu()

        # Average the losses
        total_loss1 = total_loss1 / no_patches
        total_loss2 = total_loss2 / no_patches
        total_loss = total_loss / no_patches

        process = ' Validating'
        self.logger.info("Epoch:" + str(training_index) + process + "..." +
                         "\n Loss1:" + str(total_loss1) +
                         "\n Loss2:" + str(total_loss2) +
                         "\n total_loss:" + str(total_loss))

        # write_summary(writer, training_index, similarity_loss=total_similarity_loss,
        #               continuity_loss=total_continuity_loss, total_loss=total_loss)
        write_epoch_summary(writer, epoch, loss1=total_loss1,
                            loss2=total_loss2,
                            total_loss=total_loss)

        if self.wandb is not None:
            self.wandb.log({"loss1_val": total_loss1,
                            "loss2_val": total_loss2,
                            "total_loss_val": total_loss})

        if self.LOWEST_LOSS > total_loss:  # Save best metric evaluation weights
            self.LOWEST_LOSS = total_loss
            self.logger.info(
                'Best metric... @ epoch:' + str(training_index) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            save_model(self.CHECKPOINT_PATH, {
                'epoch_type': 'best',
                'epoch': epoch,
                'state_dict': [self.UNet1.state_dict(), self.UNet2.state_dict()],
                'optimizer': [self.optimizer1.state_dict(), self.optimizer2.state_dict()],
                'amp': self.scaler.state_dict()})

    def test(self, test_logger, test_subjects=None, save_results=True):
        test_logger.debug('Testing... TODO')

    def predict(self, image_path, label_path, predict_logger):
        predict_logger.debug('Predicting... TODO')
