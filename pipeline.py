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
import pandas as pd
from skimage.filters import threshold_otsu

from evaluation.metrics import (SegmentationLoss, ConsistencyLoss, FocalTverskyLoss, DiceScore)
from utils.customutils import subjects_to_tensors, tensors_to_subjects
from utils.datasets import SRDataset
from utils.results_analyser import *
from utils.vessel_utils import (load_model, load_model_with_amp, save_model, write_epoch_summary)

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class Pipeline:

    def __init__(self, cmd_args, UNet1, UNet2, logger, dir_path, checkpoint_path, writer_training, writer_validating,
                 wandb=None):

        self.UNet1 = UNet1
        self.UNet2 = UNet2
        self.logger = logger
        self.wandb = wandb
        self.learning_rate = cmd_args.learning_rate
        self.optimizer1 = torch.optim.Adam(UNet1.parameters(), lr=cmd_args.learning_rate)
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
        self.focal_tversky_loss = FocalTverskyLoss()
        self.dice_score = DiceScore()
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
            training_set = Pipeline.create_tio_sub_ds(logger=self.logger, vol_path=self.DATASET_PATH + '/train/',
                                                      label_path=self.DATASET_PATH + '/train_label/',
                                                      patch_size=self.patch_size,
                                                      samples_per_epoch=self.samples_per_epoch,
                                                      num_worker=self.num_worker)
            self.train_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=0)
            validation_set = Pipeline.create_tio_sub_ds(logger=self.logger, vol_path=self.DATASET_PATH + '/validate/',
                                                        label_path=self.DATASET_PATH + '/validate_label/',
                                                        patch_size=self.patch_size,
                                                        samples_per_epoch=self.samples_per_epoch,
                                                        num_worker=self.num_worker,
                                                        )
            self.validate_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size,
                                                               shuffle=False, num_workers=0)

    @staticmethod
    def create_tio_sub_ds(logger, vol_path, label_path, num_worker=0, patch_size=None, samples_per_epoch=None,
                          get_subjects_only=False,is_train=True):

        # trainDS = SRDataset(logger=logger, patch_size=64,
        #                     dir_path=vol_path,
        #                     label_dir_path=label_path, pre_load=True,
        #                     return_coords=True
        #                     )
        # return trainDS
        logger.info("creating patch..")
        vols = glob(vol_path + "*.nii") + glob(vol_path + "*.nii.gz")
        labels = glob(label_path + "*.nii") + glob(label_path + "*.nii.gz")
        subjects = []
        for i in range(len(vols)):
            v = vols[i]
            filename = os.path.basename(v).split('.')[0]
            l = [s for s in labels if filename in s][0]
            # to fix spacing issue between img(0.30) and label(1.0). Don't use resample as it messes up img and label
            # patch
            t1 = tio.ScalarImage(v)
            t1 = t1.data[:, :, :, :]
            t1 = tio.ScalarImage(tensor=t1)
            t2 = tio.LabelMap(l)
            t2 = t2.data[:, :, :, :]
            t2 = tio.LabelMap(tensor=t2)
            subject = tio.Subject(
                img=t1,
                label=t2,
                aug_img=t1,
                aug_label=t2,
                subjectname=filename,
            )
            if is_train:
                transforms = tio.RandomFlip(axes=('LR',), flip_probability=1, exclude=["img", "label"])
                subject = transforms(subject)
            subjects.append(subject)

        if get_subjects_only:
            return subjects

        subjects_dataset = tio.SubjectsDataset(subjects)
        sampler = tio.data.UniformSampler(patch_size)
        patches_queue = tio.Queue(
            subjects_dataset,
            max_length=(samples_per_epoch // len(subjects)) * 4,
            samples_per_volume=(samples_per_epoch // len(subjects)),
            sampler=sampler,
            num_workers=0,
            start_background=True
        )
        return patches_queue

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

        self.logger.info(f"Loading checkpoint from {str(checkpoint_path)}")
        if self.with_apex:
            self.UNet1, self.optimizer1, self.scaler = load_model_with_amp(self.UNet1,
                                                                           self.optimizer1,
                                                                           checkpoint_path,
                                                                           batch_index="best" if load_best else "last")
        else:
            self.UNet1, self.optimizer1 = load_model(self.UNet1, self.optimizer1,
                                                     checkpoint_path,
                                                     batch_index="best" if load_best else "last")

    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.UNet1.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_loss = 0
            total_dice_score = 0
            batch_index = 0

            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):
                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_labels = patches_batch['label'][tio.DATA].float().cuda()
                aug_batch = Pipeline.normaliser(patches_batch['aug_img'][tio.DATA].float().cuda())
                aug_labels = patches_batch['aug_label'][tio.DATA].float().cuda()

                ###############################################################################
                # # Transform images
                # subjects = []
                # aug_subjects = []
                # inverse_transform_functions = []
                # for img, label in zip(local_batch, local_labels):
                #     img = tio.ScalarImage(tensor=img)
                #     label = tio.LabelMap(tensor=label)
                #     subject = tio.Subject(img=img, label=label)
                #     subjects.append(subject)
                #     transform = tio.RandomFlip(axes=('LR',), flip_probability=1)
                #     transformed_subjects = transform(subject)
                #     aug_subjects.append(transformed_subjects)
                #     inverse_transform_functions.append(transformed_subjects.get_inverse_transform())
                #
                # # convert subjects to tensors
                # local_batch, local_labels = subjects_to_tensors(subjects)
                # aug_batch, aug_labels = subjects_to_tensors(aug_subjects)
                # local_batch = local_batch.float()
                # local_labels = local_labels.float()
                # aug_batch = aug_batch.float()
                ########################################################################################

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

                # Clear gradients
                self.optimizer1.zero_grad()
                # try:
                with autocast(enabled=self.with_apex):
                    # Get the classification response map(normalized) and respective class assignments after argmax
                    model_output = self.UNet1(local_batch)
                    model_output = torch.sigmoid(model_output)
                    # model_output_aug = self.UNet2(local_batch_aug)

                    # calculate dice score
                    dice_score = self.dice_score(model_output, local_labels)
                    # calculate Ft Loss
                    ft_loss = self.focal_tversky_loss(model_output, local_labels)

                    mean_loss = ft_loss.mean()
                    # mean_loss = ((1 - dice_score) * ft_loss).mean()
                    mean_dice_score = dice_score.mean()

                    model_output_aug = self.UNet1(aug_batch)
                    model_output_aug = torch.sigmoid(model_output_aug)
                    ########################################################################################
                    # # apply inverse transform ; tensors -> subjects -> inversefunction(subjects) -> tensors
                    #
                    # # convert tensors to subjects
                    # model_output_aug_subjects = tensors_to_subjects(model_output_aug.cpu())
                    #
                    # assert len(model_output_aug_subjects) == len(inverse_transform_functions), "pred and inversefunction dont match"
                    #
                    # #apply inverse function to subjects
                    # pred_subjects = []
                    # for pred_subject, inverse_function in zip(model_output_aug_subjects,inverse_transform_functions):
                    #     pred_subjects.append(inverse_function(pred_subject))
                    #
                    # # convert subjects to tensors
                    # model_output_aug = subjects_to_tensors(pred_subject)
                    # model_output_aug = model_output_aug
                    ##########################################################################################

                    # calculate dice score
                    dice_score_aug = self.dice_score(model_output_aug, aug_labels)
                    # calculate Ft Loss
                    ft_loss_aug = self.focal_tversky_loss(model_output_aug, local_labels)

                    mean_loss_aug =  ft_loss_aug.mean()
                    # mean_loss_aug = ((1 - dice_score_aug) * ft_loss_aug).mean()
                    mean_dice_score_aug = dice_score_aug.mean()

                    total_mean_loss = mean_loss_aug + mean_loss
                    total_mean_dice_score = (mean_dice_score_aug + mean_dice_score) / 2

                self.logger.info(f"Epoch: {str(epoch)} Batch Index: {str(batch_index)} Training.. "
                                 f"\n mean_loss {str(mean_loss)} mean_loss_aug {str(mean_loss_aug)} "
                                 f"\n mean_dice_score {str(mean_dice_score)} mean_dice_score_aug {str(mean_dice_score_aug)} "
                                 f"\n total_mean_loss {str(total_mean_loss)} total_dice_score {str(total_mean_dice_score)}")
                # Calculating gradients for UNet1
                if self.with_apex:
                    self.scaler.scale(total_mean_loss).backward()

                    if self.clip_grads:
                        self.scaler.unscale_(self.optimizer1)
                        torch.nn.utils.clip_grad_norm_(self.UNet1.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.scaler.step(self.optimizer1)
                    self.scaler.update()
                else:
                    if not torch.any(torch.isnan(total_mean_loss)):
                        total_mean_loss.backward()
                    else:
                        self.logger.info("nan found in floss.... no backpropagation!!")
                    if self.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.UNet1.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.optimizer1.step()

                training_batch_index += 1

                # Initialising the average loss metrics
                total_loss += total_mean_loss.detach().item()
                total_dice_score += total_mean_dice_score.detach().item()

                # To avoid memory errors
                torch.cuda.empty_cache()

            # Calculate the average loss per batch in one epoch
            total_loss /= (batch_index + 1.0)
            total_dice_score /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n loss: " + str(total_loss) + " dice_score: " +
                             str(total_dice_score))

            write_epoch_summary(writer=self.writer_training, index=epoch,
                                summary_dict={"Loss": total_loss,
                                              "DiceScore": total_dice_score
                                              })

            if self.wandb is not None:
                self.wandb.log({"Loss": total_loss,
                                "DiceScore": total_dice_score})

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

        total_loss = 0
        total_dice_score = 0
        no_patches = 0

        for batch_index, patches_batch in enumerate(tqdm(self.validate_loader)):
            self.logger.info("loading" + str(batch_index))
            no_patches += 1
            local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
            local_labels = patches_batch['label'][tio.DATA].float().cuda()
            aug_batch = Pipeline.normaliser(patches_batch['aug_img'][tio.DATA].float().cuda())
            aug_labels = patches_batch['aug_label'][tio.DATA].float().cuda()

            # Transfer to GPU
            self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

            # try:
            with autocast(enabled=self.with_apex):
                # Get the classification response map(normalized) and respective class assignments after argmax
                model_output = self.UNet1(local_batch)
                model_output = torch.sigmoid(model_output)
                # calculate dice score
                dice_score = self.dice_score(model_output, local_labels)
                # calculate Ft Loss
                ft_loss = self.focal_tversky_loss(model_output, local_labels)

                mean_loss = ft_loss.mean()
                mean_dice_score = dice_score.mean()

                model_output_aug = self.UNet1(aug_batch)
                model_output_aug = torch.sigmoid(model_output_aug)

                # calculate dice score
                dice_score_aug = self.dice_score(model_output_aug, aug_labels)
                # calculate Ft Loss
                ft_loss_aug = self.focal_tversky_loss(model_output_aug, local_labels)

                mean_loss_aug = ft_loss_aug.mean()
                # mean_loss_aug = ((1 - dice_score_aug) * ft_loss_aug).mean()
                mean_dice_score_aug = dice_score_aug.mean()

                total_mean_loss = mean_loss_aug + mean_loss
                total_mean_dice_score = (mean_dice_score_aug + mean_dice_score) / 2
                # Log validation losses

                self.logger.info(f" Batch Index: {str(batch_index)} Validating.. "
                                 f"\n val_mean_loss {str(mean_loss)} val_mean_loss_aug {str(mean_loss_aug)} "
                                 f"\n val_mean_dice_score {str(mean_dice_score)} val_mean_dice_score_aug {str(mean_dice_score_aug)} "
                                 f"\n val_total_mean_loss {str(total_mean_loss)} val_total_dice_score {str(total_mean_dice_score)}")

                total_loss += total_mean_loss.detach().cpu()
                total_dice_score += total_mean_dice_score.detach().cpu()

        # Average the losses
        total_loss = total_loss / no_patches
        total_dice_score = total_dice_score / no_patches

        process = ' Validating'
        self.logger.info("Epoch:" + str(training_index) + process + "..." +
                         "\n loss: " + str(total_loss) + " dice_score: " +
                         str(total_dice_score))

        write_epoch_summary(writer=self.writer_training, index=epoch,
                            summary_dict={"loss_val": total_loss,
                                          "dice_score_val": total_dice_score
                                          })

        if self.wandb is not None:
            self.wandb.log({"loss_val": total_loss,
                            "total_dice_score_val": total_dice_score})

        if self.LOWEST_LOSS > (1 - total_dice_score):  # Save best metric evaluation weights
            self.LOWEST_LOSS = (1 - total_dice_score)
            self.logger.info(
                'Best metric... @ epoch:' + str(epoch) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            save_model(self.CHECKPOINT_PATH, {
                'epoch_type': 'best',
                'epoch': epoch,
                'state_dict': [self.UNet1.state_dict()],
                'optimizer': [self.optimizer1.state_dict()],
                'amp': self.scaler.state_dict()})

    def test(self, test_logger, test_subjects=None, save_results=True):
        test_logger.debug('Testing...')

        if test_subjects is None:
            test_folder_path = self.DATASET_PATH + '/test/'
            test_label_path = self.DATASET_PATH + '/test_label/'

            test_subjects = Pipeline.create_tio_sub_ds(logger=test_logger, vol_path=test_folder_path,
                                                       label_path=test_label_path,
                                                       get_subjects_only=True)

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))

        df = pd.DataFrame(columns=["Subject", "Dice", "IoU"])
        result_root = os.path.join(self.OUTPUT_PATH, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)

        self.UNet1.eval()

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
                    local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    locations = patches_batch[tio.LOCATION]

                    with autocast(enabled=self.with_apex):
                        output = self.UNet1(local_batch)
                        if type(output) is tuple or type(output) is list:
                            output = output[0]
                        output = torch.sigmoid(output).detach().cpu()
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
                    iou3D = iou(result, label)
                    datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3D], "IoU": [iou3D]})
                    df = pd.concat([df, datum], ignore_index=True)

                if save_results:
                    save_nifti(result, os.path.join(result_root, subjectname + ".nii.gz"))

                    resultMIP = np.max(result, axis=-1)
                    Image.fromarray((resultMIP * 255).astype('uint8'), 'L').save(
                        os.path.join(result_root, subjectname + "_MIP.tif"))

                    if label is not None:
                        overlay = create_diff_mask_binary(result, label)
                        save_tif_rgb(overlay, os.path.join(result_root, subjectname + "_colour.tif"))

                        overlayMIP = create_diff_mask_binary(resultMIP, np.max(label, axis=-1))
                        Image.fromarray(overlayMIP.astype('uint8'), 'RGB').save(
                            os.path.join(result_root, subjectname + "_colourMIP.tif"))

                test_logger.info("Testing " + subjectname + "..." +
                                 "\n Dice:" + str(dice3D) +
                                 "\n JacardIndex:" + str(iou3D))

            df.to_excel(os.path.join(result_root, "Results_Main.xlsx"))

    def predict(self, image_path, label_path, predict_logger):
        predict_logger.debug('Predicting... TODO')
