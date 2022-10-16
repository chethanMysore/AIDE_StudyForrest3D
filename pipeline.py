# !/usr/bin/env python
"""

"""

import torch
import torch.utils.data
import torchio as tio
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.transforms import functional as F
import random
import os
import numpy as np
from glob import glob
import torchvision as tv
import pandas as pd
from skimage.filters import threshold_otsu
import random
from evaluation.metrics import (SegmentationLoss, ConsistencyLoss, FocalTverskyLoss, DiceScore)
from utils.customutils import subjects_to_tensors, tensors_to_subjects
from utils.datasets import SRDataset
from utils.results_analyser import *
from utils.transformations_utils import RandomAffineTransformation, RandomRotateTransformation, \
    RandomHorizontalFlipTransformation
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
            validation_set = Pipeline.create_tio_sub_ds(logger=self.logger,
                                                        vol_path=self.DATASET_PATH + '/validate/',
                                                        label_path=self.DATASET_PATH + '/validate_label/',
                                                        patch_size=self.patch_size,
                                                        samples_per_epoch=self.samples_per_epoch,
                                                        num_worker=self.num_worker,
                                                        stride_depth=self.stride_depth,
                                                        stride_length=self.stride_length,
                                                        stride_width=self.stride_width,
                                                        is_train=False
                                                        )
            self.validate_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size,
                                                               shuffle=False, num_workers=0)

    @staticmethod
    def get_transformations(img):
        def get_max_displacement(img, control_points):
            image = img.as_sitk()
            bounds = np.array(image.GetSize()) * np.array(image.GetSpacing())
            num_control_points = np.array(control_points)
            grid_spacing = bounds / (num_control_points - 2)
            potential_folding = grid_spacing / 2

            min_displacement = [30., 40., 10.18]  # min displacement when num_of_control_points = 10
            displacement_points = ()

            for min_displacement, max_displacement in zip(min_displacement, potential_folding):
                pt = round(random.uniform(min_displacement, max_displacement), 2)
                displacement_points += (pt,)
            return displacement_points

        num_of_control_points = tuple(random.randint(5, 10) for _ in range(3))

        return tio.Compose([
            tio.RandomFlip(axes=[*set(random.choices(['LR', 'AP', 'IS'], k=2))],
                           flip_probability=0.75,
                           exclude=["img", "label"]),
            tio.RandomElasticDeformation(num_control_points=num_of_control_points, locked_borders=2,
                                         max_displacement=get_max_displacement(img, num_of_control_points),
                                         exclude=["img", "label"])
        ])

    @staticmethod
    def create_tio_sub_ds(logger, vol_path, label_path, stride_width=None, stride_length=None, stride_depth=None,
                          num_worker=0, patch_size=None, samples_per_epoch=None,
                          get_subjects_only=False, is_train=True):

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
                subjectname=filename
            )

            # if is_train:
            #     transforms = Pipeline.get_transformations(t1)
            #     subject = transforms(subject)

            subjects.append(subject)

        if get_subjects_only:
            return subjects

        if is_train:
            logger.info("creating training patch..")
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
            logger.info("creating validation patch..")
            overlap = np.subtract(patch_size, (stride_length, stride_width, stride_depth))
            grid_samplers = []
            for i in range(len(subjects)):
                grid_sampler = tio.inference.GridSampler(
                    subjects[i],
                    patch_size,
                    overlap,
                )
                grid_samplers.append(grid_sampler)
            return torch.utils.data.ConcatDataset(grid_samplers)

    @staticmethod
    def normaliser(batch):
        for i in range(batch.shape[0]):
            if batch[i].max() > 0.0:
                batch[i] = batch[i] / batch[i].max()
        return batch

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

    def apply_reverse_transformation(self, tensor_list, transformation_instances, epoch, batch_index):
        # if no transformation found
        if not any(transformation_instances):
            print(f"Epoch {epoch} Batch {batch_index} No inverse transformation found on the whole batch")
            return tensor_list
        else:
            transformed_imgs = []
            for indx, (img, transformations) in enumerate(zip(tensor_list, transformation_instances)):
                if transformations:
                    inverse_transforms = [t.get_inverse_transform() for t in transformations]
                    inverse_transforms.reverse()
                    # print(
                    #     f"Epoch {epoch} Batch {batch_index} indx {indx} Applying {len(inverse_transforms)} inverse transformation: "
                    #     f"{str(inverse_transforms)}")
                    transform = T.Compose(inverse_transforms)
                    transformed_imgs.append(transform(img))
                else:
                    # print(f"Epoch {epoch} Batch {batch_index} indx {indx} No inverse transformation found")
                    transformed_imgs.append(img)
            return torch.stack(transformed_imgs, dim=0)

    def apply_transformation(self, batch, label, epoch, batch_index):
        transformed_labels = []
        transformed_imgs = []
        transformation_instances = []

        for indx, (img, label) in enumerate(zip(batch, label)):
            transformations = []
            applied_transformation_instance = []
            if random.random() < 0.5:
                random_rotate = RandomRotateTransformation()
                transformations.append(random_rotate.get_transform())
                applied_transformation_instance.append(random_rotate)
            if random.random() < 0.5:
                random_flip = RandomHorizontalFlipTransformation()
                transformations.append(random_flip.get_transform())
                applied_transformation_instance.append(random_flip)
            if transformations:
                # print(f"Epoch {epoch} Batch {batch_index} img_index {indx} "
                #       f"Applied {len(transformations)} transformations {str(transformations)} ")
                transform = T.Compose(transformations)
                transformed_imgs.append(transform(img))
                transformed_labels.append(transform(label))
            else:
                # print(f"Epoch {epoch} Batch {batch_index} img_index {indx} : No transformation applied to batch")
                transformed_imgs.append(img)
                transformed_labels.append(label)

            transformation_instances.append(applied_transformation_instance)

        aug_batch = torch.stack(transformed_imgs, dim=0)
        aug_labels = torch.stack(transformed_labels, dim=0)

        return aug_batch, aug_labels, transformation_instances

    def train(self):
        self.logger.debug("Training...")
        rate_schedule = np.ones(self.num_epochs)

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.UNet1.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            self.UNet2.train()
            total_loss_1 = 0
            total_loss_2 = 0
            total_dice_score_1 = 0
            total_dice_score_2 = 0
            batch_index = 0
            rate_schedule[epoch] = min((float(epoch) / 10.0) ** 2, 1.0)

            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):
                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float())
                local_labels = patches_batch['label'][tio.DATA].float()

                aug_batch, aug_labels, transformation_instances = self.apply_transformation(local_batch, local_labels,
                                                                                            epoch, batch_index)

                aug_batch = aug_batch.cuda()
                aug_labels = aug_labels.cuda()
                local_batch = local_batch.cuda()
                local_labels = local_labels.cuda()

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

                # Clear gradients
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()

                # try:
                with autocast(enabled=self.with_apex):
                    # Get the classification response map(normalized) and respective class assignments after argmax

                    model_output_aug_1 = self.UNet1(aug_batch).detach()
                    model_output_aug_1 = torch.sigmoid(model_output_aug_1)
                    model_output_revaug_1 = self.apply_reverse_transformation(model_output_aug_1,
                                                                              transformation_instances,epoch, batch_index)

                    model_output_aug_2 = self.UNet2(aug_batch).detach()
                    model_output_aug_2 = torch.sigmoid(model_output_aug_2)
                    model_output_revaug_2 = self.apply_reverse_transformation(model_output_aug_2,
                                                                              transformation_instances,epoch, batch_index)

                    print(model_output_aug_1.size())
                    print(model_output_aug_2.size())
                    print(model_output_revaug_1.size())
                    print(model_output_revaug_2.size())

                    weight_map_1 = 1.0 - 4.0 * model_output_revaug_1[:, 0, :, :, :]
                    weight_map_1 = weight_map_1.unsqueeze(dim=1)
                    weight_map_2 = 1.0 - 4.0 * model_output_revaug_2[:, 0, :, :, :]
                    weight_map_2 = weight_map_2.unsqueeze(dim=1)

                    model_output_1 = self.UNet1(local_batch).cuda()
                    model_output_1 = torch.sigmoid(model_output_1)
                    model_output_2 = self.UNet2(local_batch).cuda()
                    model_output_2 = torch.sigmoid(model_output_2)

                    print(weight_map_1.size())
                    print(model_output_1.size())
                    print(model_output_2.size())

                    dice_score_1 = self.dice_score(model_output_1, local_labels).mean()
                    dice_score_2 = self.dice_score(model_output_2, local_labels).mean()

                    # calculate Ft Loss
                    ft_loss_1 = self.focal_tversky_loss(model_output_1, local_labels)
                    ft_loss_2 = self.focal_tversky_loss(model_output_2, local_labels)

                    _, indx1 = ft_loss_1.sort()
                    _, indx2 = ft_loss_2.sort()

                    loss1_seg1 = self.focal_tversky_loss(model_output_1[indx2[0:7], :, :, :, :],
                                                         local_labels[indx2[0:7], :, :, :, :]).mean()
                    loss2_seg1 = self.focal_tversky_loss(model_output_2[indx1[0:7], :, :, :, :],
                                                         local_labels[indx1[0:7], :, :, :, :]).mean()
                    loss1_seg2 = self.focal_tversky_loss(model_output_2[indx2[7:], :, :, :, :],
                                                         local_labels[indx2[7:], :, :, :, :]).mean()
                    loss2_seg2 = self.focal_tversky_loss(model_output_2[indx1[7:], :, :, :, :],
                                                         local_labels[indx1[7:], :, :, :, :]).mean()

                    loss1_cor = weight_map_2[indx2[2:], :, :, :, :] * \
                                self.consistency_loss(model_output_1[indx2[7:], :, :, :, :],
                                                      model_output_revaug_2[indx2[7:], :, :, :, :])

                    loss1_cor = loss1_cor.mean()

                    loss1 = 1.0 * (loss1_seg1 + (1.0 - rate_schedule[epoch]) * loss1_seg2) + 1.0 * rate_schedule[
                        epoch] * loss1_cor

                    loss2_cor = weight_map_1[indx1[2:], :, :, :, :] * \
                                self.consistency_loss(model_output_2[indx1[7:], :, :, :, :],
                                                      model_output_revaug_1[indx1[7:], :, :, :, :])

                    loss2_cor = loss2_cor.mean()

                    loss2 = 1.0 * (loss2_seg1 + (1.0 - rate_schedule[epoch]) * loss2_seg2) + 1.0 * rate_schedule[
                        epoch] * loss2_cor

                    loss1.backward(retain_graph=True)
                    self.optimizer1.step()
                    loss2.backward()
                    self.optimizer2.step()

                self.logger.info(f"Epoch: {str(epoch)} Batch Index: {str(batch_index)} Training.. "
                                 f"\n model1_loss {str(loss1)} model2_loss {str(loss2)} "
                                 f"\n model1_dice_score {str(dice_score_1)} mean_dice_score_aug {str(dice_score_2)} "
                                 )

                # # Calculating gradients for UNet1
                # if self.with_apex:
                #     self.scaler.scale(total_mean_loss).backward()
                #
                #     if self.clip_grads:
                #         self.scaler.unscale_(self.optimizer1)
                #         torch.nn.utils.clip_grad_norm_(self.UNet1.parameters(), 1)
                #         # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
                #
                #     self.scaler.step(self.optimizer1)
                #     self.scaler.update()
                # else:
                #     if not torch.any(torch.isnan(total_mean_loss)):
                #         total_mean_loss.backward()
                #     else:
                #         self.logger.info("nan found in floss.... no backpropagation!!")
                #     if self.clip_grads:
                #         torch.nn.utils.clip_grad_norm_(self.UNet1.parameters(), 1)
                #         # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
                #
                #     self.optimizer1.step()

                training_batch_index += 1

                # Initialising the average loss metrics
                total_loss_1 += loss1.detach().item()
                total_loss_2 += loss2.detach().item()
                total_dice_score_1 += dice_score_1.detach().item()
                total_dice_score_2 += dice_score_2.detach().item()

                # To avoid memory errors
                torch.cuda.empty_cache()
                # break;

            # Calculate the average loss per batch in one epoch
            total_loss_1 /= (batch_index + 1.0)
            total_loss_2 /= (batch_index + 1.0)
            total_dice_score_1 /= (batch_index + 1.0)
            total_dice_score_2 /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n loss_1: " + str(total_loss_1) + " dice_score_1: " + str(total_dice_score_1) +
                             "\n loss_2: " + str(total_loss_2) + " dice_score_2: " + str(total_dice_score_2))

            write_epoch_summary(writer=self.writer_training, index=epoch,
                                summary_dict={"Loss_1": total_loss_1,
                                              "Loss_2": total_loss_2,
                                              "DiceScore_1": total_dice_score_1,
                                              "DiceScore_2": total_dice_score_2,
                                              })

            if self.wandb is not None:
                self.wandb.log({"Loss_1": total_loss_1,
                                "Loss_2": total_loss_2,
                                "DiceScore_1": total_dice_score_1,
                                "DiceScore_2": total_dice_score_2,
                                })

            torch.cuda.empty_cache()  # to avoid memory errors
            # self.validate(training_batch_index, epoch)
            torch.cuda.empty_cache()  # to avoid memory errors
            # break

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
        random_rotate = RandomRotateTransformation()
        random_affine = RandomAffineTransformation()
        total_loss = 0
        total_dice_score = 0
        no_patches = 0
        for batch_index, patches_batch in enumerate(tqdm(self.validate_loader)):
            self.logger.info("loading" + str(batch_index))
            no_patches += 1
            local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float())
            local_labels = patches_batch['label'][tio.DATA].float()
            aug_batch, aug_labels, transformation_instances = self.apply_transformation(local_batch, local_labels,
                                                                                        epoch, batch_index)

            aug_batch = aug_batch.cuda()
            aug_labels = aug_labels.cuda()
            local_batch = local_batch.cuda()
            local_labels = local_labels.cuda()

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
                # mean_loss = ((1 - dice_score) * ft_loss).mean()
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
