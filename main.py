#!/usr/bin/env python
"""

"""

import argparse
import random

import numpy as np
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from models.unet3d import U_Net
from pipeline import Pipeline
from utils.logger import Logger

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name",
                        default="Model_v1",
                        help="Name of the model")
    parser.add_argument("-dataset_path",
                        default="",
                        help="Path to folder containing dataset.")
    parser.add_argument("-output_path",
                        default="",
                        help="Folder path to store output "
                             "Example: /home/output/")
    parser.add_argument('-train',
                        default=True,
                        help="To train the model")
    parser.add_argument('-test',
                        default=False,
                        help="To test the model")
    parser.add_argument('-predict',
                        default=False,
                        help="To predict a segmentation output of the model and to get a diff between label and output")
    parser.add_argument('-predictor_path',
                        default="",
                        help="Path to the input image to predict an output, ex:/home/test/ww25.nii ")
    parser.add_argument('-predictor_label_path',
                        default="",
                        help="Path to the label image to find the diff between label an output"
                             ", ex:/home/test/ww25_label.nii ")

    parser.add_argument('-load_path',
                        default="",
                        help="Path to checkpoint of existing model to load, ex:/home/model/checkpoint")
    parser.add_argument('-load_best',
                        default=True,
                        help="Specifiy whether to load the best checkpoiont or the last. "
                             "Also to be used if Train and Test both are true.")
    parser.add_argument('-clip_grads',
                        default=True,
                        action="store_true",
                        help="To use deformation for training")
    parser.add_argument('-apex',
                        default=True,
                        help="To use half precision on model weights.")
    parser.add_argument("-batch_size",
                        type=int,
                        default=15,
                        help="Batch size for training")
    parser.add_argument("-num_epochs",
                        type=int,
                        default=50,
                        help="Number of epochs for training")
    parser.add_argument("-learning_rate",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument("-patch_size",
                        type=int,
                        default=32,
                        help="Patch size of the input volume")
    parser.add_argument("-stride_depth",
                        type=int,
                        default=16,
                        help="Strides for dividing the input volume into patches in "
                             "depth dimension (To be used during validation and inference)")
    parser.add_argument("-stride_width",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in "
                             "width dimension (To be used during validation and inference)")
    parser.add_argument("-stride_length",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in "
                             "length dimension (To be used during validation and inference)")
    parser.add_argument("-samples_per_epoch",
                        type=int,
                        default=8000,
                        help="Number of samples per epoch")
    parser.add_argument("-num_worker",
                        type=int,
                        default=8,
                        help="Number of worker threads")
    parser.add_argument("-segcor_weight1",
                        type=float,
                        default=1.0,
                        help="Number of worker threads")
    parser.add_argument("-segcor_weight2",
                        type=float,
                        default=10.0,
                        help="Number of worker threads")
    parser.add_argument("-wandb",
                        default=True,
                        help="Set this to true to include wandb logging")

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    DATASET_FOLDER = args.dataset_path
    OUTPUT_PATH = args.output_path

    LOAD_PATH = args.load_path
    CHECKPOINT_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '/checkpoint/'
    TENSORBOARD_PATH_TRAINING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_training/'
    TENSORBOARD_PATH_VALIDATION = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_validation/'
    TENSORBOARD_PATH_TESTING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_testing/'

    LOGGER_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '.log'

    logger = Logger(MODEL_NAME, LOGGER_PATH).get_logger()
    test_logger = Logger(MODEL_NAME + '_test', LOGGER_PATH).get_logger()
    wandb = None
    if str(args.wandb).lower() == "true":
        import wandb
        wandb.init(project="AIDE_StudyForrest", entity="ds6_vessel_seg2", notes=args.model_name)
        wandb.config = {
            "learning_rate": args.learning_rate,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "segcor_weight1": args.segcor_weight1,
            "segcor_weight2": args.segcor_weight2
        }

    # models
    # model = torch.nn.DataParallel(DFC3D(num_conv=args.num_conv, num_classes=args.num_classes))
    UNet1 = torch.nn.DataParallel(U_Net())
    UNet2 = torch.nn.DataParallel(U_Net())
    UNet1.cuda()
    UNet2.cuda()

    writer_training = SummaryWriter(TENSORBOARD_PATH_TRAINING)
    writer_validating = SummaryWriter(TENSORBOARD_PATH_VALIDATION)

    pipeline = Pipeline(cmd_args=args, UNet1=UNet1, UNet2=UNet2, logger=logger,
                        dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH,
                        writer_training=writer_training, writer_validating=writer_validating, wandb=wandb)

    # loading existing checkpoint if supplied
    if bool(LOAD_PATH):
        pipeline.load(checkpoint_path=LOAD_PATH, load_best=args.load_best)

    try:

        if args.train:
            pipeline.train()
            torch.cuda.empty_cache()  # to avoid memory errors

        if args.test:
            if args.load_best:
                pipeline.load(load_best=True)
            pipeline.test(test_logger=test_logger)
            torch.cuda.empty_cache()  # to avoid memory errors

        if args.predict:
            pipeline.predict(predict_logger=test_logger, image_path=args.predictor_path,
                             label_path=args.predictor_label_path)

    except Exception as error:
        print(error)
        logger.exception(error)

    writer_training.close()
    writer_validating.close()
