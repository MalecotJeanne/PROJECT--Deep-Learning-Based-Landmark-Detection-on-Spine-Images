"""
Docstring
"""

import argparse
import os
from datetime import datetime

import torch
from loguru import logger
from monai.data import Dataset, CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd

from models import init_model
from transforms import ResizeWithLandmarksd
from train import train_model
from utils import load_config, load_data

# from test import test_model


# Parse the arguments
parser = argparse.ArgumentParser(
    description="methods review for spine landmark detection"
)

parser.add_argument(
    "--root_dir", default="boostnet_labeldata/", type=str, help="data path"
)
parser.add_argument(
    "--save_dir", default="results", type=str, help="path to save the results"
)
parser.add_argument(
    "--logs_dir", default="logs", type=str, help="path to save the logs"
)
parser.add_argument(
    "--chkpt_dir", default=None, type=str, help="path of the checkpoint to use"
)
parser.add_argument(
    "--config_dir", default="config.yaml", type=str, help="path to the config file"
)
parser.add_argument("--model", choices=["hrnet", "resnet"], help="model to use")
parser.add_argument("--phase", choices=["train", "test"], help="train or test")
parser.add_argument("--gpu_devices", default="1,2", type=str, help="gpu devices")

args = parser.parse_args()


def main():
    """
    TODO: Docstring
    """
    # check if the logs directory exists
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    log_filename = f"{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
    logger.add(os.path.join(args.logs_dir, log_filename))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    logger.info(f"Using GPU device(s): {args.gpu_devices}")

    root_dir = args.root_dir
    path_data = os.path.join(root_dir, "data")
    path_labels = os.path.join(root_dir, "labels")

    # Check if the folders exist
    if not os.path.exists(path_data):
        logger.error(f"Data folder does not exist: {path_data}")
        return
    if not os.path.exists(path_labels):
        logger.error(f"Labels folder does not exist: {path_labels}")
        return
    if not os.path.exists(args.config_dir):
        logger.error(f"Config file does not exist: {args.config_dir}")
        return

    # Load the config file
    config = load_config(args.config_dir)

    if args.phase == "train":
        path_data = os.path.join(path_data, "training")
        path_labels = os.path.join(path_labels, "training")
        data_dict = load_data(path_data, path_labels)

    elif args.phase == "test":
        path_data = os.path.join(path_data, "test")
        path_labels = os.path.join(path_labels, "test")
        data_dict = load_data(path_data, path_labels)

    else:
        logger.error("Invalid phase. Choose between 'train' or 'test'.")
        return

    # Create dataset and dataloader
    transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            ResizeWithLandmarksd(spatial_size=(512, 1024), mode = 'bilinear', keys=["image", "landmarks"]), #FIXME: find an optimal spatial size for the images, and put it in a config
        ]
    )#TODO: define the transforms in a specific file
    logger.info("Creating dataset...")
    dataset = Dataset(data=data_dict, transform=transforms)

    # Print the number of samples in the training and test sets
    logger.info(f"Number of samples in the dataset: {len(dataset)}")

    # Load the model
    model = init_model(args.model)

    if args.phase == "train":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Train the model
        chkpt_dir = args.chkpt_dir
        train_model(dataset, model, chkpt_dir, config, device, logger)

    if args.phase == "test":
        # Test the model
        # test_model(dataset)
        print("test phase = pas encore fait")


if __name__ == "__main__":
    main()
