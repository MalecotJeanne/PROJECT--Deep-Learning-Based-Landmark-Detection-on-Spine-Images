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
from training import train_model
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
    "--results_dir", default="Results", type=str, help="path to save the results"
)
parser.add_argument(
    "--logs_dir", default=None, type=str, help="path to save the logs"
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
    results_dir = args.results_dir
    execution_name = f"{args.model}_{args.phase}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    results_dir = os.path.join(results_dir, execution_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    logs_dir = args.logs_dir
    if logs_dir is None:
        logs_dir = results_dir
    elif not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    log_filename = f"{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
    logs_filepath = os.path.join(logs_dir, log_filename)

    # logger header
    with open(logs_filepath, "w") as log_file:
        log_file.write("==========================================\n")
        log_file.write("Logs for spine landmark detection\n")
        log_file.write("--- Author: Jeanne Malecot         \n")
        log_file.write(f"--- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        log_file.write("==========================================\n")
        log_file.write("\n")

    logger.add(logs_filepath, mode="a")

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
    logger.info("Loading config file...")
    config = load_config(args.config_dir)
    logger.success("Config file {} loaded successfully".format(args.config_dir))

    if args.phase == "train":
        logger.info("Loading data for training...")
        path_data = os.path.join(path_data, "training")
        path_labels = os.path.join(path_labels, "training")
        data_dict = load_data(path_data, path_labels)

    elif args.phase == "test":
        logger.info("Loading data for testing...")
        path_data = os.path.join(path_data, "test")
        path_labels = os.path.join(path_labels, "test")
        data_dict = load_data(path_data, path_labels)

    else:
        logger.error("Invalid phase. Choose between 'train' or 'test'.")
        return

    logger.success("Data loaded successfully!")

    # Create dataset and dataloader
    transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            ResizeWithLandmarksd(
                spatial_size=(1024, 512), mode="bilinear", keys=["image", "landmarks"]
            ),  # FIXME: find an optimal spatial size for the images, and put it in a config
        ]
    )  # TODO: define the transforms in a specific file
    dataset = Dataset(data=data_dict, transform=transforms)

    # Load the model
    model = init_model(args.model)

    if args.phase == "train":
        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
        logger.info(f"Using GPU device(s): {args.gpu_devices}")
        model.to(device)
        logger.info(f"Using device: {device} for training")

        # Train the model
        chkpt_dir = args.chkpt_dir
        if chkpt_dir is None:
            chkpt_dir = os.path.join(results_dir, "checkpoints")
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        train_model(
            dataset, model, chkpt_dir, results_dir, config, device, logs_filepath
        )

    if args.phase == "test":
        # Test the model
        # test_model(dataset)
        print("test phase = pas encore fait")


if __name__ == "__main__":
    main()
