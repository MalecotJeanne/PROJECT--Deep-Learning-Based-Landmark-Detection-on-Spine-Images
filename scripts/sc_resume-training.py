"""
Script to train a model from scratch
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.distributed
import yaml
from loguru import logger
from monai.data import Dataset

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from models import init_model
from training import train_model
from transforms import training_transforms
from utils import load_config, load_data, save_dataset
from monai.transforms import Compose, LoadImaged
from monai.data import PILReader


# Parse the arguments
parser = argparse.ArgumentParser(
    description="resume training spine landmark detection"
)

parser.add_argument(
    "-d", "--data_dir", default="boostnet_labeldata/", type=str, help="data path"
)
parser.add_argument(
    "-c", "--chkpt_dir", type = str, help="path to the checkpoint file"
)
parser.add_argument("--gpu_devices", default="1,2", type=str, help="gpu devices")

args = parser.parse_args()


def main():
    """
    Main function to resume training from checkpoint
    """

    chkpt_dir = args.chkpt_dir
    results_dir = os.path.dirname(chkpt_dir)

    model_name = os.path.basename(results_dir)
    model_name = model_name.split("_")[0]
    
    log_file = [logs_filepath for logs_filepath in os.listdir(results_dir) if logs_filepath.endswith(".log")][0]
    logs_dir = os.path.join(results_dir, log_file)

    logger.add(logs_dir, mode="a")
     
    config_dir = os.path.join(results_dir, "config.yaml")

    root_dir = args.data_dir
    path_data = os.path.join(root_dir, "data")
    path_labels = os.path.join(root_dir, "labels")

    # Check if the folders exist
    if not os.path.exists(path_data):
        logger.error(f"Data folder does not exist: {path_data}")
        return
    if not os.path.exists(path_labels):
        logger.error(f"Labels folder does not exist: {path_labels}")
        return
    if not os.path.exists(config_dir):
        logger.error(f"Config file does not exist: {config_dir}")
        return

    # Load the config file
    logger.info("Loading config file...")
    config = load_config(config_dir)
    logger.success("Config file {} loaded successfully".format(config_dir))

    logger.info("Loading data...")
    path_data = os.path.join(path_data, "training")
    path_labels = os.path.join(path_labels, "training")

    data_dict = load_data(path_data, path_labels)

    logger.success("Data loaded successfully!")

    # Create dataset and dataloader
    transforms_dict = config["transforms"]
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=True, reader=PILReader(reverse_indexing=False)),
        training_transforms(transforms_dict)
    ])
    dataset = Dataset(data=data_dict, transform=transforms)

    # Load the model
    model = init_model(model_name, config["model"])

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device} for training")
    if device.type == "cuda":
        logger.info(f"Using GPU device(s): {args.gpu_devices}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    train_model(
        dataset, model, chkpt_dir, results_dir, config, device, logs_dir
    )


if __name__ == "__main__":
    main()
