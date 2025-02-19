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
    description="training spine landmark detection"
)

parser.add_argument(
    "-d", "--data_dir", default="boostnet_labeldata/", type=str, help="data path"
)
parser.add_argument(
    "-o", "--output_dir", default="Results", type=str, help="path to save the results"
)
parser.add_argument(
    "-config", "--config_dir", default="config.yaml", type=str, help="path to the config file"
)
parser.add_argument("-m", "--model", choices=["hrnet", "unet", "unet_base", "dynunet"], help="model to use")
parser.add_argument("--save_dataset", default=False, help="save the dataset images")
parser.add_argument("--gpu_devices", default="1,2", type=str, help="gpu devices")

args = parser.parse_args()


def main():
    """
    Main function to train a model from scratch
    """

    results_dir = args.output_dir
    execution_name = (
        f"{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    results_dir = os.path.join(results_dir, execution_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    log_filename = f"{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
    logs_filepath = os.path.join(results_dir, log_filename)

    # logger header
    with open(logs_filepath, "w") as log_file:
        log_file.write("==========================================\n")
        log_file.write("Logs for spine landmark detection\n")
        log_file.write("--- Author: Jeanne Malecot         \n")
        log_file.write(f"--- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        log_file.write("----------\n")
        log_file.write(f"Training {args.model} from scratch\n")
        log_file.write("==========================================\n")
        log_file.write("\n")

    logger.add(logs_filepath, mode="a")

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
    if not os.path.exists(args.config_dir):
        logger.error(f"Config file does not exist: {args.config_dir}")
        return

    # Load the config file
    logger.info("Loading config file...")
    config = load_config(args.config_dir)
    logger.success("Config file {} loaded successfully".format(args.config_dir))

    # save the config file
    config_dir = os.path.join(results_dir, "config.yaml")
    with open(config_dir, "w") as file:
        yaml.dump(config, file)

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

    if args.save_dataset:
        # save the dataset images
        dataset_dir = os.path.join(results_dir, "dataset_images")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        logger.info("Saving dataset images...")    
        save_dataset(dataset, dataset_dir, "transformed")
        logger.success("Dataset images saved successfully!")

    # Load the model
    model = init_model(args.model, config["model"][args.model])

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device} for training")
    if device.type == "cuda":
        logger.info(f"Using GPU device(s): {args.gpu_devices}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    # Train the model
    chkpt_dir = os.path.join(results_dir, "checkpoints")
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    train_model(
        dataset, model, chkpt_dir, results_dir, config, device, logs_filepath
    )


if __name__ == "__main__":
    main()
