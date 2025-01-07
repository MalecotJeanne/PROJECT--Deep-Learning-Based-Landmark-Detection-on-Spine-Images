"""
Main script to train or test the model
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.distributed
from loguru import logger
from monai.data import Dataset

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from models import init_model
from inference import test_model
from transforms import testing_transforms
from utils import get_last_folder, load_config, load_data, save_dataset
from monai.transforms import Compose, LoadImaged
from monai.data import PILReader


# Parse the arguments
parser = argparse.ArgumentParser(
    description="Inferring spine landmark detection"
)

parser.add_argument(
    "-d", "--data_dir", default="boostnet_labeldata/", type=str, help="data path"
)
parser.add_argument(
    "-m", "--model_dir", default="None", type=str, help="path to the model checkpoint"
)
parser.add_argument("--model", choices=["hrnet", "unet"], help="model to use")
parser.add_argument("--gpu_devices", default="1,2", type=str, help="gpu devices")

args = parser.parse_args()


def main():
    """
    Main function to infer a model
    """
    train_dir = args.model_dir
    model_dir = os.path.join(train_dir, "checkpoints")
    if train_dir == "None":    
        results_dir = os.path.join(os.getcwd(), "Results")
        model_train_dir = get_last_folder(results_dir, args.model)
        train_dir = os.path.join(results_dir, model_train_dir)
        model_dir = os.path.join(train_dir, "checkpoints")

    train_results_dir = model_dir.replace("checkpoints", "results")
    results_dir = os.path.join(train_results_dir, datetime.now().strftime('%Y-%m-%d_%H-%M'))

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
        log_file.write(f"Inferring {args.model}\n")
        log_file.write(f"--- Checkpoint: {model_dir}\n")
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

    # Load the config file
    logger.info("Loading config file...")
    config_dir = os.path.join(train_dir, "config.yaml")
    config = load_config(config_dir)
    logger.success("Config file {} loaded successfully".format(config_dir))

    logger.info("Loading data...")
    path_data = os.path.join(path_data, "test")
    path_labels = os.path.join(path_labels, "test")
    data_dict = load_data(path_data, path_labels)
    logger.success("Data loaded successfully!")

    # Create dataset and dataloader
    transforms_dict = config["transforms"]
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=True, reader=PILReader(reverse_indexing=False)),
        testing_transforms(transforms_dict)
    ])
    dataset = Dataset(data=data_dict, transform=transforms)

    # save the dataset images
    dataset_dir = os.path.join(results_dir, "dataset_images")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    logger.info("Saving dataset images...")    
    save_dataset(dataset, dataset_dir, "transformed")
    logger.success("Dataset images saved successfully!")

    # Load the model
    model = init_model(args.model, config["model"])

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    model.to(device)
    logger.info(f"Using device: {device} for testing")
    if device.type == "cuda":
        logger.info(f"Using GPU device(s): {args.gpu_devices}")

    # Test the model
    
    test_model(
        dataset, model, model_dir, results_dir, config, device, logs_filepath
    )


if __name__ == "__main__":
    main()
