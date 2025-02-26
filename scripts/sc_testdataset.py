"""
Script to test a dataset.
Load the data and labels, create the dataset and visualize the data transformed.
Author: Jeanne Malécot
"""

import argparse
import os
import sys
import torch

from monai.data import Dataset, PILReader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from utils import load_data, load_config, save_dataset
from transforms import training_transforms, testing_transforms, ResizeWithLandmarksd

parser = argparse.ArgumentParser(description="Script to test a dataset.")

parser.add_argument(
    "--data_dir", default="Ressources/boostnet_labeldata/", type=str, help="data path"
)
parser.add_argument(
    "--config_dir", default="config.yaml", type=str, help="path to the config file"
)
parser.add_argument(
    "--save_dir", default="CLAHE/2_8-8", type=str, help="path to save the images"
)
parser.add_argument(
    "--phase",
    default="train",
    choices=["train", "test"],
    help="to chose to visualize the training or the test transformations",
)
parser.add_argument(
    "--test_inverse",
    default=False,
    type=bool,
    help="to test the inverse transformations",
)

args = parser.parse_args()


def main():
    """
    Load the data and labels, create the dataset and visualize the data transformed.
    """
    save_dir = args.save_dir
    save_dir = os.path.join(save_dir, args.phase)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_data = os.path.join(args.data_dir, "data")
    path_labels = os.path.join(args.data_dir, "labels")

    if not os.path.exists(path_data):
        raise ValueError("Data folder does not exist")
    if not os.path.exists(path_labels):
        raise ValueError("Labels folder does not exist")

    if args.phase == "train":
        path_data = os.path.join(path_data, "training")
        path_labels = os.path.join(path_labels, "training")
        data_dict = load_data(path_data, path_labels)
    else:
        path_data = os.path.join(path_data, "test")
        path_labels = os.path.join(path_labels, "test")
        data_dict = load_data(path_data, path_labels)

    config = load_config(args.config_dir)

    transforms_dict = config["transforms"]
    transforms_no_load = (
        training_transforms(transforms_dict)
        if args.phase == "train"
        else testing_transforms(transforms_dict)
    )
    
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=True, reader=PILReader(reverse_indexing=False)),
        transforms_no_load
    ])

    dataset = Dataset(data=data_dict, transform=transforms)

    save_dataset(dataset, save_dir, "transformed", show_landmarks=False)

    if args.test_inverse is True:
        inversed_dataset = []
        for sample in dataset:
            sample['landmarks_meta_dict']['scaling_factors'] = torch.from_numpy(sample['landmarks_meta_dict']['scaling_factors'])
            sample['image'] = sample['image'] 
            sample = transforms.inverse(sample)   
            inversed_dataset.append(sample)
        save_dataset(inversed_dataset, os.path.join(save_dir, "inversed"), "inversed")




if __name__ == "__main__":
    main()
