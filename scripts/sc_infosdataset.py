"""
Script to extract pertinent info on the dataset.
Load the data and labels, create the dataset and extract information on the transformed data.
Author: Jeanne Malécot
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.data import Dataset, PILReader
from monai.transforms import Compose, LoadImaged

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from utils import load_data, load_config, save_dataset
from transforms import testing_transforms

parser = argparse.ArgumentParser(description="Script to test a dataset.")

parser.add_argument(
    "--data_dir", default="Ressources/boostnet_labeldata/", type=str, help="data path"
)
parser.add_argument(
    "--config_dir", default="config.yaml", type=str, help="path to the config file"
)
parser.add_argument(
    "--save_dir", default="Ressources/boostnet_labeldata/infos_dataset", type=str, help="path to save the images"
)

args = parser.parse_args()


def main():
    """
    Load the data and labels, create the dataset and visualize the data transformed.
    """
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_data = os.path.join(args.data_dir, "data")
    path_labels = os.path.join(args.data_dir, "labels")

    if not os.path.exists(path_data):
        raise ValueError("Data folder does not exist")
    if not os.path.exists(path_labels):
        raise ValueError("Labels folder does not exist")

    path_train_data = os.path.join(path_data, "training")
    path_train_labels = os.path.join(path_labels, "training")
    train_dict = load_data(path_train_data, path_train_labels)

    path_test_data = os.path.join(path_data, "test")
    path_test_labels = os.path.join(path_labels, "test")
    test_dict = load_data(path_test_data, path_test_labels)

    data_dict = train_dict + test_dict



    #number of valid samples
    n_valid_samples = 0
    for data in data_dict:
        if data["landmarks"] is not None and len(data["landmarks"]) == 68:
            n_valid_samples += 1
            
    #shape of untransformed images
    avg_shape = [0, 0]

    #intensity of untransformed images
    mean_intensity = 0
    std_intensity = 0
    
    #histogram of untransformed images
    hist = np.zeros(256)

    for data in tqdm(data_dict, desc="Infos on untransformed images..."):
        path_image = data["image"]
        image = plt.imread(path_image)
        avg_shape[0] += image.shape[0]
        avg_shape[1] += image.shape[1]

        mean_intensity += image.mean()
        std_intensity += image.std()

        hist += np.histogram(image, bins=256, range=(0, 255))[0] / (image.shape[0] * image.shape[1]) 

    avg_shape[0] = avg_shape[0] / len(data_dict)
    avg_shape[1] = avg_shape[1] / len(data_dict)

    mean_intensity = mean_intensity / len(data_dict)
    std_intensity = std_intensity / len(data_dict)

    hist = hist / len(data_dict)
    
    print("Saving histogram of untransformed images...")
    plt.figure()
    plt.bar(range(256), hist)
    plt.title("Histogram of the images")
    plt.xlabel("Intensity")
    plt.ylabel("Percentage of pixels")
    plt.savefig(os.path.join(save_dir, "histogram_orig.png"))
    
    # apply transforms
    config = load_config(args.config_dir)

    transforms_dict = config["transforms"]   
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=True, reader=PILReader(reverse_indexing=False)),
        testing_transforms(transforms_dict)
    ])

    dataset = Dataset(data=data_dict, transform=transforms)

    dir_dataset = os.path.join(save_dir, "dataset")
    if not os.path.exists(dir_dataset):
        os.makedirs(dir_dataset)
    save_dataset(dataset, dir_dataset, "transformed")

    
    #intensity of transformed images
    mean_intensity_tr = 0
    std_intensity_tr = 0
    hist = np.zeros(256)
    for data in tqdm(dataset, desc="Infos on transformed images..."):
        image = data["image"][0].numpy() * 255
        mean_intensity_tr += image.mean()
        std_intensity_tr += image.std()
        hist += np.histogram(image, bins=256, range=(0, 255))[0] / (image.shape[0] * image.shape[1])
    mean_intensity_tr = mean_intensity_tr / len(dataset)
    std_intensity_tr = std_intensity_tr / len(dataset)
    hist = hist / len(dataset)

    print("Saving histograms...")
    plt.figure()
    plt.bar(range(256), hist)
    plt.title("Histogram of the transformed images")
    plt.xlabel("Intensity")
    plt.ylabel("Percentage of pixels")
    plt.savefig(os.path.join(save_dir, "histogram_transformed.png"))

    #labels analysis
    average_spine_width = 0
    std_spine_width = 0
    average_spine_height = 0
    std_spine_height = 0
    average_space_vertebraes = 0
    std_space_vertebraes = 0
    for data in tqdm(dataset, desc="Lavel analysis"):
        if data["landmarks"] is not None:
            landmarks = data["landmarks"].numpy()
            
            vertebrae = landmarks.reshape(-1, 4, 2)

            widths = np.linalg.norm(vertebrae[:, 0] - vertebrae[:, 1], axis=1)
            average_spine_width += np.mean(widths)
            std_spine_width += np.std(widths)

            height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
            average_spine_height += height
            std_spine_height += np.std(landmarks[:, 1])

            spaces = np.linalg.norm(np.diff(vertebrae[:, 2], axis=0), axis=1)
            average_space_vertebraes += np.mean(spaces)
            std_space_vertebraes += np.std(spaces)

    average_spine_width = average_spine_width / len(dataset)
    std_spine_width = std_spine_width / len(dataset)

    average_spine_height = average_spine_height / len(dataset)
    std_spine_height = std_spine_height / len(dataset)

    average_space_vertebraes = average_space_vertebraes / len(dataset)
    std_space_vertebraes = std_space_vertebraes / len(dataset)

    print("Writing infos.txt...")

    with open(os.path.join(save_dir, "infos.txt"), "w") as f:
        f.write(f"boostnet_labeldata dataset:\n")
        f.write(f"===============\n")
        f.write(f"{len(dataset)} images and corresponding labels\n")
        f.write(f"train-set: {len(train_dict)} samples | test-set: {len(test_dict)} samples\n")
        f.write(f"Images with 68 landmarks labels: {n_valid_samples}/{len(dataset)}\n")
        f.write(f"\n")
        f.write(f"Before transformations:\n")
        f.write(f"----------\n")
        f.write(f"Image average shape: {avg_shape}\n")
        f.write(f"\t> Ratio height/width: {avg_shape[0]/avg_shape[1]}\n")
        f.write(f"Image intensity: \n")
        f.write(f"\t> Mean: {mean_intensity}\n")
        f.write(f"\t> Std: {std_intensity}\n")
        f.write(f"\n")
        f.write(f"After transformations:\n")
        f.write(f"----------\n")
        f.write(f"Images shape: {config['transforms']['resizing']['spatial_size']}\n")
        f.write(f"Images intensity: \n")
        f.write(f"\t> Mean: {mean_intensity_tr}\n")
        f.write(f"\t> Std: {std_intensity_tr}\n")
        f.write(f"\n")
        f.write(f"Average spine width: {average_spine_width} +/- {std_spine_width}\n")
        f.write(f"Average spine height: {average_spine_height} +/- {std_spine_height}\n")
        f.write(f"Average space between vertebraes: {average_space_vertebraes} +/- {std_space_vertebraes}\n")
        
    print("Done!")

if __name__ == "__main__":
    main()
