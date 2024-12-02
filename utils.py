"""
General utility functions for the project
Author: Jeanne Malécot
"""
import glob
import os
from io import BytesIO
import glob
import os
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
import wandb
import yaml

from scipy.io import loadmat


### Data loading functions ###

def load_data(path_data, path_labels):
    """
    Load the data and labels from the given paths
    """
    # Load the data
    data = []
    for path_img in glob.glob(os.path.join(path_data + "/*.jpg")):
        data.append(path_img)
    # Load the labels
    labels = []
    for path_label in glob.glob(os.path.join(path_labels + "/*.mat")):
        labels.append(path_label)

    # Check if the data and labels have the same length
    if len(data) != len(labels):
        raise ValueError(
            "Number of data samples and labels do not match"
        )  

    labels = load_labels_mat(labels)

    dict_data = create_dict(data, labels)

    return dict_data


def create_dict(data, labels):
    """
    Create a dictionary with the data and labels suitable for the Monai Dataset class
    """
    data_dict = [
        {"image": img, "landmarks": torch.Tensor(lbl), "image_meta_dict":None, "landmarks_meta_dict":None } for img, lbl in zip(data, labels)
    ]
    return data_dict

def load_labels_mat(dir_list):
    """
    Load the labels from the .mat files
    """
    labels = []
    for label_dir in dir_list:
        labels.append((loadmat(label_dir)["p2"]).astype(np.int16))
        if (
            labels[-1].shape[0] != 68
        ):  # TODO: explain in report why, +eventually add logger
            # Number of landmarks can be != than 68
            orig_size = labels[-1].shape[0]
            labels[-1] = torch.cat([labels[-1], torch.zeros(68 - orig_size, 2)], dim=0)

    return labels


def load_config(config_dir):
    """
    Load the yaml configuration file
    """
    with open(config_dir, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def sort_by_date(directory):
    def get_creation_time(item):
        item_path = os.path.join(directory, item)
        return os.path.getctime(item_path)

    items = os.listdir(directory)
    sorted_items = sorted(items, key=get_creation_time, reverse=True)
    return sorted_items


def get_last_folder(path, model_name):
    """
    Return the last folder (by date) in the given path
    """
    list_dir = sort_by_date(path)
    for dir in list_dir:
        if model_name in dir and "train" in dir and os.path.isdir(os.path.join(path, dir)):
            return dir
    return None


### Image processing functions ###

def save_heatmaps(images, save_dir, basename="image"):
    """
    Save the images in the given directory
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_images = len(images)
    images_copy = images.clone()
    for i in range(n_images):
        image = images_copy[i].detach().cpu().numpy()
        image = normalize_image(image)
        save_path = os.path.join(save_dir, f"{basename}_{i}.jpg")
        # save the image with cmap jet
        cv2.imwrite(
            save_path, cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_JET)
        )

def save_dataset(dataset, save_dir):
    """
    Save the dataset in the given directory
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, data in enumerate(dataset):
        image = data["image"].detach().cpu().numpy()
        #image = normalize_image(image)

        #create image in cv2 format
        if len(image.shape) == 2:  # Check if the image is single-channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[0] == 1:  # Check if the image is single-channel
            image = cv2.cvtColor(image[0], cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = data["landmarks"].detach().cpu().numpy()
        for landmark in landmarks:
            cv2.circle(
                image, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1
            )

        save_path = os.path.join(save_dir, f"image_{i}.jpg")
        cv2.imwrite(save_path, image)

def wandb_img(tensor, cmap="jet", caption="image"):
    """
    convert a tensor to a wandb image, to be saved with the desired colormap
    """
    image = tensor.detach().cpu().numpy()
    image = normalize_image(image)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap=cmap)
    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close()

    image_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return wandb.Image(image)

def normalize_image(image): 
    """
    map the values to [0, 255]
    """
    image = image - np.min(image)
    image = (image / np.max(image)) * 255
    return image


def has_n_landmarks(labels, num_landmarks):
    """
    Return the indices of the data and labels that have the expected number of landmarks
    """
    indices = []
    for i in range(len(labels)):
        if labels[i].shape[0] == num_landmarks:
            indices.append(i)
    return indices

def get_left_landmarks(landmarks):
    """
    Return the coordinates of the left landmarks (odd lines of the landmarks array)
    """
    return landmarks[::2]

def get_right_landmarks(landmarks):
    """
    Return the coordinates of the right landmarks (even lines of the landmarks array)
    """
    return landmarks[1::2]

def get_middle_line(left_ld, right_ld):
    """
    Return the middle line between the left and right landmarks
    """
    return (left_ld + right_ld) / 2
