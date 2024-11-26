"""
TODO: docstring
"""

import numpy as np
import os
import glob
import yaml
import torch
import cv2

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
        )  # TODO: add logger

    labels = load_labels_mat(labels)

    dict_data = create_dict(data, labels)

    return dict_data


def create_dict(data, labels):
    """
    Create a dictionary with the data and labels suitable for the Monai Dataset class
    """
    data_dict = [
        {"image": img, "landmarks": torch.Tensor(lbl)} for img, lbl in zip(data, labels)
    ]
    return data_dict

def load_labels_mat(dir_list):
    """
    docstring (with ref)
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
    sorted_items = sorted(items, key=get_creation_time)
    return sorted_items


def get_last_folder(path, model_name):
    """
    Return the last folder (by date) in the given path
    """
    list_dir = sort_by_date(path)
    for dir in list_dir:
        if model_name in dir and os.path.isdir(os.path.join(path, dir)):
            return dir
    return None


### Image processing functions ###

def save_images(images, save_dir, basename="image"):
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
