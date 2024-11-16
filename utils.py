"""
docstring
"""

import numpy as np
import os
import glob
import yaml
import torch

from scipy.io import loadmat


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
