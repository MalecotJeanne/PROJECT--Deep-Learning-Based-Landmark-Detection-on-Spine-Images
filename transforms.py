"""
This file contains the training and testing transforms for the model, and defines custom transforms.
Author: Jeanne Malécot
"""

import torch
import einops
from monai.transforms import Resize, Compose, LoadImaged, EnsureChannelFirstd, CopyItemsd
from monai.data import PILReader
import numpy as np


def training_transforms(transforms_dict):
    """
    Define the training transforms.
    """
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=True, reader=PILReader(reverse_indexing=False)),
            EnsureChannelFirstd(keys=["image"]),
            ResizeWithLandmarksd(
                spatial_size=tuple(transforms_dict["resizing"]["spatial_size"]),
                mode=transforms_dict["resizing"]["interpolation"],
                keys=["image", "landmarks"],
            ), 
        ]
    )


def testing_transforms(transforms_dict):
    """
    Define the testing transforms.
    """
    return Compose(
        [
            #LoadImaged(keys=["image"], image_only=True, reader=PILReader(reverse_indexing=False)),
            EnsureChannelFirstd(keys=["image"]),
            #CopyItemsd(keys=["image", "landmarks"], names=["image_meta_dict", "landmarks_meta_dict"]),
            ResizeWithLandmarksd(
                spatial_size=transforms_dict["resizing"]["spatial_size"],
                mode=transforms_dict["resizing"]["interpolation"],
                keys=["image", "landmarks"],
                meta_keys=["image_meta_dict", "landmarks_meta_dict"],
            ),
        ]
    )

class ResizeWithLandmarksd(Resize):
    """
    Resize the image and the landmarks accordingly.
    Based on the Resize transform from MONAI.
    """
    def __init__(
        self,
        spatial_size,
        mode="bilinear",
        align_corners=False,
        keys=["image", "landmarks"],
        meta_keys=["image_meta_dict", "landmarks_meta_dict"],
    ):
        super().__init__(
            spatial_size=spatial_size, mode=mode, align_corners=align_corners
        )
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, data, **kwargs):
        image, landmarks = data[self.keys[0]], data[self.keys[1]]
        original_height, original_width = image.shape[-2], image.shape[-1]
        original_size = (original_height, original_width)

        resized_image = super().__call__(image, **kwargs)
        resized_height, resized_width = resized_image.shape[-2], resized_image.shape[-1]

        scaling_factors = np.array(
            [resized_width / original_width, resized_height / original_height],
            dtype=landmarks.numpy().dtype,
        )

        data[self.keys[0]] = resized_image
        data[self.keys[1]] = torch.round(landmarks * torch.tensor(scaling_factors, dtype=landmarks.dtype))

        # saving metadata
        data[self.meta_keys[0]] = {"original_size": np.array(original_size)}
        data[self.meta_keys[1]] = {"scaling_factors": scaling_factors}

        return data

    def inverse(self, data):
        """
        Inverse the resizing.
        """
        print(data )
        image_meta, landmarks_meta = data[self.meta_keys[0]], data[self.meta_keys[1]]
        original_size = image_meta["original_size"]
        scaling_factors = landmarks_meta["scaling_factors"]

        image = data[self.keys[0]]
        inverted_image = super().__call__(image, spatial_size=original_size)

        landmarks = data[self.keys[1]]
        inverted_landmarks = torch.round(landmarks / scaling_factors)

        data[self.keys[0]] = inverted_image
        data[self.keys[1]] = inverted_landmarks
        return data


def softmax2d(input):
    """
    Apply softmax to an input of shape (batch_size, n_landmarks, h, w).
    """
    flatten = einops.rearrange(input, "b l h w -> b l (h w)")
    pdist = torch.nn.functional.softmax(flatten, dim=-1)
    pdist2d = einops.rearrange(pdist, "b l (h w) -> b l h w", h=input.shape[-2])
    return pdist2d
