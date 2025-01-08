"""
This file contains the training and testing transforms for the model, and defines custom transforms.
Author: Jeanne Malécot
"""

import torch
import einops
from monai.transforms import Resize, Compose, LoadImaged, EnsureChannelFirstd, CopyItemsd, InvertibleTransform
from monai.transforms.spatial.functional import resize
from monai.data import PILReader
import numpy as np

import cv2


def training_transforms(transforms_dict):
    """
    Define the training transforms.
    """
    return Compose(
        [
            #LoadImaged(keys=["image"], image_only=True, reader=PILReader(reverse_indexing=False)),
            EnsureChannelFirstd(keys=["image"]),
            ClaheNormalizationd(keys=["image"]),
            ResizeWithLandmarksd(
                spatial_size=tuple(transforms_dict["resizing"]["spatial_size"]),
                mode=transforms_dict["resizing"]["interpolation"],
                keys=["image", "landmarks"],
                meta_keys=["image_meta_dict", "landmarks_meta_dict"],
            ), 
        ]
    )

def testing_transforms(transforms_dict):
    """
    Define the testing transforms.
    """
    return Compose(
        [
            #LoadImaged(keys=["image"], image_only=True, reader=PILRead(reverse_indexing=False)),
            EnsureChannelFirstd(keys=["image"]),
            ClaheNormalizationd(keys=["image"]),
            ResizeWithLandmarksd(
                spatial_size=transforms_dict["resizing"]["spatial_size"],
                mode=transforms_dict["resizing"]["interpolation"],
                keys=["image", "landmarks"],
                meta_keys=["image_meta_dict", "landmarks_meta_dict"],
            ),
        ]
    )

class ClaheNormalizationd(InvertibleTransform):
    """
    Apply CLAHE normalization to the image.
    TODO: add in config file the parameters for CLAHE
    """
    def __init__(
        self,
        keys=["image"],
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    ):
        self.keys = keys
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, data):
        image = data[self.keys[0]]
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        image = image.numpy().squeeze().astype(np.uint8)
        image = clahe.apply(image)
        data["image"] = torch.tensor(image).unsqueeze(0).float()
        return data
    
    def inverse(self, data):
        return data

class ResizeWithLandmarksd(InvertibleTransform):
    """
    Resize the image and the landmarks accordingly, using the Resize transform from MONAI.
    """

    def __init__(
        self,
        spatial_size,
        mode="bilinear",
        align_corners=False,
        keys=["image", "landmarks"],
        meta_keys=["image_meta_dict", "landmarks_meta_dict"],
    ):
        """
        Args:
            spatial_size: The target spatial size for resizing.
            mode: Interpolation mode for resizing images.
            align_corners: Whether to align corners during resizing.
            keys: Keys for the image and landmarks in the input dictionary.
            meta_keys: Metadata keys to store original sizes and scaling factors.
        """
        self.spatial_size = spatial_size
        self.mode = mode
        self.align_corners = align_corners
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, data):

        image = data[self.keys[0]]
        original_height, original_width = image.shape[-2], image.shape[-1]
        original_size = (original_height, original_width)

        resized_image = resize(image, self.spatial_size, mode=self.mode, align_corners=self.align_corners, dtype=None, input_ndim=2, anti_aliasing=True, anti_aliasing_sigma=1.5, lazy=False, transform_info=None)

        resized_height, resized_width = resized_image.shape[-2], resized_image.shape[-1]
        data[self.keys[0]] = resized_image

        landmarks = data[self.keys[1]]
        scaling_factors = torch.tensor(
            [resized_width / original_width, resized_height / original_height],
            dtype=landmarks.dtype,
        )
        data[self.keys[1]] = torch.round(landmarks * scaling_factors)

        #append to the dict original size
        data[self.meta_keys[0]].update({"original_size": original_size})
        data[self.meta_keys[1]] = {"scaling_factors": scaling_factors.numpy()}

        return data

    def inverse(self, data):

        for meta_key in self.meta_keys:
            if meta_key not in data:
                raise KeyError(f"Missing metadata key '{meta_key}' in data.")

        image_meta = data[self.meta_keys[0]]
        landmarks_meta = data[self.meta_keys[1]]
        original_size = image_meta["original_size"]
        scaling_factors = landmarks_meta["scaling_factors"].clone().detach()
        image = data[self.keys[0]]
        inverted_image = resize(
            image,
            original_size,
            mode=self.mode,
            align_corners=self.align_corners,
            dtype=None,
            input_ndim=2,
            anti_aliasing=True,
            anti_aliasing_sigma=1.5,
            lazy=False,
            transform_info=None,
        )

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
