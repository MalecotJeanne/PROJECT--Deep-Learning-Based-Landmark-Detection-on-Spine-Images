"""
This file contains utility functions relative to the model and training process.
"""

import torch
import numpy as np
import os
import sys

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from utils import normalize_image
from transforms import softmax2d


def calculate_accuracy(predictions, targets, threshold=20):
    """
    Calculate accuracy based on the distance between predicted and target landmarks.
    A prediction is considered correct if the distance is less than the threshold.
    """
    # distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    # correct_predictions = distances < threshold
    # accuracy = correct_predictions.float().mean().item() * 100
    # return accuracy
    return 0

def make_same_type(outputs, landmarks, loss_method, device="cpu"):
    """
    Make the outputs and landmarks the same type for loss calculation.
    """    
    if loss_method == "heatmap":
        map_size = outputs.shape[-2:]
        heatmap_from_landmarks = ld2hm(landmarks, map_size, device)

        return outputs, heatmap_from_landmarks
    else:
        output_landmarks = hm2ld(outputs, device)
        return output_landmarks, landmarks

def hm2ld(heatmaps, device="cpu"):
    """
    Convert heatmaps to landmarks, using the mathematical expectation of the heatmap.
    Input:
        heatmaps: tensor of shape (batch_size, n_landmarks, h, w)
        device: device on which the heatmaps are stored
    Output:
        landmarks: tensor of shape (batch_size, n_landmarks, 2)
    """
    heatmaps = heatmaps.cpu().detach().numpy()
    batch_size, n_landmarks, h, w = heatmaps.shape

    landmarks = np.zeros((batch_size, n_landmarks, 2))

    for i in range(batch_size):
        for j in range(n_landmarks):
            heatmap = heatmaps[i, j]

            grid_x, grid_y = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

            x = np.sum(heatmap * grid_x)
            y = np.sum(heatmap * grid_y)

            landmarks[i, j] = np.array([x, y])

    landmarks = torch.tensor(
        landmarks, dtype=torch.float32, device=device, requires_grad=True
    )

    return landmarks

def ld2hm(landmarks, spatial_size=(512, 1024), device="cpu"):
    """
    Convert landmarks to heatmaps, using 2d gaussian kernel.
    Input:
        landmarks: tensor of shape (batch_size, n_landmarks, 2)
        device: device on which the heatmaps are stored
        spatial_size: tuple of ints, the spatial size of the heatmaps
    Output:
        heatmaps: tensor of shape (batch_size, n_landmarks, h, w)
    """
    landmarks = landmarks.cpu().detach().numpy()

    batch_size, n_landmarks, _ = landmarks.shape
    h, w = spatial_size
    
    heatmaps = np.zeros((batch_size, n_landmarks, h, w))

    for batch in range(batch_size):
        for ld in range(n_landmarks):
            x, y = landmarks[batch, ld]
            x = int(x)
            y = int(y)

            if x < 0 or x >= h or y < 0 or y >= w:
                continue

            heatmap = np.zeros((h, w))
            xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 50) #TODO: find the optimal sigma value, and put it in a config

            heatmaps[batch, ld] = heatmap
            

    # convert heatmaps to tensor
    heatmaps = torch.tensor(
        heatmaps, dtype=torch.float32, device=device, requires_grad=True
    )
    #convert heatmaps to probabilities
    heatmaps = softmax2d(heatmaps)

    return heatmaps
