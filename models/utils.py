"""
This file contains utility functions relative to the model and training process.
"""

import torch
import numpy as np
import os
import sys

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)


def make_same_type(outputs, landmarks, loss_method, device="cpu"):
    """
    Make the outputs and landmarks the same type for loss calculation.
    """    
    if loss_method == "heatmap":
        map_size = outputs.shape[-2:]
        heatmap_from_landmarks = ld2hm(landmarks, map_size, device)

        return torch.sigmoid(outputs), heatmap_from_landmarks
    
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

    kernel_ratio = 0.05
    kernel_size = kernel_ratio * min(spatial_size)

    context_ratio = 0.5
    context_size = context_ratio * np.array([spatial_size[1], spatial_size[0]])

    batch_size, n_landmarks, _ = landmarks.shape
    h, w = spatial_size
    
    heatmaps = np.zeros((batch_size, n_landmarks, h, w))

    for batch in range(batch_size):
        for ld in range(n_landmarks):
            y, x = landmarks[batch, ld]
            x = int(x)
            y = int(y)

            if x < 0 or x >= h or y < 0 or y >= w:
                continue

            heatmap = np.zeros((h, w))
            xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')    
            context = np.exp(-(((xx - x) ** 2) / (2*context_size[1]**2) + ((yy - y)**2) / (2*context_size[0]**2)))
            heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * kernel_size**2)) + 0.5*context
            heatmap = heatmap / np.max(heatmap)
            heatmaps[batch, ld] = heatmap
            

    # convert heatmaps to tensor
    heatmaps = torch.tensor(
        heatmaps, dtype=torch.float32, device=device, requires_grad=True
    )
    return heatmaps

def make_landmarks(outputs, device="cpu"):
    """
    Convert the outputs to landmarks, taking the argmax.
    This function is used for the evaluation of the model, not for the backpropagation (discontinuous).
    """
    outputs = outputs.cpu().detach().numpy()
    batch_size, n_landmarks, h, w = outputs.shape

    landmarks = np.zeros((batch_size, n_landmarks, 2))

    for i in range(batch_size):
        for j in range(n_landmarks):
            
            heatmap = outputs[i, j]
            y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            landmarks[i, j] = np.array([x, y])

    landmarks = torch.tensor(landmarks, dtype=torch.float32, device=device)
    return landmarks
