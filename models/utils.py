"""
This file contains utility functions relative to the model and training process.
"""

import torch
import numpy as np
import os
import sys

from scipy.ndimage import gaussian_filter

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)


def make_same_type(outputs, landmarks, loss_method, gt_infos, device="cpu"):
    """
    Make the outputs and landmarks the same type for loss calculation.
    """    
    if loss_method == "heatmap":
        map_size = outputs.shape[-2:]
        n_channels = outputs.shape[1]
        heatmap_from_landmarks = ld2hm(landmarks, map_size, n_channels, gt_infos, device)

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

def ld2hm(landmarks, spatial_size=(512, 1024), n_channels = 68, gt_infos = {"ld_ratio": 0.05, "context_ratio": 0.5},  device="cpu"):
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

    kernel_ratio = gt_infos["ld_ratio"]
    kernel_size = kernel_ratio * min(spatial_size)

    context_ratio = gt_infos["context_ratio"]
    context_size = context_ratio * np.array([spatial_size[1], spatial_size[0]])

    batch_size, n_landmarks, _ = landmarks.shape
    h, w = spatial_size
    
    split_lumbar = False
    
    if n_channels == 12:
        split_lumbar = True
        n_channels = 6
        heatmaps = np.zeros((batch_size, n_channels*2, h, w))
    else:
        heatmaps = np.zeros((batch_size, n_channels, h, w))

    add_center = False
    add_top_bottom = False

    orig_n_channel = n_channels
    #if n_channels is 5, we add the centers of the vertebraes
    if n_channels == 5:
        n_channels = 4
        add_center = True

    if n_channels == 6:
        n_channels = 4
        add_center = True
        add_top_bottom = True

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
            alpha = 0.5 if n_channels == n_landmarks else 0
            heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * kernel_size**2)) + alpha*context

            if split_lumbar and ld >= 12*4:
                heatmaps[batch, ld%n_channels + orig_n_channel] += heatmap
            else:   
                heatmaps[batch, ld%n_channels] += heatmap
        
        for i in range(n_channels):
            heatmaps[batch, i] = heatmaps[batch, i] / np.max(heatmaps[batch, i])

        if split_lumbar:
            for i in range(orig_n_channel, n_channels + orig_n_channel):
                heatmaps[batch, i] = heatmaps[batch, i] / np.max(heatmaps[batch, i])

        if add_center:
            center_ld = []
            for i in range(0, n_landmarks, 4):
                y_center = int((landmarks[batch, i][0] + landmarks[batch, i+1][0] + landmarks[batch, i+2][0] + landmarks[batch, i+3][0]) / 4)
                x_center = int((landmarks[batch, i][1] + landmarks[batch, i+1][1] + landmarks[batch, i+2][1] + landmarks[batch, i+3][1]) / 4)
                center_ld.append((y_center, x_center))

                heatmap = np.zeros((h, w))
                xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

                heatmap = np.exp(-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * kernel_size**2))

                if split_lumbar and i >= 12*4:
                    heatmaps[batch, orig_n_channel + 4] += heatmap
                else:
                    heatmaps[batch, 4] += heatmap

            heatmaps[batch, 4] = heatmaps[batch, 4] / np.max(heatmaps[batch, 4]) 
            if split_lumbar:
                heatmaps[batch, orig_n_channel + 4] = heatmaps[batch, orig_n_channel + 4] / np.max(heatmaps[batch, orig_n_channel + 4])

            if add_top_bottom:
                if not split_lumbar:

                    top_center = min(center_ld, key=lambda x: x[1])

                    heatmap = np.zeros((h, w))
                    xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

                    heatmap = np.exp(-((xx - top_center[1]) ** 2 + (yy - top_center[0]) ** 2) / (2 * kernel_size**2))
                    heatmaps[batch, 5] += heatmap

                else:

                    top_center_thoracic = min(center_ld[:12], key=lambda x: x[1])
                    top_center_lumbar = min(center_ld[12:], key=lambda x: x[1])

                    heatmap = np.zeros((h, w))
                    xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

                    heatmap = np.exp(-((xx - top_center_thoracic[1]) ** 2 + (yy - top_center_thoracic[0]) ** 2) / (2 * kernel_size**2))
                    heatmaps[batch, 5] += heatmap

                    heatmap = np.zeros((h, w))
                    xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

                    heatmap = np.exp(-((xx - top_center_lumbar[1]) ** 2 + (yy - top_center_lumbar[0]) ** 2) / (2 * kernel_size**2))
                    heatmaps[batch, 11] += heatmap


                heatmaps[batch, 5] = heatmaps[batch, 5] / np.max(heatmaps[batch, 5])
                if split_lumbar:
                    heatmaps[batch, 11] = heatmaps[batch, 11] / np.max(heatmaps[batch, 11])
     
    # convert heatmaps to tensor
    heatmaps = torch.tensor(
        heatmaps, dtype=torch.float32, device=device, requires_grad=True
    )
    return heatmaps

