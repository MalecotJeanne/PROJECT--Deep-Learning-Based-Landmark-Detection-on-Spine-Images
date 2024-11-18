'''
TODO: docstring
'''

import torch
import numpy as np

def hm2ld(heatmaps, device="cpu"):
    '''
    Convert heatmaps to landmarks, using the mathematical expectation of the heatmap.
    Input:
        heatmaps: tensor of shape (batch_size, n_landmarks, h, w)
        device: device on which the heatmaps are stored
    Output:
        landmarks: tensor of shape (batch_size, n_landmarks, 2)
    '''
    # convert heatmaps to numpy array
    heatmaps = heatmaps.cpu().detach().numpy()

    # get the shape of the heatmaps
    batch_size, n_landmarks, h, w = heatmaps.shape

    # create the landmarks array
    landmarks = np.zeros((batch_size, n_landmarks, 2))

    # get the x and y coordinates
    for i in range(batch_size):
        for j in range(n_landmarks):
            heatmap = heatmaps[i, j]

            grid_x, grid_y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            x = np.sum(heatmap * grid_x)
            y = np.sum(heatmap * grid_y)
            
            landmarks[i, j] = np.array([x, y])

    # convert landmarks to tensor
    landmarks = torch.tensor(landmarks, dtype=torch.float32, device=device, requires_grad=True)

    return landmarks
