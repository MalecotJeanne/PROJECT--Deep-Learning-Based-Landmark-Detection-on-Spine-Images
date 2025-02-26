"""
Custom losses to train the models, and usefull functions
Author: Jeanne Malécot
"""

import torch
import torch.nn as nn   
import numpy as np
from torch.nn import functional as F

def pick_criterion(name):
    """
    pick the criterion to use for the training.
    args:
        name: str, the name of the criterion to use.    
    """
    supported_criterion = True
    if name == "mse" or name == "MSE":
        criterion = torch.nn.MSELoss()
    elif name.lower() == "mse_weight" or name.lower() == "mseweight":
        criterion = MseWight()
    elif name == "l1" or name == "L1":
        criterion = torch.nn.L1Loss()
    elif (
        name == "cross_entropy_loss"
        or name == "CrossEntropyLoss"
        or name == "CELoss"
    ):
        criterion = torch.nn.CrossEntropyLoss()
    elif (
        name == "BCELoss"
        or name == "bce"
        or name == "BCE"
    ):
        criterion = torch.nn.BCELoss()
    elif name == "kld" or name == "KLD" or name =="KLLoss":
        criterion = KLDivergenceLoss()
    elif name == "mixed" or name == "MixedLoss":
        criterion = MixedLoss()
    else:
        criterion = torch.nn.MSELoss()
        supported_criterion = False

    return criterion, supported_criterion

def pick_accuracy(name):
    """
    pick the accuracy metric to use for the training.
    args:
        name: str, the name of the accuracy metric to use.    
    """
    supported_accuracy = True
    if name == "nme" or name == "NME":
        accuracy = NormalizedMeanError()
    elif name == "pck" or name == "PCK":
        accuracy = PercentageOfCorrectKeypoints()
    else:
        accuracy = PercentageOfCorrectKeypoints()
        supported_accuracy = False
    return accuracy, supported_accuracy


# Loss functions

class MseWight(nn.Module):
    def __init__(self):
        super(MseWight, self).__init__()
    def forward(self, pred, gt):
        criterion = nn.MSELoss(reduction='none')
        loss = criterion(pred, gt)
        ratio = torch.pow(50, gt)
        loss = torch.mul(loss, ratio)
        loss = torch.mean(loss)
        return loss

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, pred, target):

        pred = (pred + 1e-6) / (pred + 1e-6).sum(dim=(-2, -1), keepdim=True)
        target = (target + 1e-6) / (target + 1e-6).sum(dim=(-2, -1), keepdim=True)

        return F.kl_div(pred.log(), target, reduction="batchmean")

class MixedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.kld_loss = KLDivergenceLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.kld_loss(pred, target) + (1 - self.alpha) * self.mse_loss(pred, target)


# Accuracy metrics

class NormalizedMeanError:
    def __init__(self, normalizing_factor=1.0):
        """
        NormalizedMeanError (NME) accuracy metric.
        args:
            normalizing_factor: float, for distances normalization 
        """
        self.normalizing_factor = normalizing_factor

    def __call__(self, predicted, ground_truth):
        """
        compute the normalized mean error between predicted and ground truth landmarks.
        args:
            predicted: Tensor(B, N, 2), predicted landmark coordinates.
            ground_truth: Tensor(B, N, 2), ground truth landmark coordinates.
        
        returns:
            nme: float, average normalized mean error.
        """
        if isinstance(predicted, np.ndarray):
            predicted = torch.from_numpy(predicted)
        if isinstance(ground_truth, np.ndarray):
            ground_truth = torch.from_numpy(ground_truth)

        distances = torch.linalg.norm(predicted - ground_truth, dim=-1) 
        normalized_distances = distances / self.normalizing_factor  # Shape: (B, N)

        invalid_mask = (predicted[..., 0] == -1) & (predicted[..., 1] == -1)
        normalized_distances[invalid_mask] = -1

        normalized_distances = normalized_distances[normalized_distances != -1]

        return normalized_distances.mean().item()

class PercentageOfCorrectKeypoints:
    def __init__(self, normalizing_factor=1.0, threshold=10):
        """
        PercentageOfCorrectKeypoints (PCK) accuracy metric.
        args:
            normalizing_factor: float, for distances normalization 
            threshold: float, the distance threshold relative to the normalizing factor.
        """
        self.normalizing_factor = normalizing_factor
        self.threshold = threshold

    def __call__(self, predicted, ground_truth):
        """
        compute the percentage of accuractes landmarks within the threshold.
        args:
            predicted: Tensor(B, N, 2), predicted landmark coordinates.
            ground_truth: Tensor(B, N, 2), ground truth landmark coordinates.
        
        returns:
            pck: float, percentage of correct predictions.
        """
        if isinstance(predicted, np.ndarray):
            predicted = torch.from_numpy(predicted)
        if isinstance(ground_truth, np.ndarray):
            ground_truth = torch.from_numpy(ground_truth)

        distances = torch.linalg.norm(predicted - ground_truth, dim=-1)  
        normalized_distances = distances / self.normalizing_factor  

        correct_keypoints = (normalized_distances < self.threshold).float() 

        invalid_mask = (predicted[..., 0] == -1) & (predicted[..., 1] == -1)
        correct_keypoints[invalid_mask] = 0 
       
        return correct_keypoints.mean().item() * 100  
