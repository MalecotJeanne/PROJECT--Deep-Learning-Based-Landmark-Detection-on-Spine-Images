"""
Custom losses to train the models, and usefull functions
Author: Jeanne Malécot
"""

import torch
import torch.nn as nn   
import numpy as np
from monai.losses import FocalLoss
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
    elif name == "distance" or name == "DistanceLoss":
        criterion = DistanceLoss()
    elif (
        name == "adaptive_wing"
        or name == "AdaptiveWing"
        or name == "AdapWingLoss"
    ):
        criterion = AdaptiveWingLoss()
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
    elif (
        name == "focal"
        or name == "FocalLoss"
        or name == "Focal"
    ):
        criterion = FocalLoss(gamma = 4.0, reduction ="mean")
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

class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))
    
class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.distance_loss(pred, target) + (1 - self.alpha) * self.mse_loss(pred, target)

class AdaptiveWingLoss(nn.Module):
    """
    Adaptation of the Adaptive Wing Loss mentionned in the paper:
    "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression" - Xinyao Wang, Liefeng Bo,
    Li Fuxin, Oregon State University, JD Digits
    """

    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        """
        The parameters are initialized as in the paper
        """
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        A = (
            self.omega
            * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y)))
            * (self.alpha - y)
            * ((self.theta / self.epsilon) ** (self.alpha - y - 1))
            * (1 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(
            1 + (self.theta / self.epsilon) ** (self.alpha - y)
        )

        if delta_y < self.theta:
            loss = self.omega * torch.log(1 + (delta_y / self.epsilon) ** (self.alpha-y))
        else:
            loss = A * delta_y - C

        return loss 

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, pred, target):

        pred = (pred + 1e-6) / (pred + 1e-6).sum(dim=(-2, -1), keepdim=True)
        target = (target + 1e-6) / (target + 1e-6).sum(dim=(-2, -1), keepdim=True)

        return F.kl_div(pred.log(), target, reduction="batchmean")

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
        distances = torch.linalg.norm(predicted - ground_truth, dim=-1) 
        normalized_distances = distances / self.normalizing_factor  # Shape: (B, N)

        return -normalized_distances.mean().item()

class PercentageOfCorrectKeypoints:
    def __init__(self, normalizing_factor=1.0, threshold=3):
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
        distances = torch.linalg.norm(predicted - ground_truth, dim=-1)  
        normalized_distances = distances / self.normalizing_factor  
        
        correct_keypoints = (normalized_distances < self.threshold).float() 
       
        return correct_keypoints.mean().item() * 100  
