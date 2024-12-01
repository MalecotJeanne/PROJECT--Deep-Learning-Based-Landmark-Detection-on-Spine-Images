"""
Custom losses to train the models
Author: Jeanne Malécot
"""

import torch
import torch.nn as nn   
import numpy as np


class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


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


class LandmarkAccuracy:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def euclidean_distance(self, preds, targets):
        return torch.norm(preds - targets, dim=-1)

    def evaluate(self, preds, targets):

        distances = self.euclidean_distance(preds, targets)
        n_landmarks = preds.shape[1]

        correct = (distances < self.threshold).sum()

        return correct / (n_landmarks * preds.shape[0])
