"""File containing the losses for Teja-VAE"""
from torchmetrics import MeanSquaredError
import torch

def reconstruction_loss(pred, target):
    mean_squared_error = MeanSquaredError()
    mean_squared_error = mean_squared_error.to(pred.device)
    L = pred.shape[0]

    return mean_squared_error(pred, target)
