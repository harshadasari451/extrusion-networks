import torch.nn as nn
import torch

class RMSELoss(nn.Module):
    def __init__(self, eps=0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)
