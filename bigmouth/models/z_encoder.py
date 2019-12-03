import torch
from torch import nn


class SemanticZEncoder(nn.Module):
    """Convert semantic space to (z, classes)."""
    def __init__(self, semantic_dims, z_dims, classes_dims, hidden_dims=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(semantic_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
        )
        self.z_encoder = nn.Sequential(
            nn.Linear(hidden_dims, z_dims),
        )
        self.classes_encoder = nn.Sequential(
            nn.Linear(hidden_dims, classes_dims),
        )

    def forward(self, s):
        features = self.features(s)
        z = self.z_encoder(features)
        classes = self.classes_encoder(features)
        return z, classes
