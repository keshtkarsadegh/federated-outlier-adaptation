"""
Compact VGG-style convolutional network for CIFAR-10.

Three blocks of ``conv-BN-ReLU`` x2 followed by max pooling with widths
64 / 128 / 256, an adaptive average pool down to 2x2, a 256-unit fully connected
layer and the classification head - about 1.4 M parameters, in the range the
CIFAR-10 federated literature uses.

The parameter-group interface mirrors
:class:`~federated_outlier_adaptation.model.FlexibleCNN`: ``body`` is the
convolutional stack plus ``fc1``, ``head`` is ``fc2``, and ``penultimate``
returns the 256-dimensional activation feeding the head.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class CifarCNN(nn.Module):
    """Convolutional classifier for 32x32 RGB images."""

    def __init__(
        self,
        num_classes: int = 10,
        widths: Sequence[int] = (64, 128, 256),
        fc_dim: int = 256,
        dropout_rate: float = 0.1,
        use_dropout: bool = True,
        in_channels: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.widths = tuple(int(width) for width in widths)
        self.fc_dim = int(fc_dim)
        self.pool_size = int(pool_size)

        self.features = nn.Sequential()
        channels = int(in_channels)
        for block, width in enumerate(self.widths):
            self.features.add_module(
                f"conv_{block}a", nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False)
            )
            self.features.add_module(f"bn_{block}a", nn.BatchNorm2d(width))
            self.features.add_module(f"relu_{block}a", nn.ReLU(inplace=True))
            self.features.add_module(
                f"conv_{block}b", nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
            )
            self.features.add_module(f"bn_{block}b", nn.BatchNorm2d(width))
            self.features.add_module(f"relu_{block}b", nn.ReLU(inplace=True))
            self.features.add_module(f"pool_{block}", nn.MaxPool2d(kernel_size=2, stride=2))
            channels = width
        self.features.add_module(
            "final_pool", nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        )

        self.flatten_size = channels * self.pool_size * self.pool_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.num_classes)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None

    # ------------------------------------------------------------- forward
    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        """The ``fc1`` activation, shape ``[B, fc_dim]``."""
        x = self.features(x)
        x = self.flatten(x)
        return F.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.penultimate(x)
        if self.dropout is not None:
            features = self.dropout(features)
        return self.fc2(features)

    # ------------------------------------------------------ parameter groups
    def body_parameters(self):
        return list(self.features.parameters()) + list(self.fc1.parameters())

    def head_parameters(self):
        return list(self.fc2.parameters())

    def freeze_body(self):
        for parameter in self.body_parameters():
            parameter.requires_grad = False

    def unfreeze_body(self):
        for parameter in self.body_parameters():
            parameter.requires_grad = True

    def set_dropout(self, enable: bool = True):
        if self.dropout is not None:
            self.dropout = nn.Dropout(self.dropout.p) if enable else None

    def print_param_groups(self):
        print(f"Body Params: {sum(p.numel() for p in self.body_parameters())}")
        print(f"Head Params: {sum(p.numel() for p in self.head_parameters())}")
