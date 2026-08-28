"""
The reference convolutional network of the federated-averaging literature.

:class:`FedAvgCNN` is the MNIST/EMNIST model of McMahan, Moore, Ramage,
Hampson and Aguera y Arcas, *Communication-Efficient Learning of Deep Networks
from Decentralized Data*, AISTATS 2017, section 3 ("Experimental Results",
the second MNIST model)::

    "A CNN with two 5x5 convolution layers (the first with 32 channels, the
     second with 64, each followed with 2x2 max pooling), a fully connected
     layer with 512 units and ReLu activation, and a final softmax output
     layer (1,663,370 total parameters)."

The parameter count pins the remaining degrees of freedom down: with 28x28
inputs the quoted 1,663,370 parameters for ten classes are reached exactly when
both convolutions keep the spatial size (``padding=2``, i.e. "same"), so the
two poolings take 28 -> 14 -> 7 and the fully connected layer sees
``7 * 7 * 64 = 3136`` features::

    conv1   5*5*1*32  + 32  =       832
    conv2   5*5*32*64 + 64  =    51,264
    fc1     3136*512  + 512 = 1,606,144
    fc2     512*10    + 10  =     5,130
                              ---------
                              1,663,370

With the 62 classes of the by-class NIST task the head grows to
``512*62 + 62 = 31,806`` and the total is 1,690,046.

There is no normalisation layer and no dropout: the paper's model has neither,
and both interact badly with federated averaging (BatchNorm statistics are not
averageable across non-IID clients).  ``penultimate`` returns the 512-unit ReLU
activation, so the model satisfies the same contract as
:class:`~federated_outlier_adaptation.model.FlexibleCNN` and every trainer,
runner and aggregation rule works with it unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#: Spatial edge length the topology is defined for.
INPUT_SIZE = 28

#: Channel widths of the two convolutions, as published.
CONV_WIDTHS = (32, 64)

#: Units of the fully connected layer, as published.
FC_DIM = 512


class FedAvgCNN(nn.Module):
    """
    Two 5x5 convolutions, two poolings, FC-512 and a linear head.

    Args:
        num_classes: Output units (62 for the by-class NIST task, 10 for the
            digit ablation).
        input_size: Spatial edge length of the input; 28 is the resolution the
            published topology is defined at.
        in_channels: Input channels (1, grayscale).
    """

    def __init__(
        self,
        num_classes: int = 62,
        input_size: int = INPUT_SIZE,
        in_channels: int = 1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)

        self.features = nn.Sequential()
        self.features.add_module(
            "conv_0", nn.Conv2d(int(in_channels), CONV_WIDTHS[0], kernel_size=5, padding=2)
        )
        self.features.add_module("relu_0", nn.ReLU())
        self.features.add_module("pool_0", nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module(
            "conv_1", nn.Conv2d(CONV_WIDTHS[0], CONV_WIDTHS[1], kernel_size=5, padding=2)
        )
        self.features.add_module("relu_1", nn.ReLU())
        self.features.add_module("pool_1", nn.MaxPool2d(kernel_size=2, stride=2))

        with torch.no_grad():
            dummy = torch.zeros(1, int(in_channels), self.input_size, self.input_size)
            self.flatten_size = int(self.features(dummy).view(1, -1).shape[1])

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, FC_DIM)
        self.fc2 = nn.Linear(FC_DIM, self.num_classes)

        #: The published model has no dropout; the attribute exists so that the
        #: trainers' optional ``set_dropout`` calls stay valid.
        self.dropout = None

    # ------------------------------------------------------------------ forward
    def penultimate(self, x):
        """
        The 512-dimensional ReLU activation feeding the classification head.

        Args:
            x (torch.Tensor): Input batch ``[N, 1, H, W]``.

        Returns:
            torch.Tensor: Features ``[N, 512]``.
        """
        x = self.features(x)
        x = self.flatten(x)
        return F.relu(self.fc1(x))

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input batch.
            return_features (bool, optional): Additionally return the
                penultimate features.  Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                ``logits`` or ``(logits, features)``.
        """
        features = self.penultimate(x)
        logits = self.fc2(features)
        if return_features:
            return logits, features
        return logits

    # --------------------------------------------------------- parameter groups
    def body_parameters(self):
        """Feature extractor plus the fully connected layer."""
        return list(self.features.parameters()) + list(self.fc1.parameters())

    def head_parameters(self):
        """The classification head."""
        return list(self.fc2.parameters())

    def freeze_body(self):
        for parameter in self.body_parameters():
            parameter.requires_grad = False

    def unfreeze_body(self):
        for parameter in self.body_parameters():
            parameter.requires_grad = True

    def set_dropout(self, enable=True):
        """No-op: the published topology has no dropout layer."""
        self.dropout = None

    def get_regularization_loss(self):
        """No built-in weight penalty; the trainers own the regularisation."""
        return 0.0
