"""
Character-level LSTM for the LEAF Shakespeare next-character task.

The topology is LEAF's reference model (Caldas et al., 2018,
``leaf/models/shakespeare/stacked_lstm.py``): an 8-dimensional character
embedding, two stacked LSTM layers of 256 units and a linear read-out over the
80-symbol alphabet.  Only the final position of the sequence is classified.

The parameter-group interface mirrors
:class:`~federated_outlier_adaptation.model.FlexibleCNN`, so the selective
layer-freezing and feature-alignment trainers work without a special case:

* ``body`` - embedding and recurrent layers (the representation),
* ``head`` - the output projection.
"""

from __future__ import annotations

import torch
from torch import nn


class CharLSTM(nn.Module):
    """Stacked character LSTM producing ``[B, num_classes]`` logits."""

    def __init__(
        self,
        vocab: int = 80,
        embed: int = 8,
        hidden: int = 256,
        layers: int = 2,
        num_classes: int = None,
        dropout_rate: float = 0.0,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.vocab = int(vocab)
        self.embed = int(embed)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.num_classes = int(num_classes) if num_classes else self.vocab

        self.embedding = nn.Embedding(self.vocab, self.embed)
        self.lstm = nn.LSTM(self.embed, self.hidden, num_layers=self.layers, batch_first=True)
        self.fc = nn.Linear(self.hidden, self.num_classes)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout and dropout_rate > 0 else None

    # ------------------------------------------------------------- forward
    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        """Final hidden representation of the sequence, shape ``[B, hidden]``."""
        if x.dtype != torch.long:
            x = x.long()
        # The federated loops deep-copy models between rounds, which scatters
        # the recurrent weights; re-flattening keeps the fast cuDNN path.
        self.lstm.flatten_parameters()
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return output[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.penultimate(x)
        if self.dropout is not None:
            features = self.dropout(features)
        return self.fc(features)

    # ------------------------------------------------------ parameter groups
    def body_parameters(self):
        return list(self.embedding.parameters()) + list(self.lstm.parameters())

    def head_parameters(self):
        return list(self.fc.parameters())

    def freeze_body(self):
        for parameter in self.body_parameters():
            parameter.requires_grad = False

    def unfreeze_body(self):
        for parameter in self.body_parameters():
            parameter.requires_grad = True

    def print_param_groups(self):
        print(f"Body Params: {sum(p.numel() for p in self.body_parameters())}")
        print(f"Head Params: {sum(p.numel() for p in self.head_parameters())}")
