"""
The penultimate-representation accessors added for the feature-alignment
objective: shapes, and that the plain ``forward(x)`` path is unchanged.
"""

from __future__ import annotations

import torch

from federated_outlier_adaptation.model import FlexibleCNN
from federated_outlier_adaptation.trainers.feature_alignment_trainer import (
    logits_and_features,
)

from .conftest import FeaturesOnlyCNN, TinyCNN


def _model():
    torch.manual_seed(0)
    return FlexibleCNN(num_classes=10, complexity=3, dropout_rate=0.1, use_dropout=True)


def test_penultimate_shape_is_the_fc1_width():
    model = _model().eval()
    batch = torch.rand(4, 1, 128, 128)
    features = model.penultimate(batch)
    assert features.shape == (4, 128)
    assert torch.all(features >= 0), "the penultimate accessor returns a post-ReLU activation"


def test_forward_returns_logits_by_default():
    model = _model().eval()
    batch = torch.rand(3, 1, 128, 128)
    logits = model(batch)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (3, 10)


def test_forward_with_features_matches_the_separate_calls():
    model = _model().eval()
    batch = torch.rand(3, 1, 128, 128)
    with torch.no_grad():
        logits_only = model(batch)
        logits, features = model(batch, return_features=True)
        separate = model.penultimate(batch)
    assert torch.equal(logits, logits_only)
    assert torch.equal(features, separate)


def test_features_are_taken_before_dropout():
    """Dropout must not touch the returned representation."""
    model = _model().train()
    torch.manual_seed(1)
    batch = torch.rand(8, 1, 128, 128)
    _, features = model(batch, return_features=True)
    reference = model.penultimate(batch)
    assert torch.allclose(features, reference)


def test_logits_and_features_helper_handles_both_model_styles():
    batch = torch.rand(2, 1, 128, 128)
    for model in (TinyCNN().eval(), FeaturesOnlyCNN().eval()):
        with torch.no_grad():
            logits, features = logits_and_features(model, batch)
        assert logits.shape[0] == 2
        assert features.shape == (2, 16)
