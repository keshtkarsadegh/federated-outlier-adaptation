"""The added models must honour the FlexibleCNN interface contract."""

from __future__ import annotations

import pytest
import torch

from federated_outlier_adaptation.models import CharLSTM, CifarCNN
from federated_outlier_adaptation.model import FlexibleCNN

CONTRACT = ("forward", "penultimate", "body_parameters", "head_parameters", "freeze_body", "unfreeze_body")


def make_inputs(model):
    if isinstance(model, CharLSTM):
        return torch.randint(0, model.vocab, (5, 80))
    return torch.rand(5, 3, 32, 32)


@pytest.fixture(params=["char_lstm", "cifar_cnn"])
def model(request):
    if request.param == "char_lstm":
        return CharLSTM(vocab=80, embed=8, hidden=32, layers=2)
    return CifarCNN(num_classes=10, widths=(8, 16, 16), fc_dim=32)


def test_interface_matches_the_reference_model(model):
    for name in CONTRACT:
        assert callable(getattr(model, name)), name
        assert hasattr(FlexibleCNN, name) or name == "penultimate"


def test_forward_returns_class_logits(model):
    logits = model(make_inputs(model))
    assert logits.shape == (5, model.num_classes)
    assert logits.dtype == torch.float32


def test_penultimate_is_the_representation_feeding_the_head(model):
    features = model.penultimate(make_inputs(model))
    assert features.dim() == 2
    assert features.shape[0] == 5
    head = model.head_parameters()[0]
    assert features.shape[1] == head.shape[1]


def test_body_and_head_partition_the_parameters(model):
    body = {id(p) for p in model.body_parameters()}
    head = {id(p) for p in model.head_parameters()}
    every = {id(p) for p in model.parameters()}
    assert body & head == set()
    assert body | head == every


def test_freezing_toggles_only_the_body(model):
    model.freeze_body()
    assert all(not p.requires_grad for p in model.body_parameters())
    assert all(p.requires_grad for p in model.head_parameters())
    model.unfreeze_body()
    assert all(p.requires_grad for p in model.parameters())


def test_gradients_flow_through_the_whole_model(model):
    logits = model(make_inputs(model))
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(5, dtype=torch.long))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_char_lstm_follows_the_leaf_reference_topology():
    model = CharLSTM()
    assert model.vocab == 80 and model.num_classes == 80
    assert model.embedding.embedding_dim == 8
    assert model.lstm.hidden_size == 256 and model.lstm.num_layers == 2
    assert model.penultimate(torch.randint(0, 80, (2, 80))).shape == (2, 256)


def test_char_lstm_accepts_integer_inputs_of_any_width():
    model = CharLSTM(vocab=80, hidden=16, layers=1)
    for length in (10, 80):
        assert model(torch.randint(0, 80, (3, length))).shape == (3, 80)


def test_cifar_cnn_size_is_in_the_intended_range():
    model = CifarCNN()
    parameters = sum(p.numel() for p in model.parameters())
    assert 1_000_000 <= parameters <= 2_000_000
    assert model.penultimate(torch.rand(2, 3, 32, 32)).shape == (2, 256)


def test_state_dicts_round_trip(model):
    clone = type(model)(**_constructor_kwargs(model))
    clone.load_state_dict(model.state_dict())
    model.eval()
    clone.eval()
    inputs = make_inputs(model)
    assert torch.allclose(model(inputs), clone(inputs), atol=1e-6)


def _constructor_kwargs(model):
    if isinstance(model, CharLSTM):
        return {"vocab": model.vocab, "embed": model.embed, "hidden": model.hidden, "layers": model.layers}
    return {"num_classes": model.num_classes, "widths": model.widths, "fc_dim": model.fc_dim}
