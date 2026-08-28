"""
Model topologies of the additional datasets.

``model.py`` keeps :class:`~federated_outlier_adaptation.model.FlexibleCNN`,
the topology of the published NIST experiments, untouched.  The models here
expose the same interface - ``forward``, ``penultimate``, ``body_parameters``,
``head_parameters``, ``freeze_body``, ``unfreeze_body`` - so every trainer,
runner and aggregation method works with them unchanged.
"""

from federated_outlier_adaptation.models.char_lstm import CharLSTM
from federated_outlier_adaptation.models.cifar_cnn import CifarCNN
from federated_outlier_adaptation.models.fedavg_cnn import FedAvgCNN

__all__ = ["CharLSTM", "CifarCNN", "FedAvgCNN"]
