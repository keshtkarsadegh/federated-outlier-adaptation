"""
Model topologies.

``model.py`` keeps :class:`~federated_outlier_adaptation.model.FlexibleCNN`,
the topology of the published NIST experiments, untouched.
:class:`~federated_outlier_adaptation.models.fedavg_cnn.FedAvgCNN` is the one
this study runs.  Both expose the same interface - ``forward``,
``penultimate``, ``body_parameters``, ``head_parameters``, ``freeze_body``,
``unfreeze_body`` - so every trainer, runner and aggregation method works with
either unchanged.
"""

from federated_outlier_adaptation.models.fedavg_cnn import FedAvgCNN

__all__ = ["FedAvgCNN"]
