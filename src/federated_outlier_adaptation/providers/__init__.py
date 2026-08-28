"""Dataset/model providers decoupling the federated logic from a dataset."""

from federated_outlier_adaptation.providers.base import DatasetProvider
from federated_outlier_adaptation.providers.nist import NistProvider, default_provider
from federated_outlier_adaptation.providers.provenance import provider_config_block
from federated_outlier_adaptation.providers.registry import available_providers, get_provider

__all__ = [
    "DatasetProvider",
    "NistProvider",
    "available_providers",
    "default_provider",
    "get_provider",
    "provider_config_block",
]
