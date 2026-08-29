from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch import optim

from federated_outlier_adaptation import config
from federated_outlier_adaptation.providers import default_provider
from federated_outlier_adaptation.training.combined_model_loop import global_outliers_training

base_learning_rate = 1e-3
base_weight_decay = 1e-4


"""
Combined reference model for federated adaptive learning.

This script orchestrates training of a global model that incorporates both
global clients and the selected participant clients. It sets up model,
optimizer, paths, and dataset splits, and delegates the actual training loop to
`global_outliers_training`.

Main components:
    - GlobalClientsTraining: class wrapping initialization and training
    - Obtains clients, loaders and the model topology from the dataset provider
    - Stores results and model checkpoints under the provider's results root

Every dataset access goes through the provider, so the module serves NIST, LEAF
any provider alike.  With the default provider it reads the frozen
`writer_split.json` and `selected_outliers.json` exactly as before and writes
the published file names.
"""


class GlobalClientsTraining:
    """
    Wrapper for training a global model on NIST data, combining global writers
    with selected outlier clients.

    Responsibilities:
        - Initialize model, optimizer, and loss function
        - Manage experiment paths and JSON writer splits
        - Load selected outliers and global writer IDs
        - Run training using global_outliers_training

    Attributes:
        selected_outliers (list): IDs of selected outlier clients.
        global_writers (list): IDs of global writers.
        local_writers (list): IDs of local writers.
        model (torch.nn.Module): The global CNN model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        results_path (Path): Directory for experiment results.
        model_path (Path): Directory for saving model checkpoints.
    """

    def __init__(self, provider=None, results_dir: Optional[Path] = None, name: str = "global_clients"):
        """
        Args:
            name: Prefix of the two output folders.  ``global_clients`` (the
                default) is the published one; the pooled-oracle ceilings of
                the reference ladder pass one name per severity so that three
                ceilings can live in one results root.
        """
        self.selected_outliers = None
        self.global_writers = None
        self.local_writers = None
        self.metrics = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.provider = provider or default_provider()
        self.results_dir = Path(results_dir) if results_dir else Path(
            getattr(self.provider, "results_dir", config.RESULTS_DIR)
        )
        self.model = self.provider.make_model()
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=base_learning_rate, weight_decay=base_weight_decay)
        self.learning_rate = base_learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.name = str(name)
        self.results_path = self.results_dir / f"{self.name}_results"
        self.model_path = self.results_dir / f"{self.name}_model"
        self.writer_split_path = self.results_dir / "writer_split.json"
        self.selected_outliers_path = self.results_dir / "outliers" / "selected_outliers.json"

        self.path_init()
        self.get_writers_split()
    def path_init(self):
        """
        Initialize result directories for the project.

        Creates the following if they do not exist:
            - results/global_clients_results/
            - results/global_clients_model/
            - results/outliers/
            - results/writer_split.json (expected file)
        """

        self.results_path.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "outliers").mkdir(parents=True, exist_ok=True)


    def get_writers_split(self):
        """
        Load the client split.

        The provider is the source of truth; for NIST it reads exactly the
        frozen ``writer_split.json`` this method used to open directly, so the
        result is unchanged.

        Assigns:
            - self.local_writers
            - self.global_writers
        """

        self.local_writers = self.provider.local_client_ids()
        self.global_writers = self.provider.global_client_ids()

    def get_selected_outliers(self, k: int = 5, clients_file=None):
        """
        Load the IDs of the selected participant clients.

        For NIST the provider returns the frozen ``selected_outliers.json``;
        the additional datasets return their ranked selection.  ``clients_file``
        overrides both with an explicit list or a rule-based pool file, which is
        how the pooled oracle is pointed at one severity's client pool.
        """
        if clients_file:
            from federated_outlier_adaptation.outliers.selection import load_client_pool

            self.selected_outliers, _ = load_client_pool(clients_file)
            return

        self.selected_outliers = self.provider.selected_clients(k=k)


    def train(self, epochs, batch_size=64, patience=None, min_epochs=0):
        """
        Train global model with global writers and selected outliers.

        Builds datasets for:
            - Joint training set of global writers + outliers
            - Validation set
            - Global writers test set
            - Outliers test set

        Delegates the training loop to `global_outliers_training`.

        Args:
            epochs (int): Maximum number of training epochs.
            patience (int, optional): Non-improving epochs tolerated before the
                training stops; ``None`` runs every epoch, as published.
            min_epochs (int): Epochs that always run before patience may fire.
        """

        clients_outliers = self.selected_outliers + self.global_writers
        train_loader, eval_loader, _ = self.provider.build_training_dataset(clients_outliers, train_rate=0.6,
                                                                            eval_rate=0.4, batch_size=batch_size)
        _, _, global_test_loader = self.provider.build_dataset(self.global_writers, train_rate=0.0, eval_rate=0.0,
                                                               batch_size=batch_size)
        _, _, all_outliers_test_loader = self.provider.build_dataset(self.selected_outliers, train_rate=0.0,
                                                                     eval_rate=0.0, batch_size=batch_size)
        global_outliers_training(self.model, train_loader, eval_loader, all_outliers_test_loader, global_test_loader,
                                 self.model_path, self.results_path,  num_epochs=epochs,
                                 patience=patience, min_epochs=min_epochs)


def run_combined_training(
    epochs: int = 100,
    batch_size: int = 64,
    provider=None,
    k: int = 5,
    clients_file=None,
    name: str = "global_clients",
    patience=None,
    min_epochs: int = 0,
):
    """
    Phase 2 of the pipeline: global clients plus the selected participants.

    This is also the **pooled oracle** of the reference ladder - everything a
    centralised trainer could do if the privacy constraint did not exist - so
    ``clients_file`` and ``name`` let one results root hold one ceiling per
    client pool instead of a single ``global_clients`` model.
    """
    gct = GlobalClientsTraining(provider=provider, name=name)
    gct.get_selected_outliers(k=k, clients_file=clients_file)
    gct.train(
        epochs=epochs, batch_size=batch_size, patience=patience, min_epochs=min_epochs
    )
    return gct


def main(argv=None):
    """
    Command-line entry point.

    Without arguments it runs the published NIST phase 2, exactly as before.
    """
    import argparse

    from federated_outlier_adaptation.providers import get_provider

    parser = argparse.ArgumentParser(
        description="Train the combined reference model (global clients + participants)."
    )
    parser.add_argument("--provider", default="nist", help="nist")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    kwargs = {}
    if args.results_dir:
        kwargs["results_dir"] = Path(args.results_dir)
    if args.data_dir:
        kwargs["data_dir"] = Path(args.data_dir)
    provider = get_provider(args.provider, **kwargs) if (args.provider != "nist" or kwargs) else None

    run_combined_training(
        epochs=args.epochs, batch_size=args.batch_size, provider=provider, k=args.k
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

