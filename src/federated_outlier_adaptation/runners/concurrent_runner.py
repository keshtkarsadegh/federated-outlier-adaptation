"""
concurrent_runner.py

Purpose:
    Implements the concurrent federated training/evaluation loop
    shared by both:
        - standalone training runs
        - grid search sweeps (EWC, distillation, logit consistency,
          feature alignment, prox, and extreme cases).

Capabilities:
    - Load and evaluate a global baseline model.
    - Train selected outlier clients against the global model concurrently.
    - Aggregate client updates using supplied aggregation methods.
    - Track and store round-by-round accuracy metrics.
    - Record wall-clock timing and communication volume per round.
    - Support flexible scenarios via the `single_outlier` argument.

Key Usage Modes:
    1. Normal training runs:
        Train all selected outliers jointly with the global model.
    2. Grid search:
        Called by sweep scripts to benchmark multiple hyperparameter/aggregation setups.
    3. Extreme cases:
        Use `single_outlier` to restrict training to one or more specific clients
        (e.g., one-client, two-client, or duplicated-client scenarios).

Argument: `single_outlier`
    - None / empty: default, include all selected outliers.
    - list[str]: restrict to specified IDs only.
    - A repeated id yields that many independent participants (own local
      training run, own sample count, own entry in the aggregation), while the
      client evaluation loader stays over the unique ids.
    - Applies in BOTH training and grid search.

Client population options (all default-off):
    - ``participation`` / ``policy`` / ``sampler_seed`` select a subset of the
      participants per round, see :mod:`.client_sampler`.
    - ``track_clients`` records per-client accuracies each round.
    - ``stop_when_global_below_clients`` ends the simulation as soon as the
      global accuracy drops below the clients accuracy or below 0.90.

Outputs:
    - Round-by-round accuracy histories.
    - Metrics JSON files under the configured results root.
    - Accuracy plots and logs.
    - Updated global model after aggregation.
"""

import copy
import json
import time
from pathlib import Path
from typing import Optional

import torch

from federated_outlier_adaptation import config
from federated_outlier_adaptation.aggregation.concurrent_methods import ServerState
from federated_outlier_adaptation.aggregation.selector import accepts_server_state
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.outliers.selection import (
    load_client_pool,
    load_pool_meta,
)
from federated_outlier_adaptation.providers import default_provider
from federated_outlier_adaptation.runners.forgetting_signals import (
    ForgettingSignalTracker,
    build_tracker,
    empty_info,
)
from federated_outlier_adaptation.runners.population import (
    ClientPopulation,
    UntrainableClientError,
    untrainable_message,
)
from federated_outlier_adaptation.trainers import artefacts
from federated_outlier_adaptation.utils import instrumentation
from federated_outlier_adaptation.utils.eval_cache import (
    DEFAULT_EVAL_BATCH,
    POOL_INSAMPLE_SET,
    SOURCE_TEST_SET,
    SOURCE_VAL_SET,
    make_cache,
    resolve_eval_path,
)
from federated_outlier_adaptation.utils.provenance import torch_versions
from federated_outlier_adaptation.utils.seeding import set_run_seed

#: Where the federated loop starts from.
INITIALISATIONS = ("global", "scratch")


def _dataset_of(loader):
    """The dataset behind an evaluation loader, or ``None``."""
    return None if loader is None else loader.dataset


class BaseConcurrentRunner:
    """
    Federated concurrent training runner.

    Provides the shared loop for both grid search and normal training,
    handling:
        - loading models and dataset splits
        - selecting clients (outliers/global)
        - per-client training
        - federated aggregation
        - round-by-round accuracy tracking
        - timing and communication accounting

    Attributes:
        trainer (object): Trainer instance (e.g., EWCTrainer, DistillationTrainer).
        device (str): 'cuda' if available, otherwise 'cpu'.
        provider (DatasetProvider): Source of loaders, model and artefacts.
        global_model (nn.Module): Current global model.
        accuracies (list): History of (all_clients_acc, global_acc) per round.
        client_weights (list): State dicts from trained client models.
        clients_samples_counts (list): Sample counts per client.
        all_clients_test_loader (DataLoader): Evaluation loader for outliers.
        global_test_loader (DataLoader): Evaluation loader for global writers.
        instrumentation (dict): Timing/communication metrics of the last run.

    Note:
        The `single_outlier` argument controls which clients participate:
            - None / empty: include all selected outliers.
            - list[str]: restrict to given IDs, useful for extreme cases.
        This applies equally to training runs and grid search experiments.
    """

    def __init__(
        self,
        trainer,
        single_outlier=None,
        provider=None,
        seed: Optional[int] = None,
        outliers_file: Optional[Path] = None,
        k: int = 5,
        participation: float = 1.0,
        policy: str = "all",
        sampler_seed: Optional[int] = None,
        track_clients: bool = False,
        stop_when_global_below_clients: bool = False,
        clients_per_round: Optional[int] = None,
        pool_frac: Optional[float] = None,
        weighting: str = "proportional",
        server_eta: float = 1.0,
        server_kwargs: Optional[dict] = None,
        source_share: str = "off",
        source_share_cap: Optional[int] = None,
        save_model_path: Optional[str] = None,
        init: str = "global",
        eval_path: Optional[str] = None,
        insample_every: int = 1,
        old_book: Optional[str] = None,
        old_clients_file: Optional[str] = None,
        source_share_multiplier: float = 1.0,
        val_blend_source: float = 0.0,
    ):
        #: Where the per-round source series is measured; see
        #: :mod:`federated_outlier_adaptation.runners.source_series`.
        self.old_book = old_book
        self.old_clients_file = old_clients_file
        self.source_series_info: dict = {}
        self._old_book_cache = None
        self._old_writers_cache = None
        #: How many times the client's own training volume the shared old data
        #: contributes per epoch; see :mod:`..data.source_share`.
        self.source_share_multiplier = float(source_share_multiplier)
        #: Share of the early-stopping criterion drawn from the old book's
        #: validation rows.  0.0 (the default) is the published criterion.
        self.val_blend_source = float(val_blend_source)
        self.old_val_dataset = None
        self.val_blend_info: dict = {}
        self.single_outlier = single_outlier
        self.participation = participation
        self.policy = policy
        self.sampler_seed = sampler_seed
        self.track_clients = track_clients
        self.clients_per_round = clients_per_round
        self.pool_frac = pool_frac
        if init not in INITIALISATIONS:
            raise ValueError(
                f"Unknown initialisation {init!r}; expected one of {INITIALISATIONS}"
            )
        self.init = init
        self.weighting = weighting
        self.server_eta = server_eta
        self.stop_when_global_below_clients = stop_when_global_below_clients
        self.stopped_early = False
        self.population: Optional[ClientPopulation] = None
        self.participants: list = []
        self.round_participants: list = []
        #: Data-sharing upper bound: how much source data joins each client's
        #: training set per epoch.  ``off`` (the default) changes nothing.
        self.source_share = source_share
        self.source_share_cap = source_share_cap
        self.source_train_dataset = None
        self.source_share_info: dict = {"mode": source_share}
        #: Where the final server model is written, when the run is asked for it.
        self.save_model_path = Path(save_model_path) if save_model_path else None
        self.pool_accuracies: dict = {}
        #: Rule and severity of the client pool, when it came from a pool file.
        self.pool_meta: dict = {}
        self.global_clients_all_metrics_acc = None
        self.global_clients_metric_acc = None
        self.global_metrics_acc = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.provider = provider or default_provider()
        self.results_dir = Path(getattr(self.provider, "results_dir", config.RESULTS_DIR))
        self.trainer = trainer
        self.seed = seed
        self.k = k
        self.outliers_file = Path(outliers_file) if outliers_file else None

        self.global_model = None
        #: Persistent state of the extended server-side rules; ignored by the
        #: published stateless aggregation rules.
        #: Coefficients of the extended server rules (FedAvgM's momentum, the
        #: FedOpt learning rate and adaptivity, the trim fraction).  An empty
        #: mapping leaves every published default in place.
        self.server_kwargs = dict(server_kwargs or {})
        self.server_state = ServerState(
            weighting=weighting, eta=server_eta, **self.server_kwargs
        )
        self.accuracies = []
        self.selected_outliers = None
        self.client_weights = []
        self.clients_samples_counts = []
        self.all_clients_test_loader = None
        self.global_test_loader = None

        self.source_val_loader = None
        self.source_val_accuracies: list[float] = []
        #: Constraint-respecting signals of the run; see
        #: :mod:`.forgetting_signals`.  Built once the run's ``theta_g`` is on
        #: the device, so it is ``None`` outside :meth:`simulate`.
        self.signals: Optional[ForgettingSignalTracker] = None
        self.round_seconds: list[float] = []
        self.client_seconds: list[list[float]] = []
        self.comm_bytes_per_round: list[int] = []
        self.comm_bytes_up: list[int] = []
        self.comm_bytes_down: list[int] = []
        self.param_count: Optional[int] = None
        self.instrumentation: dict = {}

        #: ``cache`` materialises every per-round evaluation set once and scores
        #: it in one batched pass; ``loader`` keeps the published per-round
        #: ``DataLoader`` passes and is what the equality check compares to.
        self.eval_path = resolve_eval_path(eval_path)
        self.insample_every = max(1, int(insample_every))
        self.eval_cache = None
        self.eval_cache_info: dict = {}
        #: Most recent in-sample measurement, repeated on the rounds that
        #: ``--insample-every`` skips so the series stays aligned.
        self._last_insample: Optional[float] = None

    # ------------------------------------------------------------ evaluation
    def prepare_eval_cache(self, batch_size: int) -> None:
        """
        Materialise the run's evaluation sets, unless the loader path was asked for.

        Registers the source split, the pool's in-sample data and - through the
        population - the pool's per-client and held-out sets.  Every set that
        cannot be materialised is simply absent and keeps its loader, so the
        cache can never change what a run computes.
        """
        self.eval_cache = make_cache(
            self.eval_path, device=self.device, batch_size=DEFAULT_EVAL_BATCH
        )
        if self.eval_cache is None:
            self.eval_cache_info = {"eval_path": self.eval_path}
            return
        self.eval_cache.add(SOURCE_TEST_SET, _dataset_of(self.global_test_loader))
        self.eval_cache.add(POOL_INSAMPLE_SET, _dataset_of(self.all_clients_test_loader))
        self.eval_cache.add(SOURCE_VAL_SET, _dataset_of(self.source_val_loader))
        self.population.prepare_cache(self.eval_cache, batch_size)
        self.eval_cache_info = {"eval_path": self.eval_path, **self.eval_cache.summary()}

    def _accuracy(self, name: str, loader):
        """
        This round's accuracy on a set: cached pass when there is one.

        A set the run does not have - the source series of a results root with
        neither an old-data book nor a writer split - is ``None`` rather than a
        crash, so the series is simply absent from the record.
        """
        if self.eval_cache is not None and self.eval_cache.has(name):
            return self.eval_cache.accuracy(name)
        if loader is None:
            return None
        return self.trainer.evaluate(loader)

    def get_global_base_accuracy(self):
        """
        Load baseline test accuracies from saved results.

        Reads JSON files under the results root:
            - global_results/global_metrics.json
            - global_clients_results/global_clients_global_metrics.json
            - global_clients_results/global_clients_all_outliers_metrics.json

        Sets attributes:
            - self.global_metrics_acc
            - self.global_clients_metric_acc
            - self.global_clients_all_metrics_acc
        """
        global_clients_results_path = self.results_dir / "global_clients_results"
        global_results_path = self.results_dir / "global_results"
        global_metrics_file = global_results_path / "global_metrics.json"

        global_clients_metric_file = global_clients_results_path / "global_clients_global_metrics.json"
        global_clients_all_metrics_file = (
            global_clients_results_path / "global_clients_all_outliers_metrics.json"
        )

        def read_test_accuracy(file_path):
            """
            The stored test accuracy, or ``None`` when the file is not there.

            These three numbers are *reference lines* - the frozen global model
            and the pooled-oracle ceiling - drawn on the run's plot.  A run that
            has not had its oracle trained yet is still a valid run, so a
            missing file is a warning and a ``None``, not the end of a training
            job that has already spent its GPU-hours.
            """
            file_path = Path(file_path)
            if not file_path.is_file():
                NistLogger.warning(
                    f"Reference accuracy {file_path.name} is missing under "
                    f"{file_path.parent}; the run is unaffected and its "
                    "reference line is left empty. Run 'foa combined-train' in "
                    "this results root to produce it."
                )
                return None
            with open(file_path, "r") as f:
                data = json.load(f)
            return data.get("test_accuracy")

        if (
            self.global_clients_all_metrics_acc is None
            or self.global_clients_metric_acc is None
            or self.global_metrics_acc is None
        ):
            self.global_metrics_acc = read_test_accuracy(global_metrics_file)
            self.global_clients_metric_acc = read_test_accuracy(global_clients_metric_file)
            self.global_clients_all_metrics_acc = read_test_accuracy(global_clients_all_metrics_file)

    def load_global_model(self, global_name):
        """
        Load a pre-trained global model from disk.

        Args:
            global_name (str): Name prefix of the saved model file.

        Returns:
            nn.Module: Model loaded onto the correct device.
        """
        model = self.provider.make_model()
        if self.init == "scratch":
            # Reference point: federated training from a fresh model, so the
            # experiment measures what the federation learns rather than what
            # it preserves.
            NistLogger.info("Starting from a freshly initialised model (init=scratch).")
            return model.to(self.device)
        path = self.results_dir / f"{global_name}_model"
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        return model

    def get_writers_split(self):
        """
        Load the client split (local vs global writers), when there is one.

        A results root without ``writer_split.json`` returns two empty lists
        rather than failing: the split is only consulted by the per-round source
        series, which
        :func:`~federated_outlier_adaptation.runners.source_series.source_loaders`
        now resolves for itself, and a v6 root has no split to read.

        Returns:
            tuple[list[str], list[str]]: (local_writers, global_writers)
        """
        try:
            return self.provider.local_client_ids(), self.provider.global_client_ids()
        except FileNotFoundError:
            return [], []

    def get_selected_outliers(self):
        """
        Load IDs of the selected outlier clients.

        Returns:
            tuple[list[str], list]: (selected_outliers, empty placeholder list)
        """
        if self.outliers_file is not None:
            clients, self.pool_accuracies = load_client_pool(self.outliers_file)
            self.pool_meta = load_pool_meta(self.outliers_file)
            return clients, []
        if self.pool_frac is not None:
            return list(self.provider.outlier_pool(self.pool_frac)), []
        return self.provider.selected_clients(k=self.k), []

    def get_test_loaders(self, selected_outliers, global_writers, batch_size):
        """
        Build evaluation DataLoaders for the selected outliers and the source set.

        ``global_writers`` is accepted for compatibility and no longer consulted:
        where the source series is measured is decided by
        :func:`~federated_outlier_adaptation.runners.source_series.source_loaders`,
        which follows the old-data book when the run has one, the frozen writer
        split when it does not, and returns nothing when there is neither.

        Args:
            selected_outliers (list[str]): Outlier IDs.
            global_writers (list[str]): Ignored; see above.
            batch_size (int): Batch size for evaluation.

        Returns:
            tuple[DataLoader, DataLoader]: (all_clients_loader, source_loader)
        """
        from federated_outlier_adaptation.runners.source_series import source_loaders
        from federated_outlier_adaptation.utils.seeding import configured_fold

        all_clients_loader = self.provider.build_dataset(
            selected_outliers,
            train_rate=0.0,
            eval_rate=0.0,
            batch_size=batch_size,
            loader_seed=self.seed,
        )[2]
        global_loader, self.source_val_loader, self.source_series_info = source_loaders(
            self.provider,
            batch_size=batch_size,
            loader_seed=self.seed,
            old_book=self.old_book,
            old_clients_file=self.old_clients_file,
            fold=configured_fold(),
        )
        return all_clients_loader, global_loader

    def _share_source(self, train_loader, writer, round_index, batch_size):
        """
        Mix the source share into one client's training loader.

        Returns the loader unchanged when sharing is off or there is nothing to
        draw from, so the default path is bit-for-bit what it was.
        """
        if not self.source_share or self.source_share == "off":
            return train_loader
        if train_loader is None or self.source_train_dataset is None:
            return train_loader

        from federated_outlier_adaptation.data.source_share import (
            build_shared_loader,
            sampler_seed,
        )

        shared, info = build_shared_loader(
            train_loader.dataset,
            self.source_train_dataset,
            mode=self.source_share,
            batch_size=batch_size,
            seed=sampler_seed(self.seed, round_index, str(writer)),
            cap=self.source_share_cap,
            multiplier=self.source_share_multiplier,
        )
        self.source_share_info = info
        return shared if shared is not None else train_loader

    def aggregate(self, aggregate_method):
        """
        Aggregate client weights into the global model.

        Args:
            aggregate_method (callable): Function with signature
                (global_weights, client_weights, clients_samples_counts)
                -> aggregated_state_dict

        Updates:
            self.global_model with aggregated weights.
        """
        global_weights = {
            k: v for k, v in self.global_model.state_dict().items() if v.dtype.is_floating_point
        }

        client_weights = self.client_weights  # Already filtered when stored

        if accepts_server_state(aggregate_method):
            # Server-side optimisers (FedAvgM, FedAdam, FedYogi) carry momentum
            # and second-moment buffers across rounds; the published stateless
            # rules do not accept the keyword and are called unchanged.
            aggregated = aggregate_method(
                global_weights,
                client_weights,
                self.clients_samples_counts,
                server_state=self.server_state,
            )
        else:
            aggregated = aggregate_method(
                global_weights, client_weights, self.clients_samples_counts
            )

        self.global_model.load_state_dict(aggregated, strict=False)

    def run_round(self, batch_size, epochs, round_index: int = 0, is_last: bool = False):
        """
        Run a single round of federated training.

        Steps:
            - Evaluate current global model on global and outlier test sets.
            - Optionally evaluate every pool client on its own data.
            - Train the participants drawn for this round (copy of global model).
            - Collect trained weights and sample counts.
            - Append round accuracies to history.

        Args:
            batch_size (int): Training batch size.
            epochs (int): Epochs per client.
            round_index (int): Zero-based round counter, passed to the sampler.

        Returns:
            bool: True if training should continue, False if stopped early.
        """
        round_t0 = time.perf_counter()
        self.trainer.set_model(self.global_model)
        if self.eval_cache is not None:
            # One bound model per round: every set is scored at most once and
            # the pooled numbers, the per-client numbers and the signals all
            # read the same pass.
            self.eval_cache.bind(self.trainer.get_model())
        measure_insample = self.population.measures_insample(round_index, is_last)
        if measure_insample:
            self.population.insample_rounds.append(int(round_index))

        aggr_global_acc = self._accuracy(SOURCE_TEST_SET, self.global_test_loader)
        if measure_insample:
            aggr_all_clients_acc = self._accuracy(
                POOL_INSAMPLE_SET, self.all_clients_test_loader
            )
            self._last_insample = aggr_all_clients_acc
        else:
            aggr_all_clients_acc = self._last_insample

        self.accuracies.append((aggr_all_clients_acc, aggr_global_acc))

        # Recorded before the oracle stop rule fires, so the signal series stay
        # index-aligned with ``accuracies`` even in a run that stops early.
        if self.signals is not None:
            self.signals.record(self.global_model, self.population)

        # The oracle stop rule needs a source number to compare against; a run
        # measuring no source series simply never fires it.
        if (
            self.stop_when_global_below_clients
            and aggr_global_acc is not None
            and (
                (aggr_all_clients_acc is not None and aggr_global_acc < aggr_all_clients_acc)
                or aggr_global_acc < 0.90
            )
        ):
            clients_text = (
                "unmeasured" if aggr_all_clients_acc is None else f"{aggr_all_clients_acc:.4f}"
            )
            NistLogger.info(
                "Stopping: global accuracy "
                f"{aggr_global_acc:.4f} fell below the clients accuracy "
                f"{clients_text} or below 0.90."
            )
            self.stopped_early = True
            self.client_seconds.append([])
            self.round_seconds.append(time.perf_counter() - round_t0)
            return False

        if self.source_val_loader is not None:
            source_val = self._accuracy(SOURCE_VAL_SET, self.source_val_loader)
            if source_val is not None:
                self.source_val_accuracies.append(float(source_val))

        self.population.track(self.trainer.evaluate, batch_size, measure=measure_insample)
        self.round_participants = self.population.select(round_index)
        self.population.evaluate_pool(self.trainer.evaluate, batch_size, self.round_participants)

        self.client_weights.clear()
        self.clients_samples_counts.clear()

        per_client_seconds: list[float] = []
        for writer in self.round_participants:
            client_t0 = time.perf_counter()
            trn, evl, _ = self.provider.build_dataset(
                writer,
                train_rate=0.6,
                eval_rate=0.4,
                batch_size=batch_size,
                loader_seed=self.seed,
            )
            if trn is None or evl is None or not len(trn.dataset) or not len(evl.dataset):
                raise UntrainableClientError(
                    untrainable_message(writer, trn, evl, self.provider)
                )
            trn = self._share_source(trn, writer, round_index, batch_size)
            evl = self._blend_validation(evl, writer, batch_size)
            sample_count = self.provider.sample_count(writer)
            self.clients_samples_counts.append(sample_count)

            client_model = copy.deepcopy(self.global_model)
            self.trainer.set_model(client_model)
            self.trainer.train(trn, evl, epochs)

            trained = self.trainer.get_model()
            float_state_dict = {
                k: v for k, v in trained.state_dict().items() if v.dtype.is_floating_point
            }
            self.client_weights.append(copy.deepcopy(float_state_dict))
            per_client_seconds.append(time.perf_counter() - client_t0)

        if self.client_weights:
            payload_bytes = instrumentation.state_dict_bytes(self.client_weights[0])
            self.param_count = instrumentation.state_dict_params(self.client_weights[0])
            participants = len(self.client_weights)
            self.comm_bytes_per_round.append(
                instrumentation.round_communication_bytes(payload_bytes, participants)
            )
            # Down: the server ships the global model to every participant.
            # Up: every participant returns its update.
            self.comm_bytes_down.append(payload_bytes * participants)
            self.comm_bytes_up.append(payload_bytes * participants)

        self.client_seconds.append(per_client_seconds)
        self.round_seconds.append(time.perf_counter() - round_t0)
        return True

    def _collect_instrumentation(self):
        versions = torch_versions()
        self.instrumentation = {
            **instrumentation.summarise_timing(self.round_seconds, self.client_seconds),
            "comm_bytes_per_round": list(self.comm_bytes_per_round),
            **instrumentation.communication_summary(self.comm_bytes_up, self.comm_bytes_down),
            "param_count": self.param_count,
            "device": instrumentation.device_name(versions["device"], versions["gpu_name"]),
            "seed": self.seed,
        }
        return self.instrumentation


    # ----------------------------------------------------- old-data access
    def _old_book(self):
        """The old-data fold book of this run, or ``None``."""
        if self._old_book_cache is None and self.old_book:
            from pathlib import Path as _Path

            from federated_outlier_adaptation.data.fold_book import FoldBook

            path = _Path(self.old_book)
            if not path.is_file():
                raise FileNotFoundError(
                    f"No old-data fold book at {path}. The access-to-old-data "
                    "arms read the rows they may share off it."
                )
            self._old_book_cache = FoldBook.load(path)
        return self._old_book_cache

    def _old_writers(self):
        """The old-data writers of this run, or ``None`` for the whole book."""
        if self._old_writers_cache is None and self.old_clients_file:
            from federated_outlier_adaptation.outliers.selection import load_client_pool

            self._old_writers_cache = list(load_client_pool(self.old_clients_file)[0])
        return self._old_writers_cache

    def _old_part_pool(self, part: str, batch_size: int):
        """One part of the old book's fold-k rows, as a dataset."""
        from federated_outlier_adaptation.data.source_share import (
            old_book_rows,
        )
        from federated_outlier_adaptation.training.evaluate import rows_loader
        from federated_outlier_adaptation.utils.seeding import configured_fold

        book = self._old_book()
        if book is None:
            return None
        fold = configured_fold()
        if fold is None:
            # Without a fold there is no "fold k of the old data" to pair with
            # the client's split; fold 1 is recorded rather than guessed at.
            fold = 1
        writers = self._old_writers() or list(book.writers)
        loader = rows_loader(
            self.provider, old_book_rows(book, writers, fold, part), batch_size
        )
        return None if loader is None else loader.dataset

    def _old_train_pool(self, batch_size: int):
        """
        The rows a client may train on: the old book's fold-k TRAIN partition.

        Never its validation or test rows - those are the preservation
        measurement, and a row that is trained on and then scored as retained
        knowledge measures nothing.  A results root with no old book falls back
        to the legacy 3% source pool, so v4 and v5 runs are unchanged.
        """
        if self.old_book:
            return self._old_part_pool("train", batch_size)
        from federated_outlier_adaptation.data.source_share import source_pool_dataset

        return source_pool_dataset(
            self.provider, batch_size=batch_size, loader_seed=self.seed
        )

    def _old_val_pool(self, batch_size: int):
        """The old book's fold-k VALIDATION rows, for the monitoring-only arm."""
        return self._old_part_pool("val", batch_size)

    def _blend_validation(self, eval_loader, writer, batch_size):
        """
        Blend the old data's validation rows into one client's criterion.

        Monitoring, not training: the model is allowed to look at the old
        population when deciding whether it has finished an epoch, and never to
        take a gradient step on it.
        """
        if not self.val_blend_source or self.old_val_dataset is None:
            return eval_loader
        from federated_outlier_adaptation.data.source_share import (
            blended_val_loader,
            sampler_seed,
        )

        blended, info = blended_val_loader(
            eval_loader,
            self.old_val_dataset,
            rho=self.val_blend_source,
            batch_size=batch_size,
            # Per client and stable across rounds: the criterion a client is
            # judged by must not move underneath it between rounds.
            seed=sampler_seed(self.seed, 0, str(writer)),
        )
        self.val_blend_info = info
        return blended if blended is not None else eval_loader

    def population_info(self) -> dict:
        """Additive payload keys describing the client population of the run."""
        info = dict(self.population.info() if self.population is not None else {})
        info["source_val_accuracies"] = list(self.source_val_accuracies)
        # Where the per-round source series came from, and that it is one.
        info["source_series"] = dict(self.source_series_info)
        info.update(self.signals.info() if self.signals is not None else empty_info())
        info["evaluation"] = dict(self.eval_cache_info)
        # The server coefficients a swept rule was run with; the published
        # stateless rules ignore them and report the defaults.
        info["server"] = self.server_state.hyperparameters()
        info["source_share"] = dict(self.source_share_info)
        info["val_blend"] = dict(self.val_blend_info)
        info["pool"] = dict(self.pool_meta)
        if self.stop_when_global_below_clients:
            info = dict(info)
            info["stopped_early"] = self.stopped_early
        return info


    def _signal_fisher_dir(self, global_name):
        """
        The Fisher directory the SIGNALS should read.

        A run may name a Fisher explicitly - the per-fold directory belonging to
        the fold's own shipped model - and when it does, that is the only file
        whose curvature matches the anchor the signals measure distance from.
        The trainer already honours it; this makes the observation agree with
        the thing it observes. Without this the tracker silently recorded
        nothing while the penalty worked, which is the worst of both.
        """
        explicit = getattr(getattr(self, "trainer", None), "fisher_path", None)
        if explicit and Path(explicit).is_dir():
            return Path(explicit)
        return artefacts.fisher_dir(self.provider, global_name)

    def simulate(
        self,
        exp_name,
        global_name,
        aggregate_method,
        batch_size=64,
        epochs=100,
        max_round=100,
        grid_Search=False,
    ):
        """
        Simulate federated concurrent training over multiple rounds.

        Args:
            exp_name (str): Experiment name (used for output directories).
            global_name (str): Identifier for loading global model weights.
            aggregate_method (callable): Function to aggregate client weights.
            batch_size (int, optional): Training batch size. Default=64.
            epochs (int, optional): Client epochs per round. Default=100.
            max_round (int, optional): Maximum federated rounds. Default=100.
            grid_Search (bool, optional): Whether called inside a grid search.

        Returns:
            - If grid_Search=False:
                tuple: (accuracy_history, global_clients_all_metrics_acc,
                        global_clients_metric_acc, results root)
            - If grid_Search=True:
                tuple: (accuracy_history, results root)
        """
        set_run_seed(self.seed)
        self.server_state.reset()
        selected, self.accuracies = self.get_selected_outliers()
        self.population = ClientPopulation(
            provider=self.provider,
            selected=selected,
            single_outlier=self.single_outlier,
            participation=self.participation,
            clients_per_round=self.clients_per_round,
            policy=self.policy,
            sampler_seed=self.sampler_seed if self.sampler_seed is not None else self.seed,
            track_clients=self.track_clients,
            loader_seed=self.seed,
            insample_every=self.insample_every,
        )
        self.selected_outliers = self.population.unique_clients
        self.participants = self.population.participants

        local_writers, global_writers = self.get_writers_split()
        self.all_clients_test_loader, self.global_test_loader = self.get_test_loaders(
            self.selected_outliers, global_writers, batch_size
        )
        self.global_model = self.load_global_model(global_name)
        # The anchored server rule pulls back towards the model the run started
        # from; every other rule ignores it.
        self.server_state.frozen_global = {
            key: value.detach().clone()
            for key, value in self.global_model.state_dict().items()
            if value.dtype.is_floating_point
        }
        self.prepare_eval_cache(batch_size)
        if self.source_share and self.source_share != "off":
            # The data-sharing upper bound: the OLD DATA's fold-k *training*
            # rows, built once and drawn from by every client of every round.
            # The old book's validation and test rows are the preservation
            # measurement and never enter a training set; see
            # :mod:`..data.source_share`.
            self.source_train_dataset = self._old_train_pool(batch_size)
            pool = 0 if self.source_train_dataset is None else len(self.source_train_dataset)
            if not pool:
                raise ValueError(
                    f"--source-share {self.source_share} was asked for, but "
                    "there is no old training data to share. Pass --old-book "
                    "(and --old-clients-file), or check the writer split of "
                    "this results root."
                )
            NistLogger.info(
                f"Source sharing '{self.source_share}' x{self.source_share_multiplier:g}: "
                f"a pool of {pool} old training images joins every client's epoch."
            )
        if self.val_blend_source:
            self.old_val_dataset = self._old_val_pool(batch_size)
            size = 0 if self.old_val_dataset is None else len(self.old_val_dataset)
            if not size:
                raise ValueError(
                    "--val-blend-source was asked for, but there are no old "
                    "validation rows to blend in. Pass --old-book."
                )
            NistLogger.info(
                f"Validation blend {self.val_blend_source:g}: {size} old "
                "validation images are available to the early-stopping "
                "criterion. No old row enters a training set."
            )

        # Observation only: the tracker never feeds anything back into the loop.
        self.signals = build_tracker(
            provider=self.provider,
            model=self.global_model,
            device=self.device,
            fisher_dir=self._signal_fisher_dir(global_name),
            batch_size=batch_size,
            cache=self.eval_cache,
        )
        # Anchoring theta_g before the loop keeps the reference pass out of the
        # first round's timing and lets the proxy set join the cache.
        self.signals.prepare(self.population)
        for r in range(max_round):
            NistLogger.debug(f"Training round {r}")
            should_continue = self.run_round(
                batch_size, epochs, round_index=r, is_last=(r == max_round - 1)
            )
            if not should_continue:
                break
            self.aggregate(aggregate_method)
        self._collect_instrumentation()
        if self.save_model_path is not None:
            self.save_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.global_model.state_dict(), self.save_model_path)
            NistLogger.info(f"Saved the final server model to {self.save_model_path}")

        if not grid_Search:
            self.get_global_base_accuracy()
            return (
                self.accuracies,
                self.global_clients_all_metrics_acc,
                self.global_clients_metric_acc,
                self.results_dir,
            )
        return self.accuracies, self.results_dir
