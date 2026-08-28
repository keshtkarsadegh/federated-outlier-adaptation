"""
The corrected setting of experiment plan v4.

Five behaviours are pinned here, each of them a defect the plan names:

* the FedAvg CNN of McMahan et al. is the model, and it is the published
  topology down to its parameter count;
* ``--aggregation fedavg`` runs **two** families, not four of which two are
  algebraic duplicates, and the sweeps use the very same two rules;
* the constrained selection is an argmax over seed means, not over single runs;
* every anchored space has a ``lam = 0`` arm that is exactly plain training;
* the server coefficients the literature grids sweep - FedAvgM's momentum, the
  FedOpt learning rate and adaptivity, the trim fraction, the cyclic mixing
  weight - are parameters and not constants.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from federated_outlier_adaptation.aggregation.concurrent_methods import (
    ServerState,
    con_delta_fedavgm,
    con_delta_trimmed_mean,
)
from federated_outlier_adaptation.aggregation.selector import (
    AGG_VARIANTS,
    FEDAVG_CONCURRENT_FAMILY,
    FEDAVG_CONCURRENT_RULE,
    FEDAVG_SEQUENTIAL_FAMILY,
    FEDAVG_SEQUENTIAL_RULE,
    SKIP_FAMILY,
    resolve_aggregation,
)
from federated_outlier_adaptation.aggregation.sequential_methods import (
    SequentialWeightsAGGMETHODS,
    seq_fedavg_update,
    seq_mix_alpha,
)
from federated_outlier_adaptation.cli import build_parser
from federated_outlier_adaptation.grid_search.common import CAPPED_TASKS, WEIGHTED_TASKS
from federated_outlier_adaptation.grid_search.selection import (
    Record,
    seed_mean_records,
    select_seed_mean,
)
from federated_outlier_adaptation.model import FlexibleCNN
from federated_outlier_adaptation.models.fedavg_cnn import FedAvgCNN
from federated_outlier_adaptation.trainers.anchored_trainer import (
    ANCHOR_SPACES,
    AnchoredTrainer,
)

BATCH = 4


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def test_the_published_parameter_count_is_reproduced():
    """
    McMahan et al. state 1,663,370 parameters for the ten-class MNIST model.
    Reproducing the number exactly is what pins the padding and therefore the
    flatten size, so it is worth asserting rather than describing.
    """
    model = FedAvgCNN(num_classes=10, input_size=28)
    assert sum(p.numel() for p in model.parameters()) == 1_663_370


def test_the_62_class_head_is_the_only_difference():
    ten = sum(p.numel() for p in FedAvgCNN(num_classes=10).parameters())
    sixty_two = sum(p.numel() for p in FedAvgCNN(num_classes=62).parameters())
    assert sixty_two - ten == 512 * 52 + 52


def test_the_topology_has_no_normalisation_and_no_dropout():
    model = FedAvgCNN(num_classes=62)
    kinds = {type(module).__name__ for module in model.modules()}
    assert not any(name.startswith("BatchNorm") for name in kinds)
    assert "Dropout" not in kinds


def test_penultimate_returns_the_512_features():
    model = FedAvgCNN(num_classes=62)
    images = torch.rand(BATCH, 1, 28, 28)
    features = model.penultimate(images)
    assert features.shape == (BATCH, 512)
    assert (features >= 0).all()  # it is the ReLU activation


def test_forward_can_return_the_features_too():
    model = FedAvgCNN(num_classes=62)
    logits, features = model(torch.rand(BATCH, 1, 28, 28), return_features=True)
    assert logits.shape == (BATCH, 62)
    assert features.shape == (BATCH, 512)


def test_the_parameter_groups_split_body_and_head():
    model = FedAvgCNN(num_classes=62)
    body = {id(p) for p in model.body_parameters()}
    head = {id(p) for p in model.head_parameters()}
    assert not body & head
    assert body | head == {id(p) for p in model.parameters()}
    model.freeze_body()
    assert all(not p.requires_grad for p in model.body_parameters())
    assert all(p.requires_grad for p in model.head_parameters())


def test_the_ablation_model_works_at_28x28():
    """FlexibleCNN is kept as the architecture ablation, so it has to run here."""
    model = FlexibleCNN(num_classes=62, input_size=28)
    logits = model(torch.rand(BATCH, 1, 28, 28))
    assert logits.shape == (BATCH, 62)
    assert model.penultimate(torch.rand(BATCH, 1, 28, 28)).shape == (BATCH, 128)


def test_the_ablation_model_is_unchanged_at_128():
    assert FlexibleCNN(num_classes=10).flatten_size == FlexibleCNN(
        num_classes=10, input_size=128
    ).flatten_size


# --------------------------------------------------------------------------- #
# two families, one rule per schedule, in sweeps and in finals
# --------------------------------------------------------------------------- #
def test_the_fedavg_variant_runs_two_families():
    variant = AGG_VARIANTS["fedavg"]
    running = {family: rule for family, rule in variant.items() if rule != SKIP_FAMILY}
    assert running == {
        FEDAVG_CONCURRENT_FAMILY: FEDAVG_CONCURRENT_RULE,
        FEDAVG_SEQUENTIAL_FAMILY: FEDAVG_SEQUENTIAL_RULE,
    }


def test_the_duplicate_families_are_skipped():
    assert resolve_aggregation("concurrent", "weights", "fedavg") == SKIP_FAMILY
    assert resolve_aggregation("sequential", "delta", "fedavg") == SKIP_FAMILY


def test_the_sweep_runs_the_rule_the_final_runs():
    swept = {(scenario, metadata): rule for scenario, metadata, _, rule in WEIGHTED_TASKS}
    final = {
        family: rule
        for family, rule in AGG_VARIANTS["fedavg"].items()
        if rule != SKIP_FAMILY
    }
    assert swept == final


def test_the_capped_task_list_is_still_available():
    """It reproduces the published sweep and is now reachable only on request."""
    assert len(CAPPED_TASKS) == 4
    assert WEIGHTED_TASKS != CAPPED_TASKS


def test_the_grid_command_opts_into_the_capped_tasks():
    args = build_parser().parse_args(["grid", "--method", "kd", "--capped-tasks"])
    assert args.capped_tasks is True
    assert build_parser().parse_args(["grid", "--method", "kd"]).capped_tasks is False


# --------------------------------------------------------------------------- #
# selection on the seed mean
# --------------------------------------------------------------------------- #
def _record(name, adaptation, forgetting, seed, **kwargs):
    config = {
        "scenario": "concurrent",
        "metadata": "delta",
        "agg_method_name": FEDAVG_CONCURRENT_RULE,
        "seed": seed,
        "trainer_kwargs": dict(kwargs),
    }
    return Record(
        name=name,
        source=f"{name}/seed{seed}",
        adaptation=adaptation,
        forgetting=forgetting,
        source_reference=0.9,
        source_final=0.9 - forgetting,
        config=config,
    )


def test_runs_of_one_configuration_collapse_into_their_mean():
    groups = seed_mean_records(
        [
            _record("lam1", 0.80, 0.001, seed=1, lam=1),
            _record("lam1", 0.90, 0.003, seed=2, lam=1),
        ]
    )
    assert len(groups) == 1
    assert groups[0].adaptation == pytest.approx(0.85)
    assert groups[0].forgetting == pytest.approx(0.002)
    assert groups[0].num_seeds == 2
    assert groups[0].seeds == [1, 2]


def test_different_configurations_stay_apart():
    groups = seed_mean_records(
        [
            _record("lam1", 0.80, 0.001, seed=1, lam=1),
            _record("lam10", 0.80, 0.001, seed=1, lam=10),
        ]
    )
    assert len(groups) == 2


def test_the_luckiest_single_run_does_not_win():
    """
    ``lam10`` holds the single best run (0.95) but is worse on average; the
    selection has to pick ``lam1``.  This is the "selection on noise" defect.
    """
    records = [
        _record("lam1", 0.86, 0.001, seed=1, lam=1),
        _record("lam1", 0.88, 0.001, seed=2, lam=1),
        _record("lam10", 0.95, 0.001, seed=1, lam=10),
        _record("lam10", 0.70, 0.001, seed=2, lam=10),
    ]
    selected, _, _ = select_seed_mean(records, eps=0.005)
    assert selected.name == "lam1"
    assert selected.adaptation == pytest.approx(0.87)


def test_the_constraint_is_applied_to_the_mean_forgetting():
    """
    A configuration that is within the budget on one seed and far outside it on
    the other is infeasible, because its mean forgetting is what counts.
    """
    records = [
        _record("greedy", 0.99, 0.000, seed=1, lam=0),
        _record("greedy", 0.99, 0.040, seed=2, lam=0),
        _record("safe", 0.80, 0.001, seed=1, lam=1),
        _record("safe", 0.80, 0.001, seed=2, lam=1),
    ]
    selected, feasible, _ = select_seed_mean(records, eps=0.005)
    assert selected.name == "safe"
    assert [group.name for group in feasible] == ["safe"]


def test_a_single_seed_selects_exactly_as_before():
    records = [
        _record("lam1", 0.80, 0.001, seed=1, lam=1),
        _record("lam10", 0.85, 0.001, seed=1, lam=10),
    ]
    selected, _, groups = select_seed_mean(records, eps=0.005)
    assert selected.name == "lam10"
    assert all(group.num_seeds == 1 for group in groups)


# --------------------------------------------------------------------------- #
# the lambda = 0 arm
# --------------------------------------------------------------------------- #
@pytest.fixture()
def batch():
    torch.manual_seed(23)
    return torch.rand(BATCH, 1, 128, 128), torch.tensor([0, 1, 2, 1])


@pytest.mark.parametrize("space", ANCHOR_SPACES)
@pytest.mark.parametrize("anchor", ["frozen", "current"])
def test_lambda_zero_is_exactly_plain_training(synthetic_provider, batch, space, anchor):
    """
    The "penalty off" arm has to be plain cross-entropy in every space, so a
    selection at the bottom of a grid can be compared against it honestly.
    """
    images, labels = batch
    trainer = AnchoredTrainer(
        provider=synthetic_provider, space=space, anchor=anchor, lam=0.0
    )
    torch.manual_seed(5)
    trainer.set_model(synthetic_provider.make_model())
    trainer.model.eval()

    logits, features = trainer.model(images, return_features=True)
    total, parts = trainer.custom_loss_fn(logits, features, labels, images)
    expected = F.cross_entropy(logits, labels)

    assert parts["penalty"] == pytest.approx(0.0)
    assert total.item() == pytest.approx(expected.item(), abs=1e-6)


# --------------------------------------------------------------------------- #
# the param_l2 convention
# --------------------------------------------------------------------------- #
def _drifted(trainer, delta: float = 0.01):
    """
    Anchor the trainer on its own model, then move every parameter by ``delta``.

    The distance is then exactly ``delta`` in every coordinate, so the two
    conventions can be checked against their closed forms.
    """
    trainer.anchor_state = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(delta)


def test_the_published_param_l2_is_the_per_tensor_mean(synthetic_provider):
    trainer = AnchoredTrainer(
        provider=synthetic_provider, space="param_l2", anchor="frozen", lam=1.0
    )
    torch.manual_seed(5)
    trainer.set_model(synthetic_provider.make_model())
    _drifted(trainer, 0.01)

    tensors = sum(
        1 for name, p in trainer.model.named_parameters()
        if p.requires_grad and name in trainer.anchor_state
    )
    # Each tensor contributes mean((0.01)^2) = 1e-4, whatever its size.
    assert float(trainer._param_l2()) == pytest.approx(tensors * 1e-4, rel=1e-4)


def test_the_fedprox_convention_is_mu_over_two_times_the_squared_norm(synthetic_provider):
    trainer = AnchoredTrainer(
        provider=synthetic_provider,
        space="param_l2",
        anchor="frozen",
        lam=1.0,
        param_l2_convention="fedprox",
    )
    torch.manual_seed(5)
    trainer.set_model(synthetic_provider.make_model())
    _drifted(trainer, 0.01)

    elements = sum(
        p.numel() for name, p in trainer.model.named_parameters()
        if p.requires_grad and name in trainer.anchor_state
    )
    assert float(trainer._param_l2()) == pytest.approx(0.5 * elements * 1e-4, rel=1e-4)


def test_the_two_conventions_differ_by_orders_of_magnitude(synthetic_provider):
    """This is why lam = 1000 was selected while FedProx's grid stops at mu = 1."""
    published, fedprox = (
        AnchoredTrainer(
            provider=synthetic_provider,
            space="param_l2",
            anchor="frozen",
            lam=1.0,
            param_l2_convention=convention,
        )
        for convention in ("mean_per_tensor", "fedprox")
    )
    torch.manual_seed(5)
    model = synthetic_provider.make_model()
    for trainer in (published, fedprox):
        trainer.set_model(copy.deepcopy(model))
        _drifted(trainer, 0.01)
    assert float(fedprox._param_l2()) > 10 * float(published._param_l2())


def test_an_unknown_convention_is_rejected(synthetic_provider):
    with pytest.raises(ValueError, match="param_l2 convention"):
        AnchoredTrainer(
            provider=synthetic_provider, space="param_l2", param_l2_convention="whatever"
        )


def test_the_convention_flag_reaches_the_trainer_kwargs():
    from federated_outlier_adaptation.cli import _trainer_overrides

    args = build_parser().parse_args(
        ["final", "--trainer", "AnchoredTrainer", "--fedprox-convention"]
    )
    assert _trainer_overrides(args)["param_l2_convention"] == "fedprox"
    plain = build_parser().parse_args(["final", "--trainer", "AnchoredTrainer"])
    assert "param_l2_convention" not in _trainer_overrides(plain)


# --------------------------------------------------------------------------- #
# the swept server coefficients
# --------------------------------------------------------------------------- #
def _weights(value: float):
    return {"w": torch.full((3,), float(value))}


def test_the_trim_fraction_is_a_parameter():
    global_weights = _weights(0.0)
    clients = [_weights(v) for v in (0.0, 1.0, 2.0, 3.0, 100.0)]
    counts = [1] * len(clients)

    untrimmed = con_delta_trimmed_mean(
        global_weights, clients, counts, server_state=ServerState(trim_fraction=0.0)
    )
    trimmed = con_delta_trimmed_mean(
        global_weights, clients, counts, server_state=ServerState(trim_fraction=0.2)
    )
    assert float(untrimmed["w"][0]) == pytest.approx(106.0 / 5)
    # 20 % of five clients drops the extreme at each end: mean(1, 2, 3).
    assert float(trimmed["w"][0]) == pytest.approx(2.0)


def test_an_impossible_trim_fraction_is_rejected():
    with pytest.raises(ValueError, match="trim_fraction"):
        ServerState(trim_fraction=0.5)


def test_the_server_momentum_is_a_parameter():
    global_weights = _weights(0.0)
    clients = [_weights(1.0)]

    slow = con_delta_fedavgm(
        global_weights, clients, [1], server_state=ServerState(momentum_beta=0.0)
    )
    fast = ServerState(momentum_beta=0.9)
    con_delta_fedavgm(global_weights, clients, [1], server_state=fast)
    second = con_delta_fedavgm(global_weights, clients, [1], server_state=fast)

    assert float(slow["w"][0]) == pytest.approx(1.0)
    # m <- 0.9 m + Delta, so the second round already overshoots.
    assert float(second["w"][0]) == pytest.approx(1.9)


def test_the_defaults_are_the_published_constants():
    from federated_outlier_adaptation.aggregation import concurrent_methods as cm

    state = ServerState()
    assert state.momentum_beta == cm.SERVER_MOMENTUM
    assert state.server_lr == cm.SERVER_LR
    assert state.tau == cm.SERVER_TAU
    assert state.trim_fraction == cm.TRIM_FRACTION


def test_the_server_hyperparameters_are_reported():
    keys = ServerState().hyperparameters()
    assert keys["server_momentum_beta"] == 0.9
    assert keys["trim_fraction"] == 0.2


def test_the_cyclic_mixing_weight_is_a_swept_scalar():
    global_weights = _weights(0.0)
    client = _weights(1.0)
    assert float(seq_mix_alpha(global_weights, client, alpha=0.25)["w"][0]) == pytest.approx(0.25)
    # alpha = n_k / N reproduces the FedAvg rule exactly.
    fedavg = seq_fedavg_update(
        global_weights, client, client_sample_count=3, all_clients_samples=4
    )
    mixed = seq_mix_alpha(global_weights, client, alpha=0.75)
    assert float(mixed["w"][0]) == pytest.approx(float(fedavg["w"][0]))


def test_the_mixing_weight_is_opt_in():
    """It must not appear in the published sweeps' method set."""
    assert "seq_mix_alpha" in SequentialWeightsAGGMETHODS.EXTENDED_METHODS

    from federated_outlier_adaptation.aggregation.selector import list_method_names

    assert "seq_mix_alpha" not in list_method_names(SequentialWeightsAGGMETHODS)
    assert "seq_mix_alpha" in list_method_names(SequentialWeightsAGGMETHODS, extended=True)


def test_an_out_of_range_mixing_weight_is_rejected():
    with pytest.raises(ValueError, match="alpha"):
        seq_mix_alpha(_weights(0.0), _weights(1.0), alpha=1.5)


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #
def test_the_setting_flags_are_available():
    args = build_parser().parse_args(
        ["prepare-data", "--zip", "x.zip", "--resolution", "28", "--classes", "all"]
    )
    assert args.resolution == 28
    assert args.classes == "all"


def test_the_model_is_selectable_per_run():
    args = build_parser().parse_args(
        ["final", "--trainer", "BaseTrainer", "--model", "flexible_cnn"]
    )
    assert args.model == "flexible_cnn"


def test_the_local_reference_can_start_from_scratch():
    args = build_parser().parse_args(["local-finetune", "--init", "scratch"])
    assert args.init == "scratch"
    assert build_parser().parse_args(["local-finetune"]).init == "global"


def test_the_pool_can_be_cut_by_accuracy_with_a_severity():
    args = build_parser().parse_args(
        [
            "select-outliers",
            "--mode",
            "pool",
            "--pool-max-acc",
            "0.55",
            "--severity",
            "severe",
        ]
    )
    assert args.pool_max_acc == 0.55
    assert args.severity == "severe"


@pytest.mark.parametrize(
    "flag,dest,value",
    [
        ("--server-beta", "server_beta", 0.97),
        ("--server-lr", "server_lr", 0.01),
        ("--server-tau", "server_tau", 0.001),
        ("--trim-frac", "trim_frac", 0.1),
        ("--seq-mix-alpha", "seq_mix_alpha", 0.3),
    ],
)
def test_the_swept_server_coefficients_are_on_the_command_line(flag, dest, value):
    from federated_outlier_adaptation.cli import _server_kwargs

    args = build_parser().parse_args(
        ["final", "--trainer", "BaseTrainer", flag, str(value)]
    )
    assert getattr(args, dest) == pytest.approx(value)
    if dest != "seq_mix_alpha":
        assert value in _server_kwargs(args).values()


def test_no_server_flag_leaves_every_published_default():
    from federated_outlier_adaptation.cli import _server_kwargs

    args = build_parser().parse_args(["final", "--trainer", "BaseTrainer"])
    assert _server_kwargs(args) == {}


@pytest.mark.parametrize("epochs", [1, 5, 20])
def test_the_local_epoch_budgets_of_the_screen_parse(epochs):
    args = build_parser().parse_args(
        ["final", "--trainer", "BaseTrainer", "--epochs", str(epochs)]
    )
    assert args.epochs == epochs


def test_one_local_epoch_still_keeps_the_trained_model(synthetic_provider):
    """
    ``E = 1`` is one of the screen's budgets.  With a single epoch there is
    nothing for early stopping to compare against, so the loop must simply run
    once and keep what it trained rather than restore an untrained copy.
    """
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    trainer = BaseTrainer(provider=synthetic_provider)
    torch.manual_seed(11)
    trainer.set_model(synthetic_provider.make_model())
    before = copy.deepcopy(trainer.get_model().state_dict())

    train_loader, eval_loader, _ = synthetic_provider.build_dataset(["c0"], batch_size=4)
    trainer.train(train_loader, eval_loader, epochs=1, patience=5)

    assert len(trainer.train_accuracies) == 1
    assert len(trainer.val_accuracies) == 1
    after = trainer.get_model().state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


# --------------------------------------------------------------------------- #
# the pooled-oracle reference is an annotation, not a precondition
# --------------------------------------------------------------------------- #
def test_a_missing_oracle_leaves_the_reference_empty(synthetic_provider, tmp_path, caplog):
    """
    The two reference accuracies come from `foa combined-train`.  A run whose
    results root has no oracle yet must still finish: the reference is a line on
    a plot, not an input to the training, and failing after the GPU-hours are
    spent is the worst possible moment to notice.
    """
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    runner = BaseConcurrentRunner(
        trainer=BaseTrainer(provider=synthetic_provider), provider=synthetic_provider
    )
    runner.results_dir = tmp_path  # an empty results root: no oracle, no global metrics
    runner.get_global_base_accuracy()

    assert runner.global_metrics_acc is None
    assert runner.global_clients_metric_acc is None
    assert runner.global_clients_all_metrics_acc is None


def test_the_present_oracle_is_still_read(synthetic_provider):
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    runner = BaseConcurrentRunner(
        trainer=BaseTrainer(provider=synthetic_provider), provider=synthetic_provider
    )
    runner.get_global_base_accuracy()
    assert runner.global_metrics_acc == pytest.approx(0.5)
    assert runner.global_clients_metric_acc == pytest.approx(0.5)


def test_the_driver_stores_a_null_reference_rather_than_crashing():
    from federated_outlier_adaptation.training.driver import _optional_float

    assert _optional_float(None) is None
    assert _optional_float(0.5) == pytest.approx(0.5)


def test_the_extreme_command_mirrors_final_s_client_sources():
    """
    ``foa extreme`` already takes ``--outliers-file`` (and ``--pool-frac``) the
    way ``foa final`` does, so the extreme cases can be pointed at a severity
    pool; ``--clients`` names the participants directly and is independent of
    both.
    """
    parser = build_parser()
    args = parser.parse_args(
        [
            "extreme",
            "--case",
            "dual",
            "--outliers-file",
            "/tmp/pool.json",
            "--clients",
            "wA",
            "wA",
        ]
    )
    assert args.outliers_file == "/tmp/pool.json"
    assert args.clients == ["wA", "wA"]
    assert parser.parse_args(["extreme", "--pool-frac", "0.05"]).pool_frac == 0.05


# --------------------------------------------------------------------------- #
# the per-outlier reference models: one degenerate writer must not kill the job
# --------------------------------------------------------------------------- #
class _DegenerateWriterProvider:
    """
    A provider whose ``degenerate`` writer yields no loaders at all.

    That is what a 60/40 split of a very small writer does: at 62 classes some
    writers hold as few as 18 images and the split can round down to nothing.
    Everything else is delegated to the synthetic provider.
    """

    def __init__(self, inner, degenerate: str):
        self.inner = inner
        self.degenerate = degenerate

    def _is_degenerate(self, client_ids) -> bool:
        ids = [client_ids] if isinstance(client_ids, str) else list(client_ids)
        return ids == [self.degenerate]

    def make_model(self):
        return self.inner.make_model()

    def sample_count(self, client_id):
        return self.inner.sample_count(client_id)

    def build_dataset(self, client_ids, **kwargs):
        if self._is_degenerate(client_ids):
            return None, None, None
        return self.inner.build_dataset(client_ids, **kwargs)


def test_a_writer_without_data_is_skipped_not_raised(synthetic_provider, tmp_path, monkeypatch):
    """
    The per-outlier step is a *tail* of the global-training phase: theta_g is
    already trained and saved when it runs.  A writer whose split is empty used
    to raise ``TypeError: 'NoneType' object is not iterable`` from the training
    loop and take the whole finished job with it.  It is now a warning and a
    skip.
    """
    from federated_outlier_adaptation.training import outlier_models

    # NistLogger owns its handlers and does not propagate to the root logger,
    # so the warning is captured at the source rather than through caplog.
    warnings: list[str] = []
    monkeypatch.setattr(
        outlier_models.NistLogger, "warning", lambda message: warnings.append(str(message))
    )

    provider = _DegenerateWriterProvider(synthetic_provider, "c1")
    results = outlier_models.outliers_train(
        global_writers=["g0"],
        selected_5_writers=["c0", "c1"],
        batch_size=4,
        num_epochs=1,
        outliers_results_path=tmp_path,
        provider=provider,
    )

    assert results["c1"]["skipped"] is True
    assert "empty" in results["c1"]["reason"]
    assert results["c1"]["split_sizes"] == {"train": 0, "validation": 0}
    # ...and the writer that does have data was trained all the same.
    assert results["c0"].get("skipped") is None
    assert len(results["c0"]["train_accuracies"]) == 1

    joined = "\n".join(warnings)
    assert "Skipping the per-writer model of c1" in joined
    # the counts are in the message, so the log says why without a rerun
    assert "'train': 0" in joined and "'validation': 0" in joined
    assert "1 of 2 writer(s) were skipped" in joined


def test_the_skipped_writer_is_recorded_in_the_json(synthetic_provider, tmp_path):
    import json

    from federated_outlier_adaptation.training.outlier_models import outliers_train

    outliers_train(
        global_writers=["g0"],
        selected_5_writers=["c0", "c1"],
        batch_size=4,
        num_epochs=1,
        outliers_results_path=tmp_path,
        provider=_DegenerateWriterProvider(synthetic_provider, "c1"),
    )
    with open(tmp_path / "outlier_training_results.json") as handle:
        stored = json.load(handle)
    assert stored["c1"]["skipped"] is True
    assert "train_accuracies" in stored["c0"]


def test_the_published_per_writer_path_is_unchanged(synthetic_provider, tmp_path):
    """A writer with data is trained, scored and plotted exactly as before."""
    from federated_outlier_adaptation.training.outlier_models import outliers_train

    results = outliers_train(
        global_writers=["g0"],
        selected_5_writers=["c0"],
        batch_size=4,
        num_epochs=2,
        outliers_results_path=tmp_path,
        provider=synthetic_provider,
    )
    entry = results["c0"]
    assert len(entry["train_accuracies"]) == 2
    assert len(entry["eval_accuracies"]) == 2
    assert 0.0 <= entry["global_test_accuracy"] <= 1.0
    assert 0.0 <= entry["all_client_test_accuracies"] <= 1.0
    assert (tmp_path / "c0_accuracy_plot.png").is_file()


def test_the_skip_flag_defaults_to_the_published_behaviour():
    parser = build_parser()
    assert parser.parse_args(["global-train"]).skip_outlier_training is False
    assert (
        parser.parse_args(["global-train", "--skip-outlier-training"]).skip_outlier_training
        is True
    )


@pytest.mark.parametrize("skip", [False, True])
def test_the_skip_flag_reaches_the_pipeline(monkeypatch, skip):
    from federated_outlier_adaptation import cli
    from federated_outlier_adaptation.training import global_model

    seen = {}
    monkeypatch.setattr(cli, "_resolve_provider", lambda args: None)
    monkeypatch.setattr(
        global_model, "run_global_training", lambda **kwargs: seen.update(kwargs)
    )

    argv = ["global-train", "--epochs", "1"] + (["--skip-outlier-training"] if skip else [])
    assert cli.cmd_global_train(build_parser().parse_args(argv)) == 0
    assert seen["skip_outlier_training"] is skip
    assert seen["epochs"] == 1


# --------------------------------------------------------------------------- #
# Phase D: sweeps that can be extended, finals that can be run under two rules
# --------------------------------------------------------------------------- #
def test_the_convention_travels_from_the_sweep_to_the_final():
    """
    ``lam`` means FedProx's ``mu`` under one convention and a per-tensor-mean
    quantity under the other, so a final that executed the selected ``lam``
    under the *other* convention would run a different objective from the one
    that was measured.  The convention is therefore part of what is selected.
    """
    from federated_outlier_adaptation.grid_search.selection import (
        SELECTED_KEYS,
        Record,
        selected_hyperparameters,
    )

    assert "param_l2_convention" in SELECTED_KEYS
    record = Record(
        name="r",
        source="s",
        config={
            "trainer_kwargs": {
                "space": "param_l2",
                "lam": 0.01,
                "param_l2_convention": "fedprox",
            }
        },
    )
    chosen = selected_hyperparameters(record)
    assert chosen == {"lam": 0.01, "param_l2_convention": "fedprox"}


def test_a_space_without_a_convention_carries_none():
    from federated_outlier_adaptation.grid_search.selection import selected_hyperparameters
    from federated_outlier_adaptation.grid_search.selection import Record

    record = Record(name="r", source="s", config={"trainer_kwargs": {"lam": 1, "T": 4}})
    assert selected_hyperparameters(record) == {"lam": 1, "T": 4}


def test_one_aggregation_emits_the_published_line_unchanged():
    """The default must be byte-identical to what the planner always emitted."""
    from federated_outlier_adaptation.training.matrix import reg_final_task

    line = reg_final_task("kd", "current", 0.005, 1, {"lam": 1.0, "T": 4})
    assert " --aggregation fedavg" in line
    assert " --parent reg_kd_current_eps0005" in line
    assert " --pool-frac" in line
    assert 'space="kd"' in line and 'anchor="current"' in line
    assert "lam=1" in line and "T=4" in line


def test_each_rule_gets_its_own_folder():
    from federated_outlier_adaptation.training.matrix import reg_parent

    assert reg_parent("kd", "current", 0.005) == "reg_kd_current_eps0005"
    assert (
        reg_parent("kd", "current", 0.005, "con_delta_median")
        == "reg_kd_current_eps0005_con_delta_median"
    )


def test_the_emitted_finals_can_name_a_pool_and_a_budget():
    from federated_outlier_adaptation.training.matrix import reg_final_task

    line = reg_final_task(
        "ntd",
        "frozen",
        0.005,
        2,
        {"lam": 0.3, "T": 2},
        aggregation="con_delta_median",
        extra_args="--results-dir /r/main --rounds 100 --epochs 5",
        pool_args="--outliers-file /r/pool.json --clients-per-round 10 --policy uniform",
        parent_suffix="con_delta_median",
    )
    assert "--pool-frac" not in line
    assert "--outliers-file /r/pool.json" in line
    assert "--sampler-seed 2" in line and "--seed 2" in line
    assert "--rounds 100 --epochs 5" in line
    assert line.endswith("--skip-existing")
    assert "--aggregation con_delta_median" in line


def test_two_rules_produce_two_non_colliding_task_sets(tmp_path):
    import json

    from federated_outlier_adaptation.training.matrix import emit_regularisation_finals

    sweep = tmp_path / "anchored_kd_current_grid_search"
    sweep.mkdir()
    with open(sweep / "constrained_selection_eps0.005.json", "w") as handle:
        json.dump(
            {
                "eps": 0.005,
                "space": "kd",
                "anchor": "current",
                "selected": {"name": "x"},
                "hyperparameters": {"lam": 1.0, "T": 4},
            },
            handle,
        )
    with open(sweep / "config_points_100.json", "w") as handle:
        json.dump({"config": {"x": {"trainer_kwargs": {"space": "kd", "anchor": "current"}}}}, handle)

    tasks = emit_regularisation_finals(
        tmp_path,
        eps_values=[0.005],
        seeds=[1, 2],
        aggregations=("fedavg", "con_delta_median"),
    )
    assert len(tasks) == 4
    parents = {line.split("--parent ")[1].split()[0] for line in tasks}
    assert parents == {
        "reg_kd_current_eps0005_fedavg",
        "reg_kd_current_eps0005_con_delta_median",
    }


def test_the_sweep_index_can_be_overridden():
    """
    A second invocation writes ``accuracies_points_<index>.json`` next to the
    first, so the 'penalty off' arm joins the same sweep instead of overwriting
    it or hiding in a folder where the selection could not see it.
    """
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["grid", "--method", "anchored", "--space", "kd", "--index", "101"]
    )
    assert args.index == 101
    assert parser.parse_args(["grid", "--method", "anchored"]).index is None


def test_the_index_reaches_the_task_list():
    from federated_outlier_adaptation.grid_search.anchored import build_spec

    spec = build_spec(space="kd", anchor="frozen", lams=[0.0], temperatures=[1])
    retasked = [
        (scenario, metadata, 101, rule) for scenario, metadata, _, rule in spec.tasks
    ]
    assert {task[2] for task in retasked} == {101}
    assert [task[3] for task in retasked] == [task[3] for task in spec.tasks]


def test_the_finals_emitter_flags_are_on_the_command_line():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "select",
            "--emit-finals",
            "--eps",
            "0.005",
            "--final-aggregation",
            "fedavg",
            "con_delta_median",
            "--final-args",
            "--rounds 100",
        ]
    )
    assert args.final_aggregation == ["fedavg", "con_delta_median"]
    assert args.final_args == "--rounds 100"
    assert parser.parse_args(["select"]).final_aggregation == ["fedavg"]


# --------------------------------------------------------------------------- #
# the fixed cohort (Phase E): an enrolment, not a pool
# --------------------------------------------------------------------------- #
def _accuracy_file(tmp_path, scores):
    import json

    path = tmp_path / "clients_acc_on_global.json"
    with open(path, "w") as handle:
        json.dump([{client: value} for client, value in scores.items()], handle)
    return path


def test_the_cohort_is_the_k_worst_clients_worst_first(tmp_path):
    from federated_outlier_adaptation.outliers.client_accuracy import build_cohort

    path = _accuracy_file(tmp_path, {"a": 0.9, "b": 0.2, "c": 0.5, "d": 0.1})
    cohort = build_cohort(path, k=2)
    assert cohort["clients"] == ["d", "b"]
    assert cohort["rule"] == "worst_k"
    assert cohort["size"] == 2
    assert cohort["accuracies"] == {"d": 0.1, "b": 0.2}
    assert cohort["min_accuracy"] == pytest.approx(0.1)
    assert cohort["mean_accuracy"] == pytest.approx(0.15)


def test_the_cohort_is_deterministic_and_breaks_ties_by_id(tmp_path):
    from federated_outlier_adaptation.outliers.client_accuracy import build_cohort

    path = _accuracy_file(tmp_path, {"b": 0.5, "a": 0.5, "c": 0.9})
    assert build_cohort(path, k=2)["clients"] == ["a", "b"]
    assert build_cohort(path, k=2) == build_cohort(path, k=2)


def test_the_eligibility_rule_applies_before_the_cut(tmp_path):
    """
    A client too small to train is exactly the one that scores worst, so it
    would be enrolled first of all.  It has to be removed *before* the cut, and
    the record has to say so.
    """
    from federated_outlier_adaptation.outliers.client_accuracy import build_cohort

    path = _accuracy_file(tmp_path, {"tiny": 0.05, "a": 0.2, "b": 0.3, "c": 0.9})
    cohort = build_cohort(path, k=2, eligible=["a", "b", "c"])
    assert cohort["clients"] == ["a", "b"]
    assert cohort["excluded_from_cohort"] == ["tiny"]
    assert cohort["scored_clients"] == 4
    assert cohort["eligible_clients"] == 3
    assert cohort["ineligible_clients"] == 1


def test_a_cohort_that_excludes_nothing_says_so(tmp_path):
    from federated_outlier_adaptation.outliers.client_accuracy import build_cohort

    path = _accuracy_file(tmp_path, {"a": 0.2, "b": 0.3})
    cohort = build_cohort(path, k=2, eligible=["a", "b"])
    assert cohort["excluded_from_cohort"] == []
    assert cohort["ineligible_clients"] == 0


def test_a_cohort_larger_than_the_population_is_the_population(tmp_path):
    from federated_outlier_adaptation.outliers.client_accuracy import build_cohort

    path = _accuracy_file(tmp_path, {"a": 0.2, "b": 0.3})
    cohort = build_cohort(path, k=20)
    assert cohort["clients"] == ["a", "b"]
    assert cohort["k"] == 20 and cohort["size"] == 2


def test_the_cohort_file_reads_back_like_a_pool(tmp_path):
    from federated_outlier_adaptation.outliers.client_accuracy import write_cohort
    from federated_outlier_adaptation.outliers.selection import (
        load_client_pool,
        load_pool_meta,
    )

    path = _accuracy_file(tmp_path, {"a": 0.2, "b": 0.3, "c": 0.9})
    target, payload = write_cohort(path, tmp_path / "out", k=2, tag="cohort20")
    assert target.name == "cohort_worst2.json"

    clients, accuracies = load_client_pool(target)
    assert clients == ["a", "b"]
    assert accuracies == {"a": 0.2, "b": 0.3}

    meta = load_pool_meta(target)
    assert meta["rule"] == "worst_k"
    assert meta["tag"] == "cohort20"
    assert meta["k"] == 2


def test_the_worst_mode_is_on_the_command_line():
    from federated_outlier_adaptation.cli import SELECTION_MODES

    parser = build_parser()
    args = parser.parse_args(
        ["select-outliers", "--mode", "worst", "--k", "20", "--require-trainable",
         "--tag", "cohort20"]
    )
    assert args.mode == "worst" and args.k == 20
    assert args.require_trainable is True and args.tag == "cohort20"
    assert "worst" in SELECTION_MODES
    assert parser.parse_args(["select-outliers"]).tag is None


def test_select_cohort_writes_under_the_results_root(tmp_path):
    from federated_outlier_adaptation.outliers.selection import select_cohort

    outliers = tmp_path / "outliers"
    outliers.mkdir()
    _accuracy_file(outliers, {"a": 0.2, "b": 0.3, "c": 0.9})

    payload, path = select_cohort(results_dir=tmp_path, k=2, write_table=False)
    assert path == outliers / "cohort_worst2.json"
    assert payload["clients"] == ["a", "b"]

    # an existing cohort is reused, so a rerunning array keeps one enrolment
    again, _ = select_cohort(results_dir=tmp_path, k=2, write_table=False)
    assert again["clients"] == payload["clients"]


def test_a_scenario_prefix_keeps_two_studies_apart():
    """
    A cohort study and a population study of the same spaces share a results
    root, so their finals need distinct folders as much as two rules do.
    """
    from federated_outlier_adaptation.training.matrix import reg_final_task, reg_parent

    assert reg_parent("kd", "current", 0.005, prefix="coh_") == "coh_reg_kd_current_eps0005"
    assert (
        reg_parent("kd", "current", 0.005, suffix="fedavg", prefix="coh_")
        == "coh_reg_kd_current_eps0005_fedavg"
    )
    line = reg_final_task("kd", "current", 0.005, 1, {"lam": 1.0}, parent_prefix="coh_")
    assert "--parent coh_reg_kd_current_eps0005 " in line
    # the default is still exactly what the planner always emitted
    assert "--parent reg_kd_current_eps0005 " in reg_final_task("kd", "current", 0.005, 1)


def test_the_prefix_flag_is_on_the_command_line():
    parser = build_parser()
    args = parser.parse_args(
        ["select", "--emit-finals", "--final-parent-prefix", "coh_"]
    )
    assert args.final_parent_prefix == "coh_"
    assert parser.parse_args(["select"]).final_parent_prefix is None


# --------------------------------------------------------------------------- #
# Phase F arm 1: the data-sharing upper bound
# --------------------------------------------------------------------------- #
def _sizes(loader):
    """Sample count of one epoch, and of the epoch after it."""
    from federated_outlier_adaptation.data.source_share import SourceShareSampler

    sampler = loader.sampler
    assert isinstance(sampler, SourceShareSampler)
    first = list(iter(sampler))
    second = list(iter(sampler))
    return first, second


def _tiny_sets():
    from torch.utils.data import TensorDataset

    client = TensorDataset(torch.arange(10).float().reshape(10, 1), torch.zeros(10).long())
    source = TensorDataset(torch.arange(50).float().reshape(50, 1), torch.ones(50).long())
    return client, source


@pytest.mark.parametrize(
    "mode,client,source,cap,expected",
    [
        ("off", 10, 50, None, 0),
        ("equal", 10, 50, None, 10),
        ("full", 10, 50, None, 50),
        ("equal", 80, 50, None, 50),        # never more than the pool holds
        ("full", 10, 50, 8, 8),             # the documented cap
        ("equal", 10, 50, 4, 4),
        ("equal", 0, 50, None, 0),          # a client with nothing to mix into
        ("full", 10, 0, None, 0),           # nothing to draw from
    ],
)
def test_the_share_size_follows_the_mode(mode, client, source, cap, expected):
    from federated_outlier_adaptation.data.source_share import share_size

    assert share_size(mode, client, source, cap) == expected


def test_an_unknown_mode_is_rejected():
    from federated_outlier_adaptation.data.source_share import share_size

    with pytest.raises(ValueError, match="source-share mode"):
        share_size("some", 10, 50)


def test_mode_off_leaves_the_loader_alone():
    """The default path has to be bit-for-bit what it was."""
    from federated_outlier_adaptation.data.source_share import build_shared_loader

    client, source = _tiny_sets()
    loader, info = build_shared_loader(client, source, "off", batch_size=4, seed=1)
    assert loader is None
    assert info["source_per_epoch"] == 0
    assert info["mode"] == "off"


def test_equal_gives_a_balanced_epoch():
    from federated_outlier_adaptation.data.source_share import build_shared_loader

    client, source = _tiny_sets()
    loader, info = build_shared_loader(client, source, "equal", batch_size=4, seed=7)
    first, _ = _sizes(loader)
    assert len(first) == 20 == info["samples_per_epoch"]
    assert info["client_samples"] == 10 and info["source_per_epoch"] == 10
    # every client sample appears, exactly once
    assert sorted(index for index in first if index < 10) == list(range(10))


def test_full_takes_the_whole_pool_every_epoch():
    from federated_outlier_adaptation.data.source_share import build_shared_loader

    client, source = _tiny_sets()
    loader, info = build_shared_loader(client, source, "full", batch_size=4, seed=7)
    first, second = _sizes(loader)
    assert len(first) == 60 == info["samples_per_epoch"]
    assert sorted(first) == sorted(second) == list(range(60))


def test_the_source_sample_is_redrawn_per_epoch():
    """`equal` is a share of the source distribution, not one fixed subset."""
    from federated_outlier_adaptation.data.source_share import build_shared_loader

    client, source = _tiny_sets()
    loader, _ = build_shared_loader(client, source, "equal", batch_size=4, seed=7)
    first, second = _sizes(loader)
    drawn_first = {index for index in first if index >= 10}
    drawn_second = {index for index in second if index >= 10}
    assert drawn_first != drawn_second
    assert len(drawn_first) == len(drawn_second) == 10


def test_the_draw_is_deterministic_for_a_seed():
    from federated_outlier_adaptation.data.source_share import build_shared_loader

    client, source = _tiny_sets()
    one, _ = build_shared_loader(client, source, "equal", batch_size=4, seed=11)
    two, _ = build_shared_loader(client, source, "equal", batch_size=4, seed=11)
    assert list(iter(one.sampler)) == list(iter(two.sampler))

    # ...and a different seed draws differently. Compared at the same epoch:
    # every sampler advances its own counter as it is iterated.
    other, _ = build_shared_loader(client, source, "equal", batch_size=4, seed=12)
    fresh, _ = build_shared_loader(client, source, "equal", batch_size=4, seed=11)
    assert list(iter(fresh.sampler)) != list(iter(other.sampler))


def test_adjacent_seeds_do_not_alias_across_epochs():
    """
    The epoch is mixed into the base seed, not added to it.  Adding would make
    two clients whose seeds differ by one draw the same images one epoch apart -
    a quietly wrong experiment rather than a crash.
    """
    from federated_outlier_adaptation.data.source_share import SourceShareSampler

    a = SourceShareSampler(10, 50, 10, seed=11)
    b = SourceShareSampler(10, 50, 10, seed=12)
    assert a.epoch_seed(1) != b.epoch_seed(0)
    assert a.epoch_seed(0) != b.epoch_seed(0)
    assert a.epoch_seed(0) == SourceShareSampler(10, 50, 10, seed=11).epoch_seed(0)


def test_the_seed_is_a_function_of_run_round_and_client():
    from federated_outlier_adaptation.data.source_share import sampler_seed

    base = sampler_seed(1, 0, "c0")
    assert base == sampler_seed(1, 0, "c0")            # stable across processes
    assert base != sampler_seed(1, 1, "c0")            # ...different round
    assert base != sampler_seed(1, 0, "c1")            # ...different client
    assert base != sampler_seed(2, 0, "c0")            # ...different run
    assert 0 <= base < 2 ** 32


def test_the_mixed_loader_yields_both_populations():
    from federated_outlier_adaptation.data.source_share import build_shared_loader

    client, source = _tiny_sets()
    loader, _ = build_shared_loader(client, source, "equal", batch_size=4, seed=3)
    labels = torch.cat([batch_labels for _, batch_labels in loader])
    assert len(labels) == 20
    assert int((labels == 0).sum()) == 10      # the client's own
    assert int((labels == 1).sum()) == 10      # the source share


def test_the_runner_reports_the_share_in_provenance(synthetic_provider):
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    runner = BaseConcurrentRunner(
        trainer=BaseTrainer(provider=synthetic_provider),
        provider=synthetic_provider,
        source_share="equal",
        source_share_cap=64,
    )
    assert runner.source_share == "equal"
    assert runner.source_share_cap == 64
    assert runner.source_share_info["mode"] == "equal"
    # off by default, and then the block simply says so
    plain = BaseConcurrentRunner(
        trainer=BaseTrainer(provider=synthetic_provider), provider=synthetic_provider
    )
    assert plain.source_share == "off"
    assert plain._share_source("untouched", "c0", 0, 8) == "untouched"


def test_the_share_flags_are_on_the_command_line():
    parser = build_parser()
    args = parser.parse_args(
        ["final", "--trainer", "BaseTrainer", "--source-share", "full",
         "--source-share-cap", "8000"]
    )
    assert args.source_share == "full" and args.source_share_cap == 8000
    plain = parser.parse_args(["final", "--trainer", "BaseTrainer"])
    assert plain.source_share == "off" and plain.source_share_cap is None


# --------------------------------------------------------------------------- #
# Phase F arm 2: personalisation from a federated checkpoint
# --------------------------------------------------------------------------- #
def test_the_final_server_model_can_be_kept(synthetic_provider, tmp_path):
    """
    No federated run wrote a checkpoint before this: the personalisation arm
    would have had nothing to start from.
    """
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    runner = BaseConcurrentRunner(
        trainer=BaseTrainer(provider=synthetic_provider),
        provider=synthetic_provider,
        save_model_path=str(tmp_path / "nested" / "final_model.pt"),
    )
    assert runner.save_model_path == tmp_path / "nested" / "final_model.pt"
    # the save itself is what simulate() does at the end of a run
    runner.global_model = synthetic_provider.make_model()
    runner.save_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(runner.global_model.state_dict(), runner.save_model_path)

    restored = synthetic_provider.make_model()
    restored.load_state_dict(torch.load(runner.save_model_path))
    for (_, a), (_, b) in zip(
        sorted(runner.global_model.state_dict().items()),
        sorted(restored.state_dict().items()),
    ):
        assert torch.equal(a, b)


def test_the_save_flag_is_off_by_default():
    parser = build_parser()
    assert parser.parse_args(["final", "--trainer", "BaseTrainer"]).save_final_model is False
    assert parser.parse_args(
        ["final", "--trainer", "BaseTrainer", "--save-final-model"]
    ).save_final_model is True


def test_personalisation_starts_from_the_named_checkpoint(synthetic_provider, tmp_path):
    from federated_outlier_adaptation.training.local_finetune import run_local_finetuning

    # a checkpoint that is deliberately not theta_g
    torch.manual_seed(99)
    federated = synthetic_provider.make_model()
    checkpoint = tmp_path / "final_model.pt"
    torch.save(federated.state_dict(), checkpoint)

    payload = run_local_finetuning(
        trainer_name="BaseTrainer",
        epochs=1,
        batch_size=4,
        seed=1,
        outliers_file=str(synthetic_provider.outliers_file),
        init_checkpoint=str(checkpoint),
        parent_name="pers",
        provider=synthetic_provider,
    )
    assert payload["init"] == "global"
    assert payload["init_checkpoint"] == str(checkpoint)
    assert payload["init_checkpoint_sha256"]
    assert payload["config"]["init_checkpoint"] == str(checkpoint)
    # one entry per cohort client, with both axes reported
    assert set(payload["per_client"]) == set(synthetic_provider.selected_clients())
    for entry in payload["per_client"].values():
        assert 0.0 <= entry["insample"] <= 1.0
        assert 0.0 <= entry["source_test"] <= 1.0
        assert 0.0 <= entry["source_val"] <= 1.0


def test_a_missing_checkpoint_says_what_produces_it(synthetic_provider, tmp_path):
    from federated_outlier_adaptation.training.local_finetune import run_local_finetuning

    with pytest.raises(FileNotFoundError, match="save-final-model"):
        run_local_finetuning(
            trainer_name="BaseTrainer",
            epochs=1,
            batch_size=4,
            outliers_file=str(synthetic_provider.outliers_file),
            init_checkpoint=str(tmp_path / "absent.pt"),
            parent_name="pers_missing",
            provider=synthetic_provider,
        )


def test_without_a_checkpoint_it_is_still_r2(synthetic_provider):
    """The same command with no checkpoint is the control rung."""
    from federated_outlier_adaptation.training.local_finetune import run_local_finetuning

    payload = run_local_finetuning(
        trainer_name="BaseTrainer",
        epochs=1,
        batch_size=4,
        seed=2,
        outliers_file=str(synthetic_provider.outliers_file),
        parent_name="r2_control",
        provider=synthetic_provider,
    )
    assert payload["init"] == "global"
    assert "init_checkpoint" not in payload


# --------------------------------------------------------------------------- #
# the share arm, end to end through both runners
# --------------------------------------------------------------------------- #
# The unit tests above cover the sampler; they did not cover the runner, and the
# first cluster submission died 6.5 minutes in with an AttributeError because the
# wiring reached for `self.global_writers` - which only the sequential runner
# has.  These run the whole loop, which is the class of test that catches that.
def _share_simulation(provider, scenario, **kwargs):
    """Two seeded rounds through one runner, with the share wired in."""
    from federated_outlier_adaptation.aggregation.selector import select_class
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
    from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    if scenario == "sequential":
        runner_cls, agg_name = BaseSequentialRunner, "seq_fedavg_update"
    else:
        runner_cls, agg_name = BaseConcurrentRunner, "con_weighted_cgw"

    runner = runner_cls(
        trainer=BaseTrainer(provider=provider), provider=provider, seed=11, **kwargs
    )
    accuracies, *_ = runner.simulate(
        exp_name="share_arm",
        global_name="global",
        aggregate_method=getattr(select_class(scenario, "weights"), agg_name),
        batch_size=4,
        epochs=1,
        max_round=2,
        grid_Search=False,
    )
    return runner, accuracies


@pytest.mark.parametrize("scenario", ["concurrent", "sequential"])
def test_the_share_arm_runs_end_to_end(synthetic_provider, scenario):
    runner, accuracies = _share_simulation(
        synthetic_provider, scenario, source_share="equal"
    )
    assert len(accuracies) == 2

    info = runner.source_share_info
    assert info["mode"] == "equal"
    assert info["source_pool"] > 0
    assert info["source_per_epoch"] == info["client_samples"]
    assert info["samples_per_epoch"] == 2 * info["client_samples"]
    # ...and it is reported where a stored result can be read off
    assert runner.population_info()["source_share"] == info


@pytest.mark.parametrize("scenario", ["concurrent", "sequential"])
def test_the_full_mode_respects_its_cap(synthetic_provider, scenario):
    runner, accuracies = _share_simulation(
        synthetic_provider, scenario, source_share="full", source_share_cap=3
    )
    assert len(accuracies) == 2

    info = runner.source_share_info
    assert info["mode"] == "full"
    assert info["cap"] == 3
    assert info["source_per_epoch"] == 3
    assert info["samples_per_epoch"] == info["client_samples"] + 3


@pytest.mark.parametrize("scenario", ["concurrent", "sequential"])
def test_the_default_run_never_builds_a_pool(synthetic_provider, scenario):
    """`off` must not touch the loaders, and must not go looking for a pool."""
    runner, accuracies = _share_simulation(synthetic_provider, scenario)
    assert len(accuracies) == 2
    assert runner.source_train_dataset is None
    assert runner.population_info()["source_share"] == {"mode": "off"}


def test_the_pool_comes_from_the_provider_not_a_runner_attribute(synthetic_provider):
    """
    Both runners keep their client split in different places - the sequential
    one on an attribute, the concurrent one in a local - so the pool is resolved
    from the provider, which is the same call in either.
    """
    from federated_outlier_adaptation.data.source_share import source_pool_dataset
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner

    assert not hasattr(BaseConcurrentRunner, "global_writers")
    pool = source_pool_dataset(synthetic_provider, batch_size=4)
    assert pool is not None and len(pool) > 0


def test_a_source_population_without_data_is_reported(synthetic_provider):
    """Asking for a share with nothing to share must say so, not train on air."""
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    class _NoSource:
        def __getattr__(self, name):
            return getattr(synthetic_provider, name)

        def build_dataset(self, client_ids, **kwargs):
            if list(client_ids) == list(synthetic_provider.global_client_ids()):
                return None, None, None
            return synthetic_provider.build_dataset(client_ids, **kwargs)

    runner = BaseConcurrentRunner(
        trainer=BaseTrainer(provider=synthetic_provider),
        provider=_NoSource(),
        seed=11,
        source_share="equal",
    )
    from federated_outlier_adaptation.aggregation.selector import select_class

    # The message now names --old-book too: under v6 the shared population is
    # the old-data draw, not the frozen 3% writer split.
    with pytest.raises(ValueError, match="no old training data to share"):
        runner.simulate(
            exp_name="share_empty",
            global_name="global",
            aggregate_method=select_class("concurrent", "weights").con_weighted_cgw,
            batch_size=4,
            epochs=1,
            max_round=1,
            grid_Search=False,
        )


# --------------------------------------------------------------------------- #
# P0.1 - the convergence rule of the centralised trainings
# --------------------------------------------------------------------------- #
def test_the_stopper_ends_a_plateau():
    from federated_outlier_adaptation.training.convergence import EarlyStopper

    stopper = EarlyStopper(patience=2)
    assert stopper.update(0, 0.50) is False
    assert stopper.update(1, 0.60) is False          # improved
    assert stopper.update(2, 0.59) is False          # 1 bad
    assert stopper.update(3, 0.55) is True           # 2 bad -> stop
    assert stopper.stopped_early is True
    assert stopper.best_epoch == 1
    assert stopper.best_accuracy == pytest.approx(0.60)
    assert stopper.epochs_run == 4


def test_the_floor_is_respected():
    """An opening plateau must not end a run before the floor."""
    from federated_outlier_adaptation.training.convergence import EarlyStopper

    stopper = EarlyStopper(patience=1, min_epochs=4)
    assert stopper.update(0, 0.50) is False
    assert [stopper.update(epoch, 0.10) for epoch in (1, 2)] == [False, False]
    assert stopper.update(3, 0.10) is True           # the floor is now reached
    assert stopper.epochs_run == 4


def test_no_patience_never_stops():
    from federated_outlier_adaptation.training.convergence import EarlyStopper

    stopper = EarlyStopper(patience=None)
    assert [stopper.update(epoch, 0.10) for epoch in range(20)] == [False] * 20
    assert stopper.stopped_early is False
    assert stopper.enabled is False
    # ...but the best epoch is still tracked, which is the point of keeping it
    assert stopper.best_epoch == 0


def test_the_best_weights_are_the_ones_kept():
    from federated_outlier_adaptation.training.convergence import EarlyStopper

    best_marker, last_marker = object(), object()
    stopper = EarlyStopper(patience=2, restore_best=False)
    stopper.update(0, 0.9, best_marker)
    stopper.update(1, 0.1, last_marker)
    # restore_best=False keeps nothing, so the caller's fallback stands
    assert stopper.best(fallback=last_marker) is last_marker

    stopper = EarlyStopper(patience=2)
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
    stopper.update(0, 0.9, model)                    # best epoch: weights are 1
    with torch.no_grad():
        model.weight.fill_(7.0)
    stopper.update(1, 0.1, model)                    # a worse, later epoch
    assert torch.allclose(stopper.best().weight, torch.ones_like(model.weight))


def test_the_stopper_reports_what_it_did():
    from federated_outlier_adaptation.training.convergence import EarlyStopper

    stopper = EarlyStopper(patience=3, min_epochs=5)
    stopper.update(0, 0.4)
    metrics = stopper.metrics()
    assert metrics["early_stopping_patience"] == 3
    assert metrics["min_epochs"] == 5
    assert metrics["epochs_run"] == 1
    assert metrics["best_epoch"] == 0
    assert metrics["best_val_accuracy"] == pytest.approx(0.4)
    assert metrics["stopped_early"] is False


def test_a_trainer_records_its_convergence(synthetic_provider):
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    trainer = BaseTrainer(provider=synthetic_provider)
    assert trainer.convergence == {}
    trainer.set_model(synthetic_provider.make_model())
    train_loader, eval_loader, _ = synthetic_provider.build_dataset(["c0"], batch_size=4)
    trainer.train(train_loader, eval_loader, epochs=3, min_epochs=2)
    assert trainer.convergence["epochs_run"] == 3
    assert trainer.convergence["min_epochs"] == 2
    assert trainer.convergence["best_epoch"] is not None
    assert trainer.convergence["stopped_early"] is False


def test_local_finetuning_reports_it_per_client(synthetic_provider):
    from federated_outlier_adaptation.training.local_finetune import run_local_finetuning

    payload = run_local_finetuning(
        trainer_name="BaseTrainer",
        epochs=4,
        batch_size=4,
        seed=5,
        outliers_file=str(synthetic_provider.outliers_file),
        patience=1,
        min_epochs=2,
        parent_name="conv",
        provider=synthetic_provider,
    )
    assert payload["config"]["early_stopping_patience"] == 1
    assert payload["config"]["min_epochs"] == 2
    for entry in payload["per_client"].values():
        convergence = entry["convergence"]
        # the floor holds for every client, and never more than the budget
        assert 2 <= convergence["epochs_run"] <= 4
        assert convergence["early_stopping_patience"] == 1


def test_the_convergence_flags_are_on_the_three_commands():
    parser = build_parser()
    for command in ("global-train", "combined-train", "local-finetune"):
        args = parser.parse_args(
            [command, "--early-stopping-patience", "10", "--min-epochs", "20"]
        )
        assert args.early_stopping_patience == 10 and args.min_epochs == 20
        plain = parser.parse_args([command])
        assert plain.early_stopping_patience is None and plain.min_epochs == 0


# --------------------------------------------------------------------------- #
# P0.2 - fold plumbing
# --------------------------------------------------------------------------- #
def test_no_fold_leaves_every_seed_untouched():
    """The bit-for-bit guarantee: without a fold nothing about a run changes."""
    from federated_outlier_adaptation.utils.seeding import fold_seed, seed_suffix

    assert fold_seed(42, None) == 42
    assert fold_seed(None, None) is None
    assert seed_suffix(3) == "seed_3"
    assert seed_suffix(None) == ""


def test_a_fold_changes_the_seed_and_is_stable(monkeypatch):
    from federated_outlier_adaptation.utils.seeding import fold_seed

    assert fold_seed(42, 1) != 42
    assert fold_seed(42, 1) == fold_seed(42, 1)
    assert len({fold_seed(42, fold) for fold in range(1, 6)}) == 5
    # a seed of None with a fold is still a complete description of a draw
    assert fold_seed(None, 2) is not None


def test_adjacent_folds_and_seeds_do_not_alias():
    from federated_outlier_adaptation.utils.seeding import fold_seed

    assert fold_seed(1, 2) != fold_seed(2, 1)


def test_the_fold_is_part_of_the_output_path(monkeypatch):
    from federated_outlier_adaptation.utils.seeding import seed_suffix

    monkeypatch.setenv("FOA_FOLD", "3")
    assert seed_suffix(1) == "fold3_seed_1"
    assert seed_suffix(None) == "fold3"
    # ...and folds never collide with each other
    monkeypatch.setenv("FOA_FOLD", "4")
    assert seed_suffix(1) == "fold4_seed_1"


def test_an_impossible_fold_is_rejected(monkeypatch):
    from federated_outlier_adaptation import config

    monkeypatch.setenv("FOA_FOLD", "6")
    with pytest.raises(ValueError, match="expected a fold"):
        config.fold()
    monkeypatch.setenv("FOA_FOLD", "0")
    with pytest.raises(ValueError, match="expected a fold"):
        config.fold()


def test_the_run_seed_follows_the_fold(monkeypatch):
    import random as _random

    from federated_outlier_adaptation.utils.seeding import set_run_seed

    plain = set_run_seed(7)
    monkeypatch.setenv("FOA_FOLD", "2")
    folded = set_run_seed(7)
    assert plain == 7 and folded != 7
    # ...and a seedless run under a fold is still seeded
    assert set_run_seed(None) is not None


def test_the_fold_flag_is_on_the_command_line():
    parser = build_parser()
    assert parser.parse_args(["global-train", "--fold", "5"]).fold == 5
    assert parser.parse_args(["global-train"]).fold is None
    with pytest.raises(SystemExit):
        parser.parse_args(["global-train", "--fold", "6"])


# --------------------------------------------------------------------------- #
# v6 stage 1: g-init over the whole dataset
# --------------------------------------------------------------------------- #
def test_the_population_switch_is_on_the_command_line():
    parser = build_parser()
    args = parser.parse_args(["global-train", "--population", "all"])
    assert args.population == "all"
    assert parser.parse_args(["global-train"]).population == "source"
    with pytest.raises(SystemExit):
        parser.parse_args(["global-train", "--population", "some"])


def test_an_unknown_population_is_refused(synthetic_provider):
    from federated_outlier_adaptation.training.global_model import run_global_training

    with pytest.raises(ValueError, match="Unknown population"):
        run_global_training(epochs=1, provider=synthetic_provider, population="half")


def test_the_population_reaches_the_pipeline(monkeypatch):
    from federated_outlier_adaptation import cli
    from federated_outlier_adaptation.training import global_model

    seen = {}
    monkeypatch.setattr(cli, "_resolve_provider", lambda args: None)
    monkeypatch.setattr(
        global_model, "run_global_training", lambda **kwargs: seen.update(kwargs)
    )
    cli.cmd_global_train(
        build_parser().parse_args(
            ["global-train", "--population", "all", "--epochs", "3",
             "--early-stopping-patience", "10", "--min-epochs", "20"]
        )
    )
    assert seen["population"] == "all"
    assert seen["patience"] == 10 and seen["min_epochs"] == 20


# --------------------------------------------------------------------------- #
# v6 stage 2: choosing a fold, scoring writers, drawing the old data
# --------------------------------------------------------------------------- #
def _fold_run(root, prefix, fold, val, test, model_bytes=b"weights"):
    """A directory shaped like one fold of a cross-validated training."""
    import json

    directory = root / f"{prefix}_fold{fold}"
    (directory / "global_results").mkdir(parents=True, exist_ok=True)
    with open(directory / "global_results" / "global_metrics.json", "w") as handle:
        json.dump(
            {
                "val_accuracies": [val - 0.1, val],
                "test_accuracy": test,
                "convergence": {
                    "best_val_accuracy": val,
                    "best_epoch": 1,
                    "epochs_run": 2,
                    "stopped_early": False,
                },
            },
            handle,
        )
    (directory / "global_model").write_bytes(model_bytes + str(fold).encode())
    return directory


def test_the_best_validation_fold_wins(tmp_path):
    from federated_outlier_adaptation.analysis.fold_selection import select_best_fold

    for fold, val in ((1, 0.80), (2, 0.91), (3, 0.85)):
        _fold_run(tmp_path, "ginit", fold, val=val, test=val - 0.02)

    payload = select_best_fold(tmp_path, prefix="ginit", name="ginit", folds=(1, 2, 3))
    assert payload["selected_fold"] == 2
    assert payload["n_folds"] == 3
    assert payload["val_mean"] == pytest.approx((0.80 + 0.91 + 0.85) / 3)
    assert payload["val_spread"] > 0
    # the chosen fold's checkpoint is what was persisted, byte for byte
    assert (tmp_path / "ginit_model").read_bytes() == (
        tmp_path / "ginit_fold2" / "global_model"
    ).read_bytes()
    assert payload["sha256"]


def test_the_selection_is_recorded_next_to_the_model(tmp_path):
    import json

    from federated_outlier_adaptation.analysis.fold_selection import select_best_fold

    for fold, val in ((1, 0.5), (2, 0.6)):
        _fold_run(tmp_path, "g0", fold, val=val, test=val)
    select_best_fold(tmp_path, prefix="g0", name="g0", folds=(1, 2))

    with open(tmp_path / "g0_selection.json") as handle:
        stored = json.load(handle)
    assert stored["selected_fold"] == 2
    assert stored["rule"].startswith("highest validation accuracy")
    assert [entry["fold"] for entry in stored["folds"]] == [1, 2]
    assert stored["test_mean"] == pytest.approx(0.55)


def test_ties_break_on_the_lower_fold(tmp_path):
    from federated_outlier_adaptation.analysis.fold_selection import select_best_fold

    for fold in (1, 2, 3):
        _fold_run(tmp_path, "g", fold, val=0.7, test=0.7)
    assert select_best_fold(tmp_path, prefix="g", name="g", folds=(1, 2, 3))[
        "selected_fold"
    ] == 1


def test_selecting_before_any_fold_ran_says_so(tmp_path):
    from federated_outlier_adaptation.analysis.fold_selection import select_best_fold

    with pytest.raises(FileNotFoundError, match="Run the folds"):
        select_best_fold(tmp_path, prefix="ginit", name="ginit", folds=(1, 2))


def test_a_missing_checkpoint_is_reported(tmp_path):
    from federated_outlier_adaptation.analysis.fold_selection import select_best_fold

    directory = _fold_run(tmp_path, "g", 1, val=0.9, test=0.9)
    (directory / "global_model").unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint is not at"):
        select_best_fold(tmp_path, prefix="g", name="g", folds=(1,))


def test_the_old_data_draw_is_seeded_and_documented():
    from federated_outlier_adaptation.outliers.scoring import draw_old_data

    counts = {f"w{i:03d}": 50 + i for i in range(60)}
    payload = draw_old_data(counts, exclude=["w000"], size=10, seed=7, min_samples=60)

    assert len(payload["clients"]) == 10
    assert payload["clients"] == sorted(payload["clients"])
    assert "w000" not in payload["clients"]
    # the eligibility rule is the record, not folklore
    assert payload["seed"] == 7 and payload["min_samples"] == 60
    assert all(counts[w] >= 60 for w in payload["clients"])
    assert payload["excluded"] == ["w000"]
    # ...and it is a draw, so it repeats
    again = draw_old_data(counts, exclude=["w000"], size=10, seed=7, min_samples=60)
    assert again["clients"] == payload["clients"]
    other = draw_old_data(counts, exclude=["w000"], size=10, seed=8, min_samples=60)
    assert other["clients"] != payload["clients"]


def test_the_draw_refuses_when_too_few_are_eligible():
    from federated_outlier_adaptation.outliers.scoring import draw_old_data

    with pytest.raises(ValueError, match="Only .* eligible"):
        draw_old_data({f"w{i}": 10 for i in range(5)}, exclude=[], size=10, min_samples=100)


def test_eligibility_can_require_a_trainable_writer():
    from federated_outlier_adaptation.outliers.scoring import eligible_for_old_data

    counts = {"a": 200, "b": 200, "c": 50}
    assert eligible_for_old_data(counts, exclude=[], min_samples=100) == ["a", "b"]
    assert eligible_for_old_data(
        counts, exclude=[], min_samples=100, splittable=["a"]
    ) == ["a"]
    assert eligible_for_old_data(counts, exclude=["a"], min_samples=100) == ["b"]


def test_the_stage_two_and_three_commands_are_wired_up():
    parser = build_parser()
    for argv, command in (
        (["select-fold", "--prefix", "ginit", "--name", "ginit"], "select-fold"),
        (["score-writers", "--model-path", "/m", "--fold-book", "/b", "--fold", "3"],
         "score-writers"),
        (["draw-old-data", "--size", "100", "--seed", "1"], "draw-old-data"),
        (["evaluate-book", "--model-path", "/m", "--fold-book", "/b", "--fold", "1"],
         "evaluate-book"),
    ):
        assert parser.parse_args(argv).command == command
    trained = parser.parse_args(
        ["global-train", "--population", "file", "--writers-file", "/w.json"]
    )
    assert trained.population == "file" and trained.writers_file == "/w.json"


def test_training_on_a_file_needs_the_file(synthetic_provider):
    from federated_outlier_adaptation.training.global_model import run_global_training

    with pytest.raises(ValueError, match="needs --writers-file"):
        run_global_training(epochs=1, provider=synthetic_provider, population="file")


def test_the_two_commands_say_what_they_need(monkeypatch):
    from federated_outlier_adaptation import cli

    args = build_parser().parse_args(["score-writers", "--model-path", "/m"])
    with pytest.raises(SystemExit, match="needs --fold-book"):
        cli.cmd_score_writers(args)
    args = build_parser().parse_args(
        ["evaluate-book", "--model-path", "/m", "--fold-book", "/b"]
    )
    with pytest.raises(SystemExit, match="needs --fold K"):
        cli.cmd_evaluate_book(args)


# --------------------------------------------------------------------------- #
# the test loader must never contain training rows
# --------------------------------------------------------------------------- #
def test_the_source_path_no_longer_tests_on_its_own_training_data(monkeypatch):
    """
    The published pipeline built its test loader with rates 0.0/0.0, which in a
    stratified split means 'everything is test' - so the reported test accuracy
    was measured on the writers' whole data, the 60 % just trained on included.
    That is what made v5's theta_g report a test accuracy above its own best
    validation accuracy.  It is now one 60/20/20 call.
    """
    import inspect

    from federated_outlier_adaptation.training import global_model

    source = inspect.getsource(global_model.run_global_training)
    # the inflated construction is gone...
    assert "train_rate=0.0, eval_rate=0.0" not in source
    # ...and the three loaders come from one honest split
    assert "train_loader, eval_loader, test_loader = dataset.build_dataset(" in source
    assert "train_rate=0.6, eval_rate=0.2" in source


def test_a_fold_can_be_reselected_without_retraining(tmp_path):
    """
    When the evaluation was wrong but the checkpoints were not, the folds are
    re-scored from an evaluate-book file and re-selected; nothing retrains.
    """
    import json

    from federated_outlier_adaptation.analysis.fold_selection import select_best_fold

    # fold 1 looked best under the broken evaluation...
    _fold_run(tmp_path, "ginit", 1, val=0.92, test=0.75)
    _fold_run(tmp_path, "ginit", 2, val=0.91, test=0.74)

    # ...and the corrected scoring reverses them
    evaluations = tmp_path / "recomputed.json"
    with open(evaluations, "w") as handle:
        json.dump(
            {
                "ginit_fold1_val": {"fold": 1, "part": "val", "accuracy": 0.80},
                "ginit_fold1_test": {"fold": 1, "part": "test", "accuracy": 0.79},
                "ginit_fold2_val": {"fold": 2, "part": "val", "accuracy": 0.88},
                "ginit_fold2_test": {"fold": 2, "part": "test", "accuracy": 0.87},
            },
            handle,
        )

    payload = select_best_fold(
        tmp_path, prefix="ginit", name="ginit", folds=(1, 2), evaluations=evaluations
    )
    assert payload["accuracy_source"] == "recomputed"
    assert payload["selected_fold"] == 2
    assert payload["val_mean"] == pytest.approx(0.84)
    assert payload["test_mean"] == pytest.approx(0.83)
    # what the training had reported is kept next to what replaced it
    reported = {entry["fold"]: entry.get("reported_test_accuracy") for entry in payload["folds"]}
    assert reported == {1: 0.75, 2: 0.74}
    # and the winning checkpoint is the one that was copied
    assert (tmp_path / "ginit_model").read_bytes() == (
        tmp_path / "ginit_fold2" / "global_model"
    ).read_bytes()


def test_without_evaluations_the_training_metrics_are_used(tmp_path):
    from federated_outlier_adaptation.analysis.fold_selection import select_best_fold

    _fold_run(tmp_path, "g", 1, val=0.5, test=0.5)
    payload = select_best_fold(tmp_path, prefix="g", name="g", folds=(1,))
    assert payload["accuracy_source"] == "training metrics"


def test_the_recompute_flag_is_on_the_command_line():
    parser = build_parser()
    args = parser.parse_args(
        ["select-fold", "--prefix", "ginit", "--name", "ginit",
         "--from-evaluations", "/e.json"]
    )
    assert args.from_evaluations == "/e.json"
    assert parser.parse_args(
        ["select-fold", "--prefix", "g", "--name", "g"]
    ).from_evaluations is None
