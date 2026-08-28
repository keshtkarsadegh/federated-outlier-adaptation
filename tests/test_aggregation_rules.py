"""
Algebra of the aggregation rules.

Several parallel and cyclic rules are written in weight form and in delta form
and are algebraically the same update.  Those pairs are asserted here on random
tensors, because the README documents them as equivalent and the paper reports
them as separate variants.

The server-optimiser rules are checked separately: they must keep state across
rounds, stay out of the default method listing, and reduce to plain FedAvg in
their first step when their momentum is empty.
"""

from __future__ import annotations

import pytest
import torch

from federated_outlier_adaptation.aggregation import concurrent_methods as con
from federated_outlier_adaptation.aggregation import sequential_methods as seq
from federated_outlier_adaptation.aggregation.selector import (
    AGG_VARIANTS,
    SKIP_FAMILY,
    list_method_names,
    resolve_aggregation,
    select_class,
)

KEYS = ("a", "b")
SHAPES = {"a": (3, 2), "b": (4,)}


def random_state(seed):
    generator = torch.Generator().manual_seed(seed)
    return {k: torch.randn(SHAPES[k], generator=generator) for k in KEYS}


@pytest.fixture()
def fixture_states():
    global_weights = random_state(0)
    clients = [random_state(i) for i in (1, 2, 3)]
    counts = [10, 25, 7]
    return global_weights, clients, counts


def assert_same(first, second):
    assert set(first) == set(second)
    for key in first:
        assert torch.allclose(first[key], second[key], atol=1e-6), key


# --------------------------------------------------------------------------- #
# parallel (concurrent) rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "weight_form,delta_form",
    [
        (con.con_weighted_cw, con.con_delta_weighted_cgd),
        (con.con_scaled_cw, con.con_delta_scaled_cgd),
        (con.con_capped_cw, con.con_delta_capped_cgd),
    ],
    ids=lambda f: f.__name__,
)
def test_parallel_weight_and_delta_forms_are_identical(fixture_states, weight_form, delta_form):
    """theta + sum_k p_k (theta_k - theta) = sum_k p_k theta_k  when sum_k p_k = 1."""
    global_weights, clients, counts = fixture_states
    assert_same(
        weight_form(global_weights, clients, counts),
        delta_form(global_weights, clients, counts),
    )


def test_parallel_client_global_rules_are_a_half_server_step(fixture_states):
    """The *_cgw rules are the *_cw rules with a server step size of 0.5."""
    global_weights, clients, counts = fixture_states
    for client_only, mixed in (
        (con.con_weighted_cw, con.con_weighted_cgw),
        (con.con_scaled_cw, con.con_scaled_cgw),
        (con.con_capped_cw, con.con_capped_cgw),
    ):
        pure = client_only(global_weights, clients, counts)
        blended = mixed(global_weights, clients, counts)
        expected = {
            key: global_weights[key] + 0.5 * (pure[key] - global_weights[key]) for key in pure
        }
        assert_same(blended, expected)


# --------------------------------------------------------------------------- #
# cyclic (sequential) rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "weight_form,delta_form,kwargs",
    [
        (
            seq.seq_fedavg_update,
            seq.seq_delta_fedavg_update,
            dict(client_sample_count=7, all_clients_samples=42, num_clients=3, index=1),
        ),
        (
            seq.seq_incremental_update,
            seq.seq_delta_progressive_update,
            dict(client_sample_count=7, all_clients_samples=42, num_clients=3, index=2),
        ),
    ],
    ids=["fedavg", "incremental"],
)
def test_cyclic_weight_and_delta_forms_are_identical(weight_form, delta_form, kwargs):
    """(1-a) theta + a theta_k = theta + a (theta_k - theta)."""
    global_weights = random_state(11)
    client = random_state(12)
    assert_same(
        weight_form(global_weights, client, **kwargs),
        delta_form(global_weights, client, **kwargs),
    )


def test_cyclic_first_client_of_the_incremental_rule_does_nothing():
    """alpha = index / (num_clients + index) is zero at index 0."""
    global_weights = random_state(13)
    client = random_state(14)
    updated = seq.seq_incremental_update(global_weights, client, num_clients=3, index=0)
    assert_same(updated, global_weights)


# --------------------------------------------------------------------------- #
# H1: explicit server step size
# --------------------------------------------------------------------------- #
def test_unit_server_step_is_fedavg(fixture_states):
    global_weights, clients, counts = fixture_states
    assert_same(
        con.con_delta_eta1(global_weights, clients, counts),
        con.con_weighted_cw(global_weights, clients, counts),
    )


def test_half_server_step_is_the_anchored_published_rule(fixture_states):
    """eta_s = 0.5 with proportional weights is exactly con_weighted_cgw."""
    global_weights, clients, counts = fixture_states
    assert_same(
        con.con_delta_eta05(global_weights, clients, counts),
        con.con_weighted_cgw(global_weights, clients, counts),
    )


def test_quarter_server_step_is_a_quarter_of_the_way(fixture_states):
    global_weights, clients, counts = fixture_states
    full = con.con_delta_eta1(global_weights, clients, counts)
    quarter = con.con_delta_eta025(global_weights, clients, counts)
    for key in global_weights:
        expected = global_weights[key] + 0.25 * (full[key] - global_weights[key])
        assert torch.allclose(quarter[key], expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# H2: weighting schemes
# --------------------------------------------------------------------------- #
def test_client_weights_sum_to_one():
    for weighting in con.WEIGHTINGS:
        weights = con.client_weights([10, 25, 7], weighting)
        assert sum(weights) == pytest.approx(1.0)
        assert len(weights) == 3


def test_uniform_weighting_ignores_the_sample_counts():
    assert con.client_weights([1, 100, 3], "uniform") == pytest.approx([1 / 3] * 3)


def test_capped_weighting_bounds_a_dominant_client():
    weights = con.client_weights([1000, 1, 1], "capped")
    assert weights[0] == pytest.approx(max(weights))
    assert weights[0] < 1000 / 1002


@pytest.mark.parametrize("weighting", list(con.WEIGHTINGS))
def test_the_server_step_combines_with_every_weighting(fixture_states, weighting):
    global_weights, clients, counts = fixture_states
    state = con.ServerState(weighting=weighting)
    updated = con.con_delta_eta1(global_weights, clients, counts, server_state=state)
    expected = {
        key: sum(
            weight * client[key]
            for client, weight in zip(clients, con.client_weights(counts, weighting))
        )
        for key in global_weights
    }
    assert_same(updated, expected)


def test_unknown_weighting_is_rejected():
    with pytest.raises(ValueError):
        con.ServerState(weighting="softmax")


# --------------------------------------------------------------------------- #
# H3: robust aggregators
# --------------------------------------------------------------------------- #
def test_median_equals_the_mean_when_every_client_agrees(fixture_states):
    global_weights, _, _ = fixture_states
    client = random_state(21)
    clients = [client, client, client]
    counts = [5, 5, 5]
    assert_same(
        con.con_delta_median(global_weights, clients, counts),
        con.con_delta_eta1(global_weights, clients, counts),
    )


def test_median_ignores_a_single_extreme_client(fixture_states):
    global_weights, clients, counts = fixture_states
    poisoned = list(clients)
    poisoned[0] = {k: v + 1000.0 for k, v in clients[0].items()}

    clean = con.con_delta_median(global_weights, clients, counts)
    attacked = con.con_delta_median(global_weights, poisoned, counts)
    mean_attacked = con.con_delta_eta1(global_weights, poisoned, counts)

    for key in global_weights:
        median_shift = (attacked[key] - clean[key]).abs().max()
        mean_shift = (mean_attacked[key] - global_weights[key]).abs().max()
        assert median_shift < mean_shift


def test_trimmed_mean_drops_the_extremes():
    global_weights = {"a": torch.zeros(1)}
    values = [-100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]
    clients = [{"a": torch.tensor([v])} for v in values]
    counts = [1] * len(values)
    trimmed = con.con_delta_trimmed_mean(global_weights, clients, counts)
    # 20% trimmed from each end leaves the six ones in the middle.
    assert trimmed["a"].item() == pytest.approx(1.0)
    plain = con.con_delta_eta1(global_weights, clients, counts)
    assert plain["a"].item() == pytest.approx(sum(values) / len(values), abs=1e-4)


def test_trimmed_mean_falls_back_to_the_mean_for_few_clients(fixture_states):
    global_weights, clients, counts = fixture_states
    uniform = [1, 1, 1]
    trimmed = con.con_delta_trimmed_mean(global_weights, clients, uniform)
    state = con.ServerState(weighting="uniform")
    plain = con.con_delta_eta1(global_weights, clients, uniform, server_state=state)
    assert_same(trimmed, plain)


# --------------------------------------------------------------------------- #
# H4: anchoring to the frozen global model
# --------------------------------------------------------------------------- #
def test_anchor_with_zero_lambda_is_the_plain_rule(fixture_states):
    global_weights, clients, counts = fixture_states
    state = con.ServerState()
    state.frozen_global = random_state(33)
    assert_same(
        con.con_delta_anchor(global_weights, clients, counts, server_state=state, lambda_s=0.0),
        con.con_delta_eta1(global_weights, clients, counts),
    )


def test_anchor_pulls_back_towards_the_frozen_model(fixture_states):
    global_weights, clients, counts = fixture_states
    frozen = random_state(44)
    state = con.ServerState()
    state.frozen_global = frozen

    plain = con.con_delta_eta1(global_weights, clients, counts)
    anchored = con.con_delta_anchor_lam05(global_weights, clients, counts, server_state=state)
    for key in global_weights:
        expected = plain[key] - 0.5 * (global_weights[key] - frozen[key])
        assert torch.allclose(anchored[key], expected, atol=1e-6)


def test_anchor_without_a_frozen_model_is_the_plain_rule(fixture_states):
    global_weights, clients, counts = fixture_states
    assert_same(
        con.con_delta_anchor_lam01(global_weights, clients, counts),
        con.con_delta_eta1(global_weights, clients, counts),
    )


# --------------------------------------------------------------------------- #
# variants and listing
# --------------------------------------------------------------------------- #
def test_every_rule_preserves_the_key_set_and_shapes(fixture_states):
    global_weights, clients, counts = fixture_states
    rules = [
        con.con_delta_eta025,
        con.con_delta_eta05,
        con.con_delta_eta1,
        con.con_delta_median,
        con.con_delta_trimmed_mean,
        con.con_delta_anchor_lam01,
        con.con_delta_anchor_lam05,
        con.con_delta_fedavgm,
        con.con_delta_fedadam,
        con.con_delta_fedyogi,
    ]
    for rule in rules:
        updated = rule(global_weights, clients, counts, server_state=con.ServerState())
        assert set(updated) == set(global_weights), rule.__name__
        for key in global_weights:
            assert updated[key].shape == global_weights[key].shape, rule.__name__


def test_aggregation_variants_pick_one_rule_per_family():
    """
    Every family a variant does not skip names a rule that exists there.

    ``fedavg`` skips two of the four families on purpose: their rules are
    algebraically the same update as the two it keeps, so running all four
    reported run-to-run noise as two settings.  See
    ``selector.FEDAVG_DUPLICATE_FAMILIES``.
    """
    for variant in ("fedavg", "anchored", "capped"):
        for scenario in ("concurrent", "sequential"):
            for metadata in ("weights", "delta"):
                name = resolve_aggregation(scenario, metadata, variant)
                if name == SKIP_FAMILY:
                    continue
                assert hasattr(select_class(scenario, metadata), name)
    assert set(AGG_VARIANTS) >= {"all", "fedavg", "anchored", "capped"}


def test_the_default_variant_runs_every_rule():
    for aggregation in (None, "all", "none", ""):
        assert resolve_aggregation("concurrent", "delta", aggregation) is None


def test_a_rule_missing_from_a_family_is_skipped():
    assert resolve_aggregation("concurrent", "delta", "con_delta_median") == "con_delta_median"
    assert resolve_aggregation("sequential", "weights", "con_delta_median") == SKIP_FAMILY


def test_server_rules_are_hidden_from_the_default_listing():
    cls = select_class("concurrent", "delta")
    default = list_method_names(cls)
    extended = list_method_names(cls, extended=True)
    assert set(default) == {
        "con_delta_weighted_cgd",
        "con_delta_scaled_cgd",
        "con_delta_capped_cgd",
    }
    for name in cls.EXTENDED_METHODS:
        assert name not in default
        assert name in extended


def test_other_families_are_unaffected_by_the_extended_flag():
    """
    The two families without opt-in rules list the same set either way.

    ``sequential``/``weights`` now has one - ``seq_mix_alpha``, the swept
    mixing weight that replaces the three hard-coded coefficients - so it is
    checked with the other opt-in families below instead.
    """
    for scenario, metadata in (
        ("concurrent", "weights"),
        ("sequential", "delta"),
    ):
        cls = select_class(scenario, metadata)
        assert list_method_names(cls) == list_method_names(cls, extended=True)


def test_the_swept_mixing_rule_is_opt_in():
    cls = select_class("sequential", "weights")
    assert "seq_mix_alpha" not in list_method_names(cls)
    assert "seq_mix_alpha" in list_method_names(cls, extended=True)


# --------------------------------------------------------------------------- #
# H5: server optimisers
# --------------------------------------------------------------------------- #
def test_fedavgm_first_step_is_plain_fedavg(fixture_states):
    """With an empty momentum buffer, m = Delta and eta_s = 1."""
    global_weights, clients, counts = fixture_states
    state = con.ServerOptimizerState()
    updated = con.con_delta_fedavgm(global_weights, clients, counts, server_state=state)
    assert_same(updated, con.con_delta_weighted_cgd(global_weights, clients, counts))
    assert state.step == 1


def test_fedavgm_accumulates_momentum(fixture_states):
    global_weights, clients, counts = fixture_states
    state = con.ServerOptimizerState()
    first = con.con_delta_fedavgm(global_weights, clients, counts, server_state=state)
    second = con.con_delta_fedavgm(global_weights, clients, counts, server_state=state)
    # m2 = beta m1 + Delta = (1 + beta) Delta, so the second step is larger.
    for key in global_weights:
        first_step = first[key] - global_weights[key]
        second_step = second[key] - global_weights[key]
        assert torch.allclose(second_step, (1 + con.SERVER_MOMENTUM) * first_step, atol=1e-6)
    assert state.step == 2


@pytest.mark.parametrize(
    "rule", [con.con_delta_fedadam, con.con_delta_fedyogi], ids=lambda f: f.__name__
)
def test_adaptive_server_rules_move_the_model_and_keep_state(fixture_states, rule):
    global_weights, clients, counts = fixture_states
    state = con.ServerOptimizerState()
    updated = rule(global_weights, clients, counts, server_state=state)

    assert set(updated) == set(global_weights)
    assert any(not torch.allclose(updated[k], global_weights[k]) for k in global_weights)
    assert set(state.momentum) == set(global_weights)
    assert set(state.second) == set(global_weights)
    assert state.step == 1

    rule(global_weights, clients, counts, server_state=state)
    assert state.step == 2


def test_adaptive_server_rules_take_a_bounded_step(fixture_states):
    """m / (sqrt(v) + tau) is bounded, so the step size is close to the server lr."""
    global_weights, clients, counts = fixture_states
    state = con.ServerOptimizerState()
    updated = con.con_delta_fedadam(global_weights, clients, counts, server_state=state)
    for key in global_weights:
        step = (updated[key] - global_weights[key]).abs().max().item()
        assert step <= con.SERVER_LR * 1.5


def test_server_state_reset_clears_the_buffers(fixture_states):
    global_weights, clients, counts = fixture_states
    state = con.ServerOptimizerState()
    con.con_delta_fedyogi(global_weights, clients, counts, server_state=state)
    state.reset()
    assert not state.momentum and not state.second and state.step == 0


def test_server_rules_work_without_a_state(fixture_states):
    """A missing state must not crash; it just makes the rule memoryless."""
    global_weights, clients, counts = fixture_states
    for rule in (con.con_delta_fedavgm, con.con_delta_fedadam, con.con_delta_fedyogi):
        updated = rule(global_weights, clients, counts)
        assert set(updated) == set(global_weights)
