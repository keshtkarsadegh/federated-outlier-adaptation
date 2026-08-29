"""
Experiment plans: task counts, the shape of the generated command lines and the
resume contract of the job array.
"""

from __future__ import annotations

import pytest

from federated_outlier_adaptation.cli import build_parser
from federated_outlier_adaptation.training import matrix

EXPECTED_COUNTS = {
    # 6 trainers x 5 seeds, plus the softer KD variant and the double case
    "e1_seeds": 6 * 5 + 5 + 5,
    "e2_e5_grids": 2,
    "e2_e5_finals": 2 * 5,
    "e6_pools": 2,
    "pool": 1,
    # 2 alternative policies x 3 round sizes x 7 trainers x 3 seeds
    "e6_population": 2 * 3 * 7 * 3,
    "fixed_cohort": 10 * 5,
    "pool_confirm_grids": 2,
    # (10 rules + 3 weightings + 2 orders) x 2 trainers x 3 seeds
    "aggregation_hypotheses": (10 + 3 + 2) * 2 * 3,
    # 8 spaces x 2 anchors x 2 grid seeds; the finals need the selected lambda,
    # so without stored selections the plan ends with one `foa select` line
    "regularisation_family": 8 * 2 * 2 + 1,
    "references": 3 + 3,
    # 3 distance spaces, one anchor, one seed
    "lean_nist_sweeps": 3,
    # (FedAvg + early-stopped FedAvg) x 5 seeds, plus 2 policies x 2 arms x 3 seeds
    "lean_nist_finals_pre": 2 * 5 + 2 * 2 * 3,
    # the finals need the selected lambda, so the plan is one select line
    "lean_nist_reg_finals": 1,
    # 10 hypotheses x 2 arms x 3 seeds
    "lean_agg_hypotheses": 10 * 2 * 3,
    # 5 sweeps + the select line + 6 hypotheses + 2 policies x 2 arms
    "lean_replica": 5 + 1 + 6 + 4,
    # pool + federated pre-training + local fine-tuning + 9 trainers
    # + 2 extended rules + 1 cyclic order + 1 extreme + 1 sweep
    # + the constrained selection + the signal analysis
    "smoke": 1 + 1 + 1 + 9 + 2 + 1 + 1 + 1 + 1 + 1,
    "dual": 3,
    "regen_base": 4,
    "regen_grids": 9 + 3,
    # 3 round sizes x 7 trainers x (5 fedavg + 3 capped) seeds, plus 7 x 3 extended
    "regen_finals": 3 * 7 * 5 + 3 * 7 * 3 + 7 * 3,
    "regen_extreme": 3 * 3,
    "regen_all": 1 + 4 + 12 + 2 + 189 + 9,
}


@pytest.mark.parametrize("plan,expected", sorted(EXPECTED_COUNTS.items()))
def test_plan_task_counts(plan, expected):
    assert len(matrix.plan_tasks(plan)) == expected


def test_every_plan_is_covered_by_the_expectations():
    assert set(matrix.PLANS) == set(EXPECTED_COUNTS)


def test_unknown_plan_raises():
    with pytest.raises(KeyError):
        matrix.plan_tasks("nope")


@pytest.mark.parametrize("plan", sorted(EXPECTED_COUNTS))
def test_every_task_is_a_foa_command(plan):
    for task in matrix.plan_tasks(plan):
        assert task.startswith("foa ")
        assert "\n" not in task


@pytest.mark.parametrize("plan", sorted(EXPECTED_COUNTS))
def test_training_tasks_are_resumable(plan):
    for task in matrix.plan_tasks(plan):
        if task.startswith(("foa final", "foa extreme", "foa grid", "foa base-fl", "foa all-aggs")):
            assert "--skip-existing" in task, task


@pytest.mark.parametrize("plan", sorted(EXPECTED_COUNTS))
def test_every_task_parses_as_a_cli_invocation(plan):
    import shlex

    parser = build_parser()
    for task in matrix.plan_tasks(plan):
        argv = shlex.split(task)[1:]
        parser.parse_args(argv)


def test_tasks_are_unique_per_plan():
    for plan in matrix.PLANS:
        tasks = matrix.plan_tasks(plan)
        assert len(set(tasks)) == len(tasks), plan


def test_e1_covers_every_seed_and_trainer():
    tasks = matrix.plan_tasks("e1_seeds")
    for trainer in matrix.E1_TRAINERS:
        for seed in matrix.E1_SEEDS:
            assert f"foa final --trainer {trainer} --seed {seed} --skip-existing" in tasks
    assert sum("--set T=4 alpha=0.95" in task for task in tasks) == len(matrix.E1_SEEDS)
    assert sum("--case double" in task for task in tasks) == len(matrix.E1_SEEDS)


def test_e1_kd_variant_writes_to_its_own_parent():
    tasks = [t for t in matrix.plan_tasks("e1_seeds") if "T=4" in t]
    assert tasks
    for task in tasks:
        assert "--parent final_result_T4" in task


def test_e6_policy_matrix():
    assert matrix.e6_policies(5, 1.0) == ("all",)
    assert matrix.e6_policies(20, 1.0) == ("all",)
    assert matrix.e6_policies(5, 0.2) == ("uniform", "worst_first")
    assert matrix.e6_policies(20, 0.5) == ("uniform", "worst_first")
    assert matrix.e6_policies(50, 0.2) == ("uniform", "worst_first", "round_robin")
    assert matrix.e6_policies(50, 1.0) == ("all",)


def test_e6_tasks_carry_the_population_options():
    tasks = matrix.plan_tasks("e6_population")
    assert all("--track-clients" in task for task in tasks)
    assert all("--clients-per-round" in task for task in tasks)
    assert all("--sampler-seed" in task for task in tasks)
    assert all(f"--pool-frac {matrix.POOL_FRAC:g}" in task for task in tasks)


def test_e6_sweeps_the_alternative_policies_only():
    """The default ``uniform`` policy is covered by ``regen_finals``."""
    policies = set()
    for task in matrix.plan_tasks("e6_population"):
        parts = task.split()
        policies.add(parts[parts.index("--policy") + 1])
    assert policies == {"worst_first", "round_robin"}


def test_e6_parents_are_distinct_per_configuration():
    parents = set()
    for task in matrix.plan_tasks("e6_population"):
        parts = task.split()
        parents.add(parts[parts.index("--parent") + 1])
    # 2 alternative policies x 3 round sizes
    assert len(parents) == 2 * len(matrix.CLIENTS_PER_ROUND)


def test_e6_covers_every_round_size_and_trainer():
    tasks = matrix.plan_tasks("e6_population")
    for clients in matrix.CLIENTS_PER_ROUND:
        assert any(f"--clients-per-round {clients} " in t + " " for t in tasks), clients
    for trainer in matrix.POOL_TRAINERS:
        assert any(f"--trainer {trainer} " in t for t in tasks), trainer


def test_regen_all_puts_grids_before_the_finals():
    tasks = matrix.plan_tasks("regen_all")
    last_grid = max(i for i, t in enumerate(tasks) if t.startswith("foa grid"))
    first_final = min(i for i, t in enumerate(tasks) if t.startswith("foa final"))
    assert last_grid < first_final


def test_regen_all_is_the_concatenation_of_its_parts():
    parts = []
    for plan in matrix.REGEN_ORDER:
        parts += matrix.plan_tasks(plan)
    assert matrix.plan_tasks("regen_all") == parts


def test_regen_all_builds_the_pool_first():
    tasks = matrix.plan_tasks("regen_all")
    assert tasks[0].startswith("foa select-outliers --mode pool")


def test_regen_base_covers_both_stopping_rules():
    tasks = matrix.plan_tasks("regen_base")
    assert sum("--stop-when-global-below-clients" in t for t in tasks) == 1
    assert any(t == "foa base-fl --skip-existing" for t in tasks)


def test_regen_base_runs_the_extended_aggregations_separately():
    tasks = matrix.plan_tasks("regen_base")
    extended = [t for t in tasks if "--extended-aggregations" in t]
    assert len(extended) == 1
    assert "--parent all_aggs_extended_fl" in extended[0]
    assert any(t == "foa all-aggs --skip-existing" for t in tasks)


def test_regen_finals_is_the_pool_setting():
    tasks = matrix.plan_tasks("regen_finals")
    assert all("--pool-frac" in t and "--clients-per-round" in t for t in tasks)
    assert all("--policy uniform" in t for t in tasks)
    for trainer in matrix.POOL_TRAINERS:
        assert any(f"--trainer {trainer} " in t for t in tasks), trainer
    assert sum("--extended-aggregations" in t for t in tasks) == (
        len(matrix.POOL_TRAINERS) * len(matrix.POOL_SEEDS)
    )


def test_fixed_cohort_is_the_published_setting():
    tasks = matrix.plan_tasks("fixed_cohort")
    assert all("--outliers-file" not in t and "--pool-frac" not in t for t in tasks)
    assert all("--clients-per-round" not in t for t in tasks)
    assert all("--policy" not in t for t in tasks)
    for trainer in matrix.ALL_TRAINERS:
        assert sum(f"--trainer {trainer} " in t for t in tasks) == len(matrix.E1_SEEDS)


def test_pool_plans_point_at_the_rule_based_pool():
    """The pool is named by its rule, so the task lines carry no absolute path."""
    for plan in ("regen_finals", "e6_population", "aggregation_hypotheses"):
        for task in matrix.plan_tasks(plan):
            assert f"--pool-frac {matrix.POOL_FRAC:g}" in task, plan
            assert "--outliers-file" not in task, plan


def test_aggregation_hypotheses_cover_every_rule():
    tasks = matrix.plan_tasks("aggregation_hypotheses")
    for rule in matrix.HYPOTHESIS_RULES:
        assert any(f"--aggregation {rule} " in t + " " for t in tasks), rule
    for weighting in matrix.HYPOTHESIS_WEIGHTINGS:
        assert any(f"--weighting {weighting}" in t for t in tasks), weighting
    for order in ("fixed", "shuffle"):
        assert any(f"--client-order {order}" in t for t in tasks), order
    # every rule outside the published set needs the opt-in flag
    for task in tasks:
        if "--aggregation con_delta_" in task:
            assert "--extended-aggregations" in task


def test_regularisation_family_sweeps_every_space_and_defers_its_finals():
    """Without stored selections the plan is the sweeps plus one select line."""
    tasks = matrix.plan_tasks("regularisation_family")
    assert all(t.startswith("foa grid") for t in tasks[:-1])
    assert not any(t.startswith("foa final") for t in tasks)
    for space in matrix.ANCHOR_SPACES:
        assert any(f"--space {space} " in t + " " for t in tasks), space

    select = tasks[-1]
    assert select.startswith("foa select ")
    assert "--emit-finals" in select
    for eps in matrix.SELECTION_EPS:
        assert f"{eps:g}" in select
    assert select.endswith(matrix.REG_FINALS_TASK_FILE)


def test_references_cover_both_baselines():
    tasks = matrix.plan_tasks("references")
    assert sum(t.startswith("foa local-finetune") for t in tasks) == len(matrix.POOL_SEEDS)
    assert sum("--init scratch" in t for t in tasks) == len(matrix.POOL_SEEDS)


def test_generated_sweeps_pin_their_gpu_fan_out():
    """
    Sixteen CUDA processes at batch 512 do not fit on a 40 GB accelerator, so
    every generated ``foa grid`` line carries a fan-out that does.
    """
    seen = 0
    for plan in matrix.PLANS:
        for task in matrix.plan_tasks(plan):
            if not task.startswith("foa grid "):
                continue
            seen += 1
            assert "--outer-workers" in task, task
            assert "--inner-workers" in task, task
            outer = int(task.split("--outer-workers ")[1].split()[0])
            inner = int(task.split("--inner-workers ")[1].split()[0])
            assert outer * inner <= 6, task
    assert seen


def test_worker_pinning_leaves_an_explicit_choice_alone():
    explicit = "foa grid --method kd --outer-workers 1 --inner-workers 1"
    assert matrix.with_grid_workers(explicit) == explicit
    assert matrix.with_grid_workers("foa final --trainer BaseTrainer") == (
        "foa final --trainer BaseTrainer"
    )
    assert matrix.with_grid_workers("foa grid --method kd").endswith(
        f"--outer-workers {matrix.GRID_OUTER_WORKERS} "
        f"--inner-workers {matrix.GRID_INNER_WORKERS}"
    )


def test_regen_grids_covers_every_method_and_case():
    tasks = matrix.plan_tasks("regen_grids")
    for method in matrix.GRID_METHODS:
        assert any(f"--method {method} " in t + " " for t in tasks), method
    for case in matrix.EXTREME_CASES:
        assert any(f"--case {case}" in t for t in tasks), case


# --------------------------------------------------------------------------- #
# the smoke plan
# --------------------------------------------------------------------------- #
#: Commands that carry neither a parent folder nor a resume flag.
SMOKE_EXEMPT = (
    "foa select-outliers",
    "foa global-train-fl",
    "foa select",
    "foa signals",
)


def _smoke_tasks():
    return matrix.plan_tasks("smoke")


def test_smoke_tasks_are_cheap():
    for task in _smoke_tasks():
        if task.startswith(("foa final", "foa extreme", "foa grid")):
            assert f"--rounds {matrix.SMOKE_ROUNDS}" in task, task
            assert f"--epochs {matrix.SMOKE_EPOCHS}" in task, task
            assert f"--outer-workers {matrix.SMOKE_OUTER_WORKERS}" in task, task
            assert f"--inner-workers {matrix.SMOKE_INNER_WORKERS}" in task, task


def test_smoke_tasks_write_under_a_smoke_parent_and_resume():
    for task in _smoke_tasks():
        if task.startswith(SMOKE_EXEMPT):
            continue
        assert f"--parent {matrix.SMOKE_PREFIX}_" in task, task
        assert "--skip-existing" in task, task


def test_smoke_never_writes_the_published_global_model():
    """The federated pre-training goes to its own model name."""
    pretraining = [t for t in _smoke_tasks() if t.startswith("foa global-train-fl")]
    assert len(pretraining) == 1
    assert f"--global-name {matrix.SMOKE_GLOBAL_NAME}" in pretraining[0]
    assert matrix.SMOKE_GLOBAL_NAME not in ("global", "global_fl")
    assert all("--global-name global " not in t + " " for t in _smoke_tasks())


def test_smoke_covers_every_trainer_on_the_pool_setting():
    tasks = [t for t in _smoke_tasks() if t.startswith("foa final")]
    for trainer in matrix.SMOKE_TRAINERS:
        matching = [t for t in tasks if f"--trainer {trainer} " in t]
        assert matching, trainer
        for task in matching:
            assert f"--pool-frac {matrix.POOL_FRAC:g}" in task
            assert f"--clients-per-round {matrix.SMOKE_CLIENTS_PER_ROUND}" in task
            assert "--policy uniform" in task
            assert "--track-clients" in task
            assert f"--seed {matrix.SMOKE_SEED}" in task


def test_smoke_configures_the_anchored_trainer():
    tasks = [t for t in _smoke_tasks() if "--trainer AnchoredTrainer" in t]
    assert len(tasks) == 1
    assert f'space="{matrix.SMOKE_ANCHOR_SPACE}"' in tasks[0]
    assert f'anchor="{matrix.SMOKE_ANCHOR_KIND}"' in tasks[0]


def test_smoke_covers_the_extended_rules_and_the_cyclic_order():
    tasks = _smoke_tasks()
    extended = [t for t in tasks if "--extended-aggregations" in t]
    assert len(extended) == 2
    assert any("--aggregation con_delta_anchor_lam01" in t for t in extended)
    assert any("--aggregation con_delta_median" in t for t in extended)
    assert sum("--client-order shuffle" in t for t in tasks) == 1


@pytest.mark.parametrize("rule", ["con_delta_anchor_lam01", "con_delta_median"])
def test_the_smoke_aggregation_rules_exist(rule):
    """A rule that no family owns would silently run zero jobs."""
    from federated_outlier_adaptation.aggregation.selector import (
        SKIP_FAMILY,
        resolve_aggregation,
    )

    resolved = [
        resolve_aggregation(scenario, metadata, rule, extended=True)
        for scenario in ("concurrent", "sequential")
        for metadata in ("weights", "delta")
    ]
    assert any(name != SKIP_FAMILY for name in resolved), rule


def test_an_unknown_aggregation_rule_is_rejected_instead_of_running_nothing():
    """
    ``con_delta_anchor`` is the parameterised helper, not a registered rule, so
    naming it used to resolve to "skip" in all four families and the run did
    nothing at all.  It now fails loudly.
    """
    from federated_outlier_adaptation.training.final_experiments import (
        generate_final_result_all_parallel,
    )

    with pytest.raises(ValueError, match="none of the four"):
        generate_final_result_all_parallel(
            trainer_name="BaseTrainer",
            aggregation="con_delta_anchor",
            extended_aggregations=True,
        )


def test_smoke_covers_the_extreme_case_and_the_sweep():
    tasks = _smoke_tasks()
    assert sum(t.startswith("foa extreme --case dual") for t in tasks) == 1
    grids = [t for t in tasks if t.startswith("foa grid")]
    assert len(grids) == 1
    assert f"--space {matrix.SMOKE_GRID_SPACE}" in grids[0]
    assert f"--anchor {matrix.SMOKE_GRID_ANCHOR}" in grids[0]




def test_smoke_ends_with_the_two_read_only_analyses():
    """
    They come last so that appending them never renumbers a training task.
    """
    tasks = _smoke_tasks()
    assert tasks[-2].startswith("foa select ")
    assert tasks[-1].startswith("foa signals ")
    assert all("--eps" in task for task in tasks[-2:])


def test_smoke_analyses_point_at_the_given_results_root(tmp_path):
    tasks = matrix.plan_tasks("smoke", results_dir=tmp_path)
    assert tasks[-2] == f"foa select --root {tmp_path} --eps 0.005"
    assert tasks[-1] == f"foa signals --root {tmp_path} --eps 0.005"


def test_smoke_starts_with_the_pool_and_the_reference_points():
    tasks = _smoke_tasks()
    assert tasks[0].startswith("foa select-outliers --mode pool")
    assert any(t.startswith("foa local-finetune") for t in tasks)




# --------------------------------------------------------------------------- #
# the lean plans
# --------------------------------------------------------------------------- #
def test_lean_sweeps_are_short_and_untracked():
    """
    A sweep only ranks configurations, so it pays for neither the long ranges
    nor the per-client evaluation.
    """
    tasks = matrix.plan_tasks("lean_nist_sweeps")
    assert len(tasks) == len(matrix.LEAN_SWEEP_SPACES)
    for task in tasks:
        assert "--track-clients" not in task
        assert "--anchor frozen" in task
        assert f"--seed {matrix.LEAN_SEED} " in task + " "
        assert "--lams " + " ".join(f"{lam:g}" for lam in matrix.LEAN_LAMS) in task
    kd = [task for task in tasks if "--space kd " in task + " "]
    assert kd and all(
        "--temperatures " + " ".join(f"{t:g}" for t in matrix.LEAN_TEMPERATURES) in task
        for task in kd
    )


def test_lean_sweeps_only_temper_the_softmax_spaces():
    from federated_outlier_adaptation.grid_search.anchored import TEMPERATURE_SPACES

    for task in matrix.plan_tasks("lean_replica"):
        if not task.startswith("foa grid"):
            continue
        space = task.split("--space ")[1].split()[0]
        assert ("--temperatures" in task) == (space in TEMPERATURE_SPACES), task


def test_lean_finals_compare_two_arms_with_the_same_stopping_rule():
    tasks = matrix.plan_tasks("lean_nist_finals_pre")
    baselines = [t for t in tasks if "--parent lean_m10_uniform_fedavg " in t + " "]
    stopped = [t for t in tasks if "--parent lean_m10_uniform_fedavg_es" in t]
    assert len(baselines) == len(matrix.LEAN_FINAL_SEEDS)
    assert len(stopped) == len(matrix.LEAN_FINAL_SEEDS)
    assert all("early_stopping=true" in t for t in stopped)
    assert all("early_stopping" not in t for t in baselines)


def test_lean_selection_study_covers_both_policies_and_both_arms():
    tasks = [t for t in matrix.plan_tasks("lean_nist_finals_pre") if "lean_sel_" in t]
    parents = {t.split("--parent ")[1].split()[0] for t in tasks}
    assert len(parents) == len(matrix.LEAN_POLICIES) * 2
    for policy in matrix.LEAN_POLICIES:
        assert any(f"--policy {policy}" in t for t in tasks), policy
    assert all(
        f"--clients-per-round {matrix.LEAN_SELECTION_CLIENTS}" in t for t in tasks
    )


def test_lean_hypotheses_cover_every_arm_once_per_seed():
    tasks = matrix.plan_tasks("lean_agg_hypotheses")
    for tag, _flags in matrix.LEAN_HYPOTHESES:
        matching = [t for t in tasks if f"_{tag}_" in t]
        assert len(matching) == 2 * len(matrix.LEAN_STUDY_SEEDS), tag
    assert any("--aggregation cyclic" in t for t in tasks)
    assert any("--client-order shuffle" in t for t in tasks)
    for task in tasks:
        if "--aggregation con_delta_" in task:
            assert "--extended-aggregations" in task


def test_the_cyclic_variant_runs_only_the_sequential_families():
    from federated_outlier_adaptation.aggregation.selector import (
        SKIP_FAMILY,
        resolve_aggregation,
    )

    assert resolve_aggregation("concurrent", "weights", "cyclic") == SKIP_FAMILY
    assert resolve_aggregation("concurrent", "delta", "cyclic") == SKIP_FAMILY
    assert resolve_aggregation("sequential", "weights", "cyclic") is None
    assert resolve_aggregation("sequential", "delta", "cyclic") is None


def test_lean_reg_finals_defer_to_the_selection():
    tasks = matrix.plan_tasks("lean_nist_reg_finals")
    assert len(tasks) == 1
    assert tasks[0].startswith("foa select ")
    assert "--emit-finals" in tasks[0]
    assert f"--eps {matrix.LEAN_EPS:g}" in tasks[0]




def test_lean_replica_sweeps_one_space_per_mechanism():
    tasks = [t for t in matrix.plan_tasks("lean_replica") if t.startswith("foa grid")]
    spaces = {t.split("--space ")[1].split()[0] for t in tasks}
    assert spaces == set(matrix.LEAN_REPLICA_SPACES)


# --------------------------------------------------------------------------- #
# resuming and the measured cost model
# --------------------------------------------------------------------------- #
def test_resume_keeps_only_the_lines_without_outputs(tmp_path):
    from federated_outlier_adaptation.training.tasks import expected_outputs, missing_tasks

    tasks = matrix.plan_tasks("lean_nist_finals_pre")[:3]
    assert missing_tasks(tasks, results_dir=tmp_path) == tasks

    for path in expected_outputs(tasks[0], results_dir=tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    assert missing_tasks(tasks, results_dir=tmp_path) == tasks[1:]


def test_resume_keeps_a_line_whose_outputs_are_only_partly_there(tmp_path):
    from federated_outlier_adaptation.training.tasks import expected_outputs, missing_tasks

    task = matrix.plan_tasks("lean_nist_finals_pre")[0]
    outputs = expected_outputs(task, results_dir=tmp_path)
    assert len(outputs) > 1
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text("{}")
    assert missing_tasks([task], results_dir=tmp_path) == [task]


def test_resume_keeps_lines_whose_output_it_cannot_predict(tmp_path):
    from federated_outlier_adaptation.training.tasks import missing_tasks

    lines = ["foa select-outliers --mode pool --pool-frac 0.05", "foa signals --eps 0.005"]
    assert missing_tasks(lines, results_dir=tmp_path) == lines


def test_expected_outputs_follow_the_skip_existing_contract(tmp_path):
    from federated_outlier_adaptation.training.tasks import expected_outputs

    grid = expected_outputs(matrix.plan_tasks("lean_nist_sweeps")[0], results_dir=tmp_path)
    assert grid and all(path.name.startswith("accuracies_points_") for path in grid)
    final = expected_outputs(matrix.plan_tasks("lean_nist_finals_pre")[0], results_dir=tmp_path)
    assert final and all(path.name == "summary_0.json" for path in final)


def test_a_task_expands_into_runs_and_rounds():
    from federated_outlier_adaptation.training.tasks import task_rounds, task_runs

    from federated_outlier_adaptation.grid_search.common import WEIGHTED_TASKS

    final = matrix.plan_tasks("lean_nist_finals_pre")[0]
    assert task_rounds(final) == 100
    # --aggregation fedavg runs the two families whose rules are distinct; the
    # other two are the same update written on weights instead of deltas.
    assert task_runs(final) == 2

    sweep = matrix.plan_tasks("lean_nist_sweeps")[0]
    assert task_rounds(sweep) == 50
    # lambdas x temperatures x the sweep's (scenario, metadata) tasks, which are
    # the same two rules the finals run.
    assert task_runs(sweep) == (
        len(matrix.LEAN_LAMS) * len(matrix.LEAN_TEMPERATURES) * len(WEIGHTED_TASKS)
    )


def test_the_measured_estimate_prices_runs_and_rounds():
    from federated_outlier_adaptation.training.tasks import plan_gpu_hours, task_runs

    tasks = matrix.plan_tasks("lean_nist_finals_pre")
    estimate = plan_gpu_hours(tasks, seconds_per_round=3.6)
    assert estimate["tasks"] == len(tasks)
    assert estimate["runs"] == sum(task_runs(task) for task in tasks)
    assert estimate["gpu_hours"] == pytest.approx(estimate["runs"] * 100 * 3.6 / 3600.0)
    assert estimate["unpriced_tasks"] == 0


def test_the_sweep_unit_cost_can_differ_from_the_finals_one():
    from federated_outlier_adaptation.training.tasks import plan_gpu_hours

    tasks = matrix.plan_tasks("lean_nist_sweeps")
    cheap = plan_gpu_hours(tasks, seconds_per_round=1.0, sweep_seconds_per_round=1.0)
    dear = plan_gpu_hours(tasks, seconds_per_round=1.0, sweep_seconds_per_round=2.0)
    assert dear["gpu_hours"] == pytest.approx(2 * cheap["gpu_hours"])


def test_a_select_line_is_counted_but_not_priced():
    from federated_outlier_adaptation.training.tasks import plan_gpu_hours

    estimate = plan_gpu_hours(matrix.plan_tasks("lean_nist_reg_finals"), seconds_per_round=3.6)
    assert estimate["tasks"] == 1
    assert estimate["unpriced_tasks"] == 1
    assert estimate["gpu_hours"] == 0.0


def test_gpu_hour_estimate():
    assert matrix.estimate_gpu_hours(0) == 0.0
    assert matrix.estimate_gpu_hours(5, minutes_per_task=12.0) == pytest.approx(1.0)
    assert matrix.estimate_gpu_hours(30, minutes_per_task=12.0) == pytest.approx(6.0)


def test_plan_summary_lists_every_plan():
    summary = matrix.plan_summary()
    assert {name for name, _, _ in summary} == set(matrix.PLANS)
    for name, count, hours in summary:
        assert count == EXPECTED_COUNTS[name]
        assert hours == pytest.approx(count * 12.0 / 60.0)


def test_write_tasks_round_trip(tmp_path):
    tasks = matrix.plan_tasks("dual")
    path = matrix.write_tasks(tasks, tmp_path / "sub" / "tasks.txt")
    assert path.is_file()
    with open(path) as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    assert lines == tasks


# --------------------------------------------------------------------------- #
# the provider dimension
# --------------------------------------------------------------------------- #
def test_the_published_dataset_emits_unchanged_task_lines():
    # The smoke plan deliberately visits all three datasets, so it is the one
    # plan whose task lines name a provider of their own.
    for plan in matrix.PLANS:
        assert matrix.plan_tasks(plan) == matrix.plan_tasks(plan, provider="nist")
        if plan == "smoke":
            continue
        assert all("--provider" not in task for task in matrix.plan_tasks(plan))








@pytest.mark.parametrize("provider", ["nist"])
@pytest.mark.parametrize("plan", sorted(EXPECTED_COUNTS))
def test_task_counts_do_not_depend_on_the_dataset(plan, provider):
    assert len(matrix.plan_tasks(plan, provider=provider)) == EXPECTED_COUNTS[plan]




def test_main_providers_are_registered():
    """
    The matrix must not name a provider the registry cannot build - that would
    be a plan for a dataset nothing can load.

    The converse is not required, and used to be asserted by mistake. The matrix
    is the prior pipeline's experiment planner; a provider added for a study of
    its own has no place in it, and forcing one in would generate plans nobody
    intends to run.
    """
    from federated_outlier_adaptation.providers import available_providers

    assert set(matrix.MAIN_PROVIDERS) <= set(available_providers())
    assert set(matrix.MAIN_PROVIDERS)


def test_main_providers_are_the_registered_ones():
    from federated_outlier_adaptation.providers import available_providers

    assert set(matrix.MAIN_PROVIDERS) == set(available_providers())
