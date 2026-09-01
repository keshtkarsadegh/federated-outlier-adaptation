#!/usr/bin/env python3
"""
SUPERSEDED - see tools/report_tables.py and tools/extreme_stopping.py.

This is the PRIOR programme's asset builder. It reads the ``d01_fl`` prefix,
which no folder under Digits_study01 carries, so the plain-FL rung it means to
supply comes back empty and the assets it writes are quietly short a row.

The current bundle is ``tools/report_tables.py --what all --csv <dir>`` for
every table and ``tools/extreme_stopping.py --fig`` for the extreme-case
figure; ``foa signals`` writes the Pareto plots. Kept as a record of what the
prior programme emitted, not repointed.

Every table and figure of the results sections, emitted from the stored runs.

Nothing here is typed in and nothing is retrained: each asset is built by
reading the study's run folders, so a number in the manuscript cannot drift
from the number in the artefact that produced it.

Screen runs and full-horizon runs share a folder prefix and are told apart by
the length of their per-round series: the screen is a short horizon, the finals
are the full budget. Only full-horizon runs reach a table or a figure.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

FOLDS = (1, 2, 3, 4, 5)
FULL_ROUNDS = 100


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.fmean(v) if v else None


def sd(v):
    v = [x for x in v if x is not None]
    return statistics.stdev(v) if len(v) > 1 else 0.0


def read_run(path: Path):
    """(cell, rounds, adaptation, preservation, per-round preservation)."""
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    for cell, body in payload.items():
        if not isinstance(body, dict) or "final_evaluation" not in body:
            continue
        fin = body["final_evaluation"]
        adapt = fin.get("clients", {}).get("accuracy")
        olds = fin.get("old", {}).get("accuracies", {})
        pres = mean(list(olds.values())) if olds else None
        series = body.get("source_val_accuracies") or []
        if adapt is None:
            continue
        return (cell, len(series), adapt, pres, series,
                body.get("heldout_client_accuracies") or [])
    return None


def collect_any(study: Path, prefix: str, lo: int = FULL_ROUNDS, hi: int = 10 ** 9):
    """As collect(), but for runs whose horizon falls inside [lo, hi]."""
    arms = defaultdict(dict)
    for folder in sorted(study.glob(f"{prefix}_*")):
        name = folder.name
        if "_fold" not in name:
            continue
        arm, _, rest = name.partition("_fold")
        arm = arm[len(prefix) + 1:]
        try:
            fold = int(rest[0])
        except (ValueError, IndexError):
            continue
        for summary in folder.rglob("summary_0.json"):
            got = read_run(summary)
            if got and lo <= got[1] <= hi:
                arms[arm][fold] = got
                break
    return arms


def collect(study: Path, prefix: str):
    """{arm: {fold: (cell, rounds, adapt, pres, series)}} for one stage."""
    arms = defaultdict(dict)
    for folder in sorted(study.glob(f"{prefix}_*")):
        name = folder.name
        if "_fold" not in name:
            continue
        arm, _, rest = name.partition("_fold")
        arm = arm[len(prefix) + 1:]
        try:
            fold = int(rest[0])
        except (ValueError, IndexError):
            continue
        for summary in folder.rglob("summary_0.json"):
            got = read_run(summary)
            if got and got[1] >= FULL_ROUNDS:
                arms[arm][fold] = got
                break
    return arms


def agg_arm(runs):
    """Fold-mean adaptation and preservation of one arm."""
    a = [v[2] for v in runs.values()]
    p = [v[3] for v in runs.values()]
    return mean(a), sd(a), mean(p), sd(p), len(runs)


def esc(text):
    return str(text).replace("_", r"\_")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    study = Path(args.study)
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The screen and the finals share a stem: d01_agg_* is the short-horizon
    # screen, d01_aggfull_* the full-horizon re-run of every method at its own
    # best setting. Only the finals belong in a table.
    agg = collect(study, "d01_aggfull")
    reg = collect(study, "d01_regfull")
    combo = collect(study, "d01_combo")
    fl = collect(study, "d01_fl")
    print(f"full-horizon arms: agg {len(agg)}  reg {len(reg)}  combo {len(combo)}  fl {len(fl)}")

    def family_of(runs, arm=""):
        if arm.startswith(("concurrent_", "sequential_")):
            return arm.split("_", 1)[0]
        cell = next(iter(runs.values()))[0]
        return "sequential" if "sequential" in cell else "concurrent"

    def display(arm):
        """The penalty finals carry their family in the name; the table has a
        column for that already, so it is not repeated in the cell."""
        for family in ("concurrent_", "sequential_"):
            if arm.startswith(family):
                return arm[len(family):]
        return arm

    # ------------------------------------------------------------- references
    def rung(folder, pattern):
        vals = []
        for path in sorted((study / folder).glob(pattern)):
            payload = json.loads(path.read_text())
            pooled = payload.get("pooled", {})
            a = pooled.get("own_weighted")
            if a is None and isinstance(pooled.get("own"), dict):
                a = pooled["own"].get("mean")
            p = payload.get("old_mean")
            if p is None:
                old = pooled.get("old")
                p = old.get("mean") if isinstance(old, dict) else old
            vals.append((a, p))
        return (mean([v[0] for v in vals]), mean([v[1] for v in vals]), len(vals)) if vals else (None, None, 0)

    g0 = json.loads((study / "g0_perfold_evaluations.json").read_text())
    g0_old = json.loads((study / "g0_evaluations.json").read_text())
    do_a = mean([g0[f"cohort_fold{f}"]["accuracy"] for f in FOLDS if f"cohort_fold{f}" in g0])
    do_p = mean([g0_old[f"old_data_fold{f}"]["accuracy"] for f in FOLDS if f"old_data_fold{f}" in g0_old])

    iso_g = rung("isolated", "isolated_global_*.json")
    iso_s = rung("isolated", "isolated_scratch_*.json")
    cen_g = rung("centralized", "centralized_outliers_global_*.json")
    cen_s = rung("centralized", "centralized_outliers_scratch_*.json")
    # A plain-FedAvg run folder holds BOTH round shapes, one cell each, so the
    # reference table reports them separately rather than taking whichever cell
    # happened to be read first.
    def fl_by_family():
        found = defaultdict(list)
        for folder in sorted(study.glob("d01_fl_global_fold*")):
            for summary in folder.rglob("summary_0.json"):
                payload = json.loads(summary.read_text())
                for cell, body in payload.items():
                    if not isinstance(body, dict) or "final_evaluation" not in body:
                        continue
                    fin = body["final_evaluation"]
                    olds = fin.get("old", {}).get("accuracies", {})
                    family = "sequential" if "sequential" in cell else "concurrent"
                    found[family].append((fin.get("clients", {}).get("accuracy"),
                                          mean(list(olds.values())) if olds else None))
        return {f: (mean([v[0] for v in rows]), mean([v[1] for v in rows]))
                for f, rows in found.items()}

    fl_fam = fl_by_family()

    rows = [("Shipped model, untouched", do_a, do_p),
            ("Isolated, from scratch", iso_s[0], iso_s[1]),
            ("Isolated, fine-tuned from $\\thg$", iso_g[0], iso_g[1]),
            ("Plain federated learning, parallel rounds",
             fl_fam.get("concurrent", (None, None))[0],
             fl_fam.get("concurrent", (None, None))[1]),
            ("Plain federated learning, cyclic rounds",
             fl_fam.get("sequential", (None, None))[0],
             fl_fam.get("sequential", (None, None))[1]),
            ("Centralized, from scratch", cen_s[0], cen_s[1]),
            ("Centralized, from $\\thg$ \\emph{(ceiling)}", cen_g[0], cen_g[1])]
    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{The reference rungs, each a mean over the five folds. The shipped",
             r"model fixes $F=0$ and $G=0$; centralized training is infeasible under the",
             r"constraint and is reported as a ceiling only.}",
             r"\label{tab:references}", r"\small",
             r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"Reference & Adaptation & Preservation \\", r"\midrule"]
    for label, a, p in rows:
        lines.append(f"{label} & {a:.4f} & {p:.4f} \\\\" if a is not None and p is not None
                     else f"{label} & --- & --- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tables" / "references.tex").write_text("\n".join(lines) + "\n")
    print("  tables/references.tex")

    # ------------------------------------------------ aggregation / reg tables
    def winners_table(arms, label, caption, tag, top=6):
        by_family = defaultdict(list)
        for arm, runs in arms.items():
            if len(runs) < 3:
                continue
            a, asd, p, psd, n = agg_arm(runs)
            if a is None or p is None:
                continue
            by_family[family_of(runs, arm)].append((display(arm), a, asd, p, psd, n))
        lines = [r"\begin{table}[tbp]", r"\centering", f"\\caption{{{caption}}}",
                 f"\\label{{tab:{tag}}}", r"\small",
                 r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
                 r"Family & " + label + r" & Adaptation & Preservation \\"]
        for family in ("concurrent", "sequential"):
            entries = sorted(by_family.get(family, []), key=lambda r: -r[1])[:top]
            if not entries:
                continue
            lines.append(r"\midrule")
            for i, (arm, a, asd, p, psd, n) in enumerate(entries):
                fam = family if i == 0 else ""
                lines.append(f"{fam} & \\code{{{esc(arm)}}} & {a:.4f} & {p:.4f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (out / "tables" / f"{tag}.tex").write_text("\n".join(lines) + "\n")
        print(f"  tables/{tag}.tex")
        return by_family

    agg_fam = winners_table(
        agg, "Server rule",
        "The best server rules at the full horizon, six per family, ordered by "
        "adaptation. Each cell is a mean over the five folds with a "
        "cross-entropy local objective throughout.", "agg_winners")
    reg_fam = winners_table(
        reg, "Local penalty",
        "The best client penalties at the full horizon, six per family, ordered "
        "by adaptation. Each cell is a mean over the five folds under plain "
        "FedAvg aggregation.", "reg_winners")

    # ------------------------------------------------------------ combinations
    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{The combination grid: the three best server rules crossed with the",
             r"three best penalties within each family, at the full horizon over five folds.",
             r"Families are never crossed, because a rule belongs to one round shape.}",
             r"\label{tab:combos}", r"\small",
             r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"Combination & Adaptation & Preservation \\", r"\midrule"]
    entries = []
    for arm, runs in combo.items():
        a, asd, p, psd, n = agg_arm(runs)
        if a is not None and p is not None:
            entries.append((arm, a, p, n))
    for arm, a, p, n in sorted(entries, key=lambda r: -r[1]):
        lines.append(f"\\code{{{esc(arm)}}} & {a:.4f} & {p:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tables" / "combos.tex").write_text("\n".join(lines) + "\n")
    print(f"  tables/combos.tex ({len(entries)} pairs)")

    # ---------------------------------------------------------------- extremes
    ext = collect(study, "d01_extreme")
    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{The smallest federations, winning combination, full participation,",
             r"mean over five folds. \emph{Double} gives the two worst writers one client",
             r"each; \emph{dual} merges their rows into a single client, so the two hold",
             r"byte-identical data and differ only in whether the aggregation sees them",
             r"separately.}", r"\label{tab:extreme}", r"\small",
             r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"Configuration & Adaptation & Preservation \\", r"\midrule"]
    for name in ("single", "double", "dual"):
        runs = ext.get(name, {})
        if not runs:
            lines.append(f"{name} & --- & --- \\\\")
            continue
        a, asd, p, psd, n = agg_arm(runs)
        lines.append(f"{name.capitalize()} & {a:.4f} & {p:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tables" / "extreme.tex").write_text("\n".join(lines) + "\n")
    print("  tables/extreme.tex")

    # ----------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200,
                         "axes.grid": True, "grid.alpha": 0.25})

    def pareto(by_family, title, path, highlight=()):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        marks = {"concurrent": ("o", "#1f77b4"), "sequential": ("^", "#d62728")}
        for family, entries in by_family.items():
            m, c = marks.get(family, ("s", "grey"))
            ax.scatter([e[3] for e in entries], [e[1] for e in entries],
                       s=16, marker=m, c=c, alpha=0.65, label=family, edgecolors="none")
        ax.axvline(do_p, ls=":", lw=1, c="k")
        ax.text(do_p, ax.get_ylim()[0], " shipped model", fontsize=7, va="bottom", rotation=90)
        for name, a, p in highlight:
            ax.scatter([p], [a], s=60, marker="*", c="#2ca02c", zorder=5)
            ax.annotate(name, (p, a), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("preservation (old-population test accuracy)")
        ax.set_ylabel("adaptation (cohort test accuracy)")
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        print(f"  {path.name}")

    pareto(agg_fam, "Server rules, full horizon", out / "figures" / "pareto_aggregation.pdf")
    pareto(reg_fam, "Client penalties, full horizon", out / "figures" / "pareto_regularisation.pdf")

    # traces
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    want = [("plain FedAvg", fl.get("global", {}), "#7f7f7f"),
            ("winner: trimmed 0.4 x feature_l2",
             combo.get("trimmed_0p4_feature_l2_lam0p1", {}), "#1f77b4"),
            ("balanced: anchor 0.03 x FedNTD",
             combo.get("anchor_0p03_ntd_b0p01_t0p5", {}), "#2ca02c")]
    for label, runs, colour in want:
        if not runs:
            continue
        series = [v[4] for v in runs.values() if v[4]]
        if not series:
            continue
        n = min(len(s) for s in series)
        curve = [mean([s[i] for s in series]) for i in range(n)]
        ax.plot(range(1, n + 1), curve, lw=1.4, c=colour, label=label)
    ax.axhline(do_p, ls=":", lw=1, c="k")
    ax.text(2, do_p, "shipped model", fontsize=7, va="bottom")
    ax.set_xlabel("round")
    ax.set_ylabel("source validation accuracy")
    ax.set_title("Preservation over the budget, mean of five folds", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "figures" / "traces_preservation.pdf")
    plt.close(fig)
    print("  traces_preservation.pdf")

    # scaling
    scal = json.loads((study / "tables" / "scaling_cv.json").read_text())
    main_w = agg_arm(combo.get("trimmed_0p4_feature_l2_lam0p1", {}))
    main_b = agg_arm(combo.get("anchor_0p03_ntd_b0p01_t0p5", {}))
    main_c = (fl_fam.get("concurrent", (None, None))[0], None,
              fl_fam.get("concurrent", (None, None))[1])
    points = {
        "winner": [(5, scal["five_winner"]), (10, {"cv_adapt": main_w[0], "cv_pres": main_w[2]}),
                   (20, scal["c20d10_winner"])],
        "balanced": [(5, scal["five_balanced"]), (10, {"cv_adapt": main_b[0], "cv_pres": main_b[2]}),
                     (20, scal["c20d10_balanced"])],
        "plain FedAvg": [(5, scal["five_control"]), (10, {"cv_adapt": main_c[0], "cv_pres": main_c[2]})],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    colours = {"winner": "#1f77b4", "balanced": "#2ca02c", "plain FedAvg": "#7f7f7f"}
    for label, series in points.items():
        xs = [p[0] for p in series]
        axes[0].plot(xs, [p[1]["cv_adapt"] for p in series], "o-", c=colours[label], label=label, lw=1.4)
        axes[1].plot(xs, [p[1]["cv_pres"] for p in series], "o-", c=colours[label], label=label, lw=1.4)
    axes[1].axhline(do_p, ls=":", lw=1, c="k")
    axes[1].text(5, do_p, " shipped model", fontsize=7, va="bottom")
    for ax, name in zip(axes, ("adaptation", "preservation")):
        ax.set_xlabel("clients in the federation")
        ax.set_ylabel(name)
        ax.set_xticks([5, 10, 20])
    axes[0].legend(frameon=False, fontsize=8, loc="lower center")
    fig.suptitle("Federation size, five-fold means", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figures" / "scaling_size.pdf")
    plt.close(fig)
    print("  scaling_size.pdf")

    # ------------------------------------------------------- appendix figures
    # 1. THE SCREEN AGAINST THE FULL HORIZON. The short screen and the real
    #    budget disagree, and they disagree systematically: the rules that carry
    #    server-side momentum lead early and fall back. This is the figure
    #    behind the decision not to filter any method at screening time.
    screen = collect_any(study, "d01_agg", lo=1, hi=FULL_ROUNDS - 1)
    pairs = []
    for arm, runs in screen.items():
        if arm not in agg:
            continue
        s = mean([v[2] for v in runs.values()])
        f = mean([v[2] for v in agg[arm].values()])
        if s is not None and f is not None:
            pairs.append((arm, s, f))
    if pairs:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for arm, s, f in pairs:
            momentum = arm.startswith(("fedavgm", "fedadam", "fedyogi", "seq_mix"))
            ax.scatter([s], [f], s=22, marker="s" if momentum else "o",
                       c="#d62728" if momentum else "#1f77b4", alpha=0.75,
                       edgecolors="none")
        lo = min(min(p[1] for p in pairs), min(p[2] for p in pairs))
        hi = max(max(p[1] for p in pairs), max(p[2] for p in pairs))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1, c="k")
        ax.scatter([], [], s=22, marker="s", c="#d62728", label="momentum or mixing rule")
        ax.scatter([], [], s=22, marker="o", c="#1f77b4", label="other rules")
        ax.set_xlabel("adaptation at the screen horizon")
        ax.set_ylabel("adaptation at the full horizon")
        ax.set_title("The screen does not rank the finals", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "figures" / "screen_vs_full.pdf")
        plt.close(fig)
        print(f"  screen_vs_full.pdf ({len(pairs)} rules)")

    # 2. ADAPTATION over the budget, the companion of the preservation traces.
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for label, runs, colour in want:
        if not runs:
            continue
        series = []
        for folder_runs in runs.values():
            pass
        for arm_runs in [runs]:
            for v in arm_runs.values():
                pass
        curves = []
        for v in runs.values():
            body_series = v[5] if len(v) > 5 else None
            if body_series:
                curves.append(body_series)
        if not curves:
            continue
        n = min(len(c) for c in curves)
        ax.plot(range(1, n + 1), [mean([c[i] for c in curves]) for i in range(n)],
                lw=1.4, c=colour, label=label)
    ax.axhline(do_a, ls=":", lw=1, c="k")
    ax.text(2, do_a, "shipped model", fontsize=7, va="bottom")
    ax.set_xlabel("round")
    ax.set_ylabel("cohort held-out accuracy")
    ax.set_title("Adaptation over the budget, mean of five folds", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "figures" / "curves_adaptation.pdf")
    plt.close(fig)
    print("  curves_adaptation.pdf")

    # 3. THE SMALLEST FEDERATIONS. single and dual share a forgetting slope;
    #    double, which holds the same rows behind a client boundary, resists.
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, colour in (("single", "#7f7f7f"), ("dual", "#d62728"), ("double", "#1f77b4")):
        runs = ext.get(name, {})
        curves = [v[4] for v in runs.values() if v[4]]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        ax.plot(range(1, n + 1), [mean([c[i] for c in curves]) for i in range(n)],
                lw=1.4, c=colour, label=name)
    ax.axhline(do_p, ls=":", lw=1, c="k")
    ax.set_xlabel("round")
    ax.set_ylabel("source validation accuracy")
    ax.set_title("Preservation in the smallest federations", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "figures" / "curves_extreme.pdf")
    plt.close(fig)
    print("  curves_extreme.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
