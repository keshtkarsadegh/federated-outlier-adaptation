"""
The paper pipeline: every view a figure or a generator reads is shipped, and
shipped correctly.

The figures are the one part of the manuscript that used to be built by hand.
Now they are `tools/export_*.py` into `tables/paper_figures/` and `fig_*.py` out
of it, and each of the three joints between those pieces fails silently: a view
the exporter stopped writing still leaves the old CSV in the tree, a figure that
starts reading a new view finds nothing until somebody runs it, and a trace whose
arms disagree at round 0 draws a difference that is about the shipped model
rather than about the training.

`make_numbers.py` and `make_paper_tables.py` sit in the same directory and read
the same way, so the same joint is checked for them: they read thirty views
spread over four bundles, and a view that stopped shipping would be found by
the manuscript build rather than here.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
FIGURES = TOOLS / "paper_figures"
STUDY = REPO / "study" / "artifacts" / "Digits_study01"
TABLES = STUDY / "tables"
VIEWS = TABLES / "paper_figures"
PAPER = TABLES / "paper"

for entry in (str(TOOLS), str(FIGURES), str(REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _rows(name: str) -> list:
    with open(VIEWS / name, newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_view_the_exporters_name_is_shipped():
    """A view outside the tree is a figure nobody can redraw from the clone."""
    import export_baseline_views
    import export_combo_folds  # noqa: F401  - imported for its side-free presence
    import export_traces

    expected = set(export_traces.VIEWS) | set(export_baseline_views.VIEWS)
    expected.add("combos_folds")
    shipped = {path.stem for path in VIEWS.glob("*.csv")}
    assert expected <= shipped, f"named by an exporter, not in the tree: {sorted(expected - shipped)}"
    assert shipped <= expected, f"in the tree, written by no exporter: {sorted(shipped - expected)}"


def test_every_csv_a_figure_reads_resolves():
    """
    Resolved through `figstyle` itself rather than by looking in one directory.

    The views the figures need are split across two bundles - one written by
    `report_tables.py`, one by the exporters - and a test that knew only about
    the second would pass while `fig_problem` could not find `references.csv`.
    """
    import re

    import figstyle

    wanted = set()
    for script in sorted(FIGURES.glob("fig_*.py")):
        wanted |= set(re.findall(r'"([a-z0-9_]+\.csv)"', script.read_text()))
    assert wanted, "no figure reads any view; the pattern has stopped matching"
    for name in sorted(wanted):
        figstyle.locate(name)          # raises SystemExit when it is not there


def test_round_zero_is_one_shared_point_in_every_trace():
    """
    Every arm of a figure starts at the shipped model, and the rounds have no gap.

    `figstyle.read_traces` refuses a view that breaks either rule, so a stale one
    fails at figure time with a build half-run. It is cheaper to fail here.
    """
    for path in sorted(VIEWS.glob("traces_*.csv")):
        table = _rows(path.name)
        rounds = [int(row["round"]) for row in table]
        assert rounds == list(range(len(table))), f"{path.name}: rounds skip"
        arms = [column[:-4] for column in table[0] if column.endswith("_src")]
        assert arms, f"{path.name}: no arm column"
        starts = {(table[0][f"{arm}_src"], table[0][f"{arm}_cohort"]) for arm in arms}
        assert len(starts) == 1, f"{path.name}: round 0 is not one shared point"
        for arm in arms:
            assert f"{arm}_cohort" in table[0], f"{path.name}: {arm} has no cohort column"


def test_the_shipped_views_carry_the_columns_their_tools_declare():
    """A renamed column reaches a figure as a missing key, one build later."""
    import export_baseline_views
    import export_combo_folds

    for name, columns in (
            ("g0_training.csv", export_baseline_views.G0_COLUMNS),
            ("isolated_clients.csv", export_baseline_views.ISOLATED_COLUMNS),
            ("combos_folds.csv", export_combo_folds.COLUMNS)):
        with open(VIEWS / name, newline="") as handle:
            header = next(csv.reader(handle))
        assert header == list(columns), f"{name}: header is not what its tool writes"


def test_every_csv_the_numbers_and_the_tables_read_resolves():
    """
    The same check for the two generators, and it is a wider one.

    `numbers.tex` and the thirteen `tables/*.tex` are read out of thirty-three
    views that live in four different bundles because four different tools write
    them. A
    view that stopped shipping does not fail here unless it is looked for
    through the generator's own resolution, which is why the names are read out
    of the source and handed back to `locate`.
    """
    import re

    import make_numbers
    import make_paper_tables

    for module in (make_numbers, make_paper_tables):
        source = Path(module.__file__).read_text()
        wanted = set(re.findall(r'rows\("([A-Za-z0-9_.-]+\.csv)"\)', source))
        assert len(wanted) >= 15, f"{module.__name__}: the pattern stopped matching"
        for name in sorted(wanted):
            module.locate(name)        # raises SystemExit when it is not there


def test_the_view_names_built_by_interpolation_are_shipped_too():
    """
    `sizes_%s.csv` and `fairness_%s.csv` are assembled from a tag, so the literal
    name never appears in the source and the check above cannot see it. The tags
    are the four carry settings, named in the table generator itself.
    """
    import make_numbers
    import make_paper_tables

    tags = [tag for tag, _, _ in make_paper_tables.SETTING]
    assert sorted(tags) == ["c10d10", "c20d10", "c20d20", "five"], f"settings changed: {tags}"
    for tag in tags:
        make_paper_tables.locate(f"sizes_{tag}.csv")
        make_numbers.locate(f"sizes_{tag}.csv")
    for tag in ("c20d10", "c20d20"):
        make_numbers.locate(f"fairness_{tag}.csv")


def test_the_paper_bundle_wins_when_two_bundles_carry_one_name():
    """
    `tables/combos.csv` is the raw grid dump the stage left behind and
    `tables/paper/combos.csv` is the view the manuscript quotes: one name, two
    files, different rows. The order `_data_dirs` fixes is the only thing that
    keeps the manuscript reading the second, so it is pinned here.
    """
    import make_numbers

    assert (TABLES / "combos.csv").is_file() and (TABLES / "paper" / "combos.csv").is_file()
    assert make_numbers.locate("combos.csv") == str(TABLES / "paper" / "combos.csv")


def test_neither_generator_writes_into_the_checkout_by_default(monkeypatch):
    """
    Both rewrite LaTeX in place. Beside the script is the right answer in the
    manuscript checkout and the wrong one here, where beside the script is the
    tracked tree - so from a source checkout with no `FOA_PAPER_OUT` set the
    answer has to be a refusal rather than a default.
    """
    import make_numbers
    import make_paper_tables

    monkeypatch.delenv("FOA_PAPER_OUT", raising=False)
    for module in (make_numbers, make_paper_tables):
        with pytest.raises(SystemExit):
            module.out_dir()


def test_the_decoupling_view_is_rebuilt_from_the_shipped_records(tmp_path):
    """
    The one view whose generator was written after the file it produces.

    `decouple_example.csv` was cut by hand and the paragraph it feeds is the
    defence of the whole two-stage selection, so a tool that merely produces
    something of the same shape would be worse than no tool at all. It reads
    only `outliers/`, which ships in full, so the regeneration is checked byte
    for byte against the shipped file rather than field by field.
    """
    import export_decouple_example

    export_decouple_example.write_csv(tmp_path / "decouple_example.csv",
                                      export_decouple_example.rows_of(STUDY))
    assert (tmp_path / "decouple_example.csv").read_bytes() == \
        (PAPER / "decouple_example.csv").read_bytes()


def test_the_signal_extracts_carry_the_columns_their_tool_declares():
    """
    The signals extracts need the stored runs and cannot be rebuilt from the
    clone, so what is pinned here is the joint that would break silently: a
    column the tool stopped writing, or one the shipped file does not carry.
    """
    import export_signals_summary

    with open(PAPER / "signals_summary_extract.csv", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == list(export_signals_summary.COLUMNS)

    with open(PAPER / "signals_extras_extract.csv", newline="") as handle:
        keys = {row[0] for row in csv.reader(handle)}
    assert {"stopping_arms", "fixed_mean_score", "oracle_mean_score",
            "baseline_a0", "baseline_p0"} <= keys


def _generated_tables(tmp_path, monkeypatch):
    """The .tex the table generator actually writes, into a scratch directory."""
    import make_paper_tables

    monkeypatch.setenv("FOA_PAPER_OUT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["make_paper_tables.py"])
    make_paper_tables.main()
    return tmp_path / "tables"


def test_every_delta_the_combination_table_prints_lands_on_a_printed_row(tmp_path, monkeypatch):
    """
    Table 4's delta column has to be reproducible from Tables 2 and 3.

    `tab:combos` prints, for each of its eighteen rows, the score minus the
    score of whichever of the row's two halves scores higher on its own, and
    its note sends the reader to `tab:agg_winners` and `tab:reg_winners` to
    find that half. Six of the eighteen cross a COMPOSITE penalty, and a
    `tab:reg_winners` filtered to `not cell.startswith("hybrid")` printed none
    of the composites - so those six deltas could not be checked from the page,
    and the screened blend stood there as the only mixed penalty on it, inviting
    a recomputation against the wrong row that moves the count of combinations
    clearing their better half.

    So what is pinned here is the joint itself, on the emitted LaTeX rather
    than on the generator's intentions: for every combination, the parent the
    delta is taken against is a row one of the two tables prints, with that
    parent's own name and its own score on it.
    """
    import make_paper_tables as tables

    out = _generated_tables(tmp_path, monkeypatch)
    printed = {"reg": (out / "reg_winners.tex").read_text(),
               "agg": (out / "agg_winners.tex").read_text()}

    agg = tables.rows("agg-winners.csv")
    regu = tables.rows("reg-winners.csv")
    combo = tables.rows("combos.csv")
    agg_cells = {row["cell"] for row in agg}
    assert len(combo) == 18

    composites = 0
    for row in combo:
        rule, penalty = tables.split_combo(row["cell"], agg_cells)
        a = tables.pick(agg, cell=rule, family=row["family"])
        b = tables.pick(regu, cell=penalty, family=row["family"])
        parent = a if float(a["score"]) >= float(b["score"]) else b
        composites += parent["cell"].startswith("hybrid")

        name = tables.label(parent["cell"])
        score = "%.2f" % (100 * float(parent["score"]))
        where = printed["agg"] if parent is a else printed["reg"]
        assert [ln for ln in where.splitlines() if name in ln and score in ln], (
            f"{row['cell']}/{row['family']}: its delta is taken against "
            f"{parent['cell']} at {score}, which no printed row carries"
        )

    assert composites == 6, composites


def test_the_three_mixed_penalties_are_told_apart_by_name(tmp_path, monkeypatch):
    """
    Three different arms mix distillation with consolidation and all three
    used to render as "KD+EWC blend".

    The construction stage built a composite out of a schedule's own selected
    halves, inheriting their lambda and T and sweeping the mix alone, and did
    it twice - once per schedule. A separate screen swept lambda, T and m
    together. The paper crosses one composite, reports another as selected, and
    prints the screen beside both, so one name over the three left a reader no
    way to tell which arm a row was about. A composite cell also spells only
    its mix, so if paper_names does not carry its lambda and T, nothing does.
    """
    import paper_names

    arms = ("hybrid_seq_mix0p75", "hybrid_seq_mix0p5", "hybrid_mix0p5",
            "blend_lam0p1_T0p5_mix0p25", "blend_lam0p1_T0p25_mix0p5")
    names = [paper_names.label(cell) for cell in arms]
    assert len(set(names)) == len(names), names

    assert len({paper_names.short(cell) for cell in
                ("hybrid_mix0p5", "hybrid_seq_mix0p5",
                 "blend_lam0p1_T0p5_mix0p25")}) == 3

    for cell in ("hybrid_seq_mix0p75", "hybrid_mix0p5"):
        rendered = paper_names.label(cell)
        assert "\\lambda{=}" in rendered and "T{=}" in rendered, rendered

    out = _generated_tables(tmp_path, monkeypatch)
    reg = (out / "reg_winners.tex").read_text()
    for cell in ("hybrid_seq_mix0p75", "hybrid_seq_mix0p5", "hybrid_mix0p5",
                 "blend_lam0p1_T0p5_mix0p25", "blend_lam0p1_T0p25_mix0p5"):
        assert paper_names.label(cell) in reg, cell


def test_the_signals_table_prices_all_eight_signals_from_the_shipped_view(tmp_path, monkeypatch):
    """
    The signals subsection is written off `tables/signals.tex` and off nothing
    else, so every signal the study permits has to be on it and every cell of
    it has to be a cell of the view.

    Two joints break silently here. A signal the runner stopped recording, or
    one the exporter left without its stopping columns, would quietly shorten
    a table whose whole argument is that the best tracker and the best rule
    are different rows - eight of eight is the claim, not "the ones that had
    data that day". And a mean or a delta retyped into the emitter rather than
    read would put a number on the page that no file backs: the generator's
    own assertions hold the cell it reads to the file, and this holds the cell
    that reaches the page to the cell it was read from.
    """
    import make_paper_tables as tables
    import paper_names

    out = _generated_tables(tmp_path, monkeypatch)
    lines = (out / "signals.tex").read_text().splitlines()

    sigs = tables.rows("signals_summary_extract.csv")
    assert len(sigs) == 8, [row["signal"] for row in sigs]
    assert {row["signal"] for row in sigs} == set(tables.SIGNAL_ORDER)

    for row in sigs:
        name = paper_names.SIGNAL[row["signal"]]
        printed = [ln for ln in lines if ln.startswith(name)]
        assert len(printed) == 1, (row["signal"], printed)
        cells = [c.strip() for c in printed[0].rstrip("\\ ").split(" & ")]
        assert len(cells) == 8, cells

        gain = "%.2f" % (100 * float(row["mean_vs_fixed"]))
        assert cells[1] == row["observed_on"]
        assert cells[2] == "%.3f" % float(row["median_rho"])
        assert cells[3] == "%.1f" % (100 * float(row["share_ge_0p9"])) + r"\%"
        assert cells[4] == row["best_delta"]
        assert cells[5] == "%d" % int(row["arms_fired"])
        assert cells[6] == "%.2f" % (100 * float(row["mean_stopped_score"]))
        assert cells[7] == ("$%s$" % gain if gain.startswith("-")
                            else "$+%s$" % gain)

    # The three reference rows are what make the eight readable as worth
    # something: all of them are the one plateau row that runs over every arm,
    # so the horizon a signal is priced against is the horizon the patience
    # rule is priced against.
    every = tables.pick(tables.rows("plateau_stages.csv"), stage="all")
    for label, column in (("Fixed horizon", "fixed_mean"),
                          ("Patience rule", "rule_mean"),
                          ("Oracle stop", "oracle_mean")):
        printed = [ln for ln in lines if label in ln]
        assert len(printed) == 1, label
        assert "%.2f" % (100 * float(every[column])) in printed[0], label

    caption = [ln for ln in lines if ln.startswith("\\caption{")]
    assert len(caption) == 1, caption
    assert "\\textbf{validation}" in caption[0], caption[0]


def test_every_macro_the_template_carries_is_mapped_and_every_signal_quotable():
    """
    numbers.tex is rewritten in place from the macro NAMES the file already
    carries, so a macro added to one of the two files and not the other fails
    in two different directions: a name with no registry entry reaches the
    page as \\TBD{unmapped}, and a registry entry with no name is computed and
    thrown away. The template is the copy this repository ships to seed that
    rewrite, and neither failure is visible from the tables.

    The signals subsection is why this is pinned now. It quotes a correlation
    and a share for each of the eight signals in prose, beside a table that
    prints all eight, and a signal whose rho had no macro would be retyped by
    hand - which is the one thing numbers.tex exists to prevent.
    """
    import make_numbers

    text = (FIGURES / "numbers.template.tex").read_text()
    block = text.split(make_numbers.BEGIN, 1)[1].split(make_numbers.END, 1)[0]
    names = make_numbers.NAME_RE.findall(block)
    assert len(names) == len(set(names)), "a macro is defined twice"

    registry = make_numbers.build()
    assert sorted(set(names)) == sorted(registry)

    rho = [n for n in registry if n.startswith("nSignal") and n.endswith("Rho")]
    share = [n for n in registry
             if n.startswith("nSignal") and n.endswith("Share")]
    assert len(rho) == 8, sorted(rho)
    assert len(share) == 8, sorted(share)
