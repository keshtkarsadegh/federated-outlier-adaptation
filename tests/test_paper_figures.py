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

import ast
import csv
import importlib
import re
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
        # The table generator names ten of its views by interpolation now -
        # the four carry settings and the five fairness blocks - so the literal
        # count is lower there than in the macro generator. The interpolated
        # names are checked by the test below, which reads the block lists.
        assert len(wanted) >= 14, f"{module.__name__}: the pattern stopped matching"
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

    # THE FAIRNESS TABLE NAMES ITS OWN BLOCKS NOW, one per setting an arm was
    # carried to plus the search setting it was chosen at. A block whose view
    # stopped shipping is a block that vanishes from the table with no error,
    # which is exactly how the five-client block came to be missing from it.
    keys = [key for key, _, _ in make_paper_tables.FAIRNESS_BLOCKS]
    assert sorted(keys) == ["c10d10", "c20d10", "c20d20", "combo", "five"], keys
    assert set(keys) - {"combo"} == set(tags), (
        "tab:fairness and tab:scaling no longer report the same settings")
    for key in keys:
        make_paper_tables.locate(f"fairness_{key}.csv")


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


#: Every byte a generated caption, table or macro file may not carry. Newline
#: is the one control character any of them has a use for; the rest arrive by
#: accident, and they arrive invisibly - a terminal swallows them, a diff shows
#: nothing, and LaTeX sets whatever is left.
CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


def _controls(text):
    """The control characters in one string, named, for a readable failure."""
    return sorted({"0x%02x" % ord(character) for character in CONTROL.findall(text)})


def _generated_captions(out, monkeypatch):
    """Every caption the figure scripts write, into a scratch directory.

    The scripts are run rather than read: a caption is a Python string on the
    way to a file, and the two spellings of one that matter here differ only
    after the interpreter has had it.
    """
    import matplotlib.pyplot as plt

    import figstyle

    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(figstyle, "OUT", str(out))
    for path in sorted(FIGURES.glob("fig_*.py")):
        importlib.import_module(path.stem).main()
        plt.close("all")
    return sorted(out.glob("*_caption.txt"))


def test_no_generated_caption_table_or_macro_carries_a_control_character(
        tmp_path, monkeypatch):
    """
    Nothing the manuscript inputs may carry a byte a reader cannot see.

    `fig_combo`'s caption sets $\beta$ and $\tau$, and it was written as a
    plain Python string: the interpreter turned the two backslash sequences
    into a backspace and a tab before `figstyle.caption` ever saw them, the
    whitespace-normalising join ate the tab, and the caption file shipped a
    literal 0x08 in front of "eta". It rendered as "$eta=0.01$, $ au=0.5$" and
    looked, in every terminal and every diff, like a caption.

    So the check is on the bytes of everything the generators write, and not on
    the source that wrote it: a raw string is one fix for one file, and this is
    the property the manuscript actually needs.
    """
    for path in _generated_captions(tmp_path / "figures", monkeypatch):
        found = _controls(path.read_text())
        assert not found, (path.name, found)

    for path in sorted(_generated_tables(tmp_path / "tex", monkeypatch).glob("*.tex")):
        found = _controls(path.read_text())
        assert not found, (path.name, found)

    import make_numbers

    monkeypatch.setenv("FOA_PAPER_OUT", str(tmp_path / "tex"))
    monkeypatch.setattr(sys, "argv", ["make_numbers.py"])
    assert make_numbers.main() == 0
    numbers = tmp_path / "tex" / "numbers.tex"
    found = _controls(numbers.read_text())
    assert not found, found


def test_no_string_in_a_figure_script_smuggles_a_python_escape():
    """
    The same failure, caught where it is written rather than where it lands.

    A caption that never reaches a file - one behind a flag, one added today
    and rendered tomorrow - is not covered by the test above until somebody
    runs it. Every string constant in these modules is read out of the syntax
    tree instead, so a backslash that Python understands and LaTeX meant
    differently fails at once.
    """
    for path in sorted(FIGURES.glob("fig_*.py")) + [FIGURES / "figstyle.py"]:
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            found = _controls(node.value)
            assert not found, (path.name, node.lineno, found)


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


#: A printed name that stands for a mixed penalty.  Matched case-insensitively
#: because the tables set the word both ways.
MIXED_NAME = re.compile(r"composite|blend", re.IGNORECASE)

#: How a printed mix reads.  Not "$m{=}": a full setting sets one formula for
#: all three coefficients, so the mix arrives as the tail of
#: "$\lambda{=}0.0101,T{=}2,m{=}0.5$" as often as it arrives on its own.
MIX = "m{=}"


def _row_cells(text):
    """Every column entry of every data row of one emitted table.

    The header sits between the top rule and the first midrule and the note
    after the bottom rule, so reading between the first midrule and the bottom
    rule sees the data rows and nothing else. The block bands and the family
    headings are multicolumn lines and are dropped: a heading names a family
    of arms, which has no one mix to carry, and a caption that defines the
    composite is prose about the idea rather than a claim about a row.
    """
    body = text.split("\\midrule", 1)[-1].split("\\bottomrule")[0]
    for line in body.splitlines():
        line = line.strip()
        if not line.endswith("\\\\") or "\\multicolumn" in line:
            continue
        for cell in line[:-2].split("&"):
            yield cell.replace("\\quad", "").strip()


def test_every_composite_a_table_names_carries_its_mix(tmp_path, monkeypatch):
    """
    "KD+EWC composite, cyclic-tuned" is the name of three arms, not one.

    The construction stage built one composite per schedule and swept three
    mixes of each, and the mix is the only dial it turned - the lambda and the
    T are inherited from the halves. So a row that prints the parentage and
    stops has named a family and been read as an arm: tab:combos crosses the
    cyclic-tuned composite at m = 0.5 and tab:reg_winners leads with the same
    parentage at m = 0.75, and nothing on the combination row said which.

    The rule is therefore on the emitted LaTeX and on every table at once: a
    data cell that names a mixed penalty either carries its mix or is a
    carried arm's short name that the table's own note expands, with the
    parentage and the mix in the sentence that expands it. Anything else is a
    row the reader cannot resolve without leaving the page.
    """
    out = _generated_tables(tmp_path, monkeypatch)
    named = 0
    for path in sorted(out.glob("*.tex")):
        text = path.read_text()
        note = text.split("\\bottomrule")[-1]
        for cell in _row_cells(text):
            if not MIXED_NAME.search(cell):
                continue
            named += 1
            if MIX in cell:
                continue
            windows = [note[i:i + 400] for i in range(len(note))
                       if note.startswith(cell, i)]
            assert windows, (
                f"{path.name} prints {cell!r}, which names a mixed penalty "
                f"without its mix, and its note never expands the name")
            assert any(MIX in w and "-tuned" in w for w in windows), (
                f"{path.name} prints {cell!r} without its mix, and the note "
                f"expands it without saying which composite it is and at "
                f"what mix")
    assert named >= 20, ("the tables have stopped naming composites, or the "
                         "pattern has stopped matching: %d" % named)

    # The two tables the composite rows are argued from say it on the row
    # itself: both cross a composite three times, and both print rows that
    # differ in nothing but which composite they crossed.
    for name in ("combos.tex", "selection_axis_cross.tex"):
        cells = [c for c in _row_cells((out / name).read_text())
                 if MIXED_NAME.search(c)]
        assert len(cells) == 6, (name, cells)
        for cell in cells:
            assert MIX in cell, (name, cell)


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

        assert cells[1] == row["observed_on"]
        assert cells[2] == "%.3f" % float(row["median_rho"])
        assert cells[3] == "%.1f" % (100 * float(row["share_ge_0p9"])) + r"\%"
        assert cells[4] == row["best_delta"]
        assert cells[5] == "%d" % int(row["arms_fired"])
        assert cells[6] == "%.2f" % (100 * float(row["mean_stopped_score"]))

        # THE DELTA COLUMN CLOSES ON THE PAGE. Both of its operands are printed
        # in this table - the score of the row, and the fixed-horizon score at
        # the foot of it - so the column is the difference of the two printed
        # values and not the stored difference rounded once. Rounded once,
        # three of these eight rows came out a hundredth away from the
        # subtraction a reader does: 1.46 less 7.74 printed as -6.27. The
        # stored value is still checked, to within the hundredth that
        # separates the two conventions, so a real drift still fails here.
        fixed = "%.2f" % (100 * float(tables.pick(
            tables.rows("plateau_stages.csv"), stage="all")["fixed_mean"]))
        delta = "%.2f" % (float(cells[6]) - float(fixed))
        assert cells[7] == ("$%s$" % delta if delta.startswith("-")
                            else "$+%s$" % delta), row["signal"]
        assert abs(float(delta) - 100 * float(row["mean_vs_fixed"])) <= 0.01 + 1e-9

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

    # The same identity for the two reference rows that carry a delta: the
    # whole column subtracts the fixed horizon as printed, references included.
    fixed = 100 * float(every["fixed_mean"])
    for label, column in (("Patience rule", "rule_mean"),
                          ("Oracle stop", "oracle_mean")):
        row = [ln for ln in lines if label in ln][0]
        cells = [c.strip() for c in row.rstrip("\\ ").split(" & ")]
        delta = "%.2f" % (float(cells[6]) - float("%.2f" % fixed))
        assert cells[7] == ("$%s$" % delta if delta.startswith("-")
                            else "$+%s$" % delta), (label, cells)

    caption = [ln for ln in lines if ln.startswith("\\caption{")]
    assert len(caption) == 1, caption
    assert "\\textbf{validation}" in caption[0], caption[0]


def test_the_joint_tuning_macros_are_the_cells_the_extension_table_reports():
    """
    Section 4.7 names the configuration the joint search chose, and no table
    prints it: tab:jointtune reports the arms' scores and the dials live only
    in the screen's ranked view. So the eight macros are read out of one file
    and the arm they describe out of another, and a re-run that moved the
    winner in either would leave the sentence describing a cell the table does
    not report. Both halves are pinned here, in the grid's own spelling.
    """
    import make_numbers

    registry = make_numbers.build()
    with (PAPER / "extension_combo_screen_selected.csv").open(newline="") as fh:
        screen = list(csv.DictReader(fh))
    with (PAPER / "extension_combo_tune_selected.csv").open(newline="") as fh:
        reported = list(csv.DictReader(fh))

    for fam, tag in (("concurrent", "Par"), ("sequential", "Cyc")):
        won = [r for r in screen if r["family"] == fam and r["rank"] == "1"]
        assert len(won) == 1, fam
        arms = [r["arm"] for r in reported
                if r["family"] == fam and r["role"] == "tuned pair"]
        assert arms == [won[0]["cell"]], (fam, arms)
        for col, name in (("c_ewc", "Ewc"), ("c_kd", "Kd"),
                          ("T", "T"), ("m", "Mix")):
            value, src = registry[f"nJointSel{name}{tag}"]
            assert float(value) == float(won[0][col]), (name, tag, value)
            assert "extension_combo_screen_selected.csv" in src

    # The sentence these eight serve says the two schedules agree on the mix
    # and on nothing else; the agreement is the half of that a macro can hold.
    assert registry["nJointSelMixPar"][0] == registry["nJointSelMixCyc"][0]


def test_the_selection_axis_counts_are_over_the_rows_that_table_prints():
    """
    "Most arms move between the two axes" is a claim about the page, and a
    count taken over selection_axis.csv would be a claim about sixty rows the
    table does not print.  So the two macros are rebuilt here the way the
    table builds its blocks, and the arms that do NOT move are named: they
    are few enough that a drift in any one of them would otherwise read as a
    plausible new total.
    """
    import make_numbers
    import make_paper_tables as tables

    registry = make_numbers.build()
    with (PAPER / "selection_axis.csv").open(newline="") as fh:
        view = list(csv.DictReader(fh))

    printed = []
    for stage, _, _ in tables.SELECTION_STAGES:
        for family, _ in tables.SCHEDULE_BLOCKS:
            block = sorted([r for r in view if r["stage"] == stage
                            and r["schedule"] == family],
                           key=lambda r: int(r["val_score_rank"]))
            # The cross prints all nine; a selecting block prints the depth
            # plus every arm a mark is owed to below it.
            depth = (len(block) if stage in tables.SELECTION_FULL
                     else tables.SELECTION_DEPTH)
            printed += block[:depth]
            printed += [r for r in block[depth:]
                        if r["role"]
                        or (r["cell"], family) in tables.SELECTION_CARRIED
                        or (r["cell"], family) in tables.SELECTION_SHOWN]

    assert registry["nSelectionRankTotal"][0] == str(len(printed))
    # Two selecting stages at the depth, plus the one arm a mark is owed to
    # below it -- the test-leading composite on the parallel penalties, which
    # no validation ordering selected -- plus both cross blocks entire. The
    # balanced arm of tab:scaling is inside that last term now: validation put
    # it ninth of nine and the cross block prints all nine.
    selecting = [s for s, _, _ in tables.SELECTION_STAGES
                 if s not in tables.SELECTION_FULL]
    crossings = len([r for r in view if r["stage"] in tables.SELECTION_FULL])
    assert int(registry["nSelectionRankTotal"][0]) == (
        len(selecting) * len(tables.SCHEDULE_BLOCKS) * tables.SELECTION_DEPTH
        + 1 + crossings)
    still = {(r["cell"], r["schedule"]) for r in printed
             if r["val_score_rank"] == r["test_score_rank"]}
    # Five, and it was three while the cross printed five crossings of nine:
    # the two new ones sit at the foot of their cross blocks, where the two
    # orderings agree because there is nothing below them to disagree about.
    assert still == {("fedavgm_b0p3", "parallel"),
                     ("seq_delta_capped", "cyclic"),
                     ("hybrid_seq_mix0p5", "parallel"),
                     ("weight_q0_hybrid_seq_mix0p5", "parallel"),
                     ("seq_fedavg_hybrid_mix0p5", "cyclic")}, sorted(still)
    assert registry["nSelectionRankMoved"][0] == str(len(printed) - len(still))
    for name in ("nSelectionRankTotal", "nSelectionRankMoved"):
        assert "selection_axis.csv" in registry[name][1]


def test_the_screened_composite_gaps_are_against_the_rows_the_table_prints():
    """
    Three macros price the coefficient screen against the construction sweep,
    and each is a difference of two rows of one table -- so each is rounded
    once, from full precision, and a reader who subtracts the PRINTED scores
    can land a digit away from it.  The arms are pinned here because two of
    the three are easy to confuse: the row that leads tab:reg_winners is the
    cyclic-tuned composite at m = 0.75, which no selection chose, and the row
    the cyclic cross carried is the parallel-tuned one at m = 0.5.
    """
    import make_numbers
    import make_paper_tables as tables

    registry = make_numbers.build()
    with (PAPER / "reg-winners.csv").open(newline="") as fh:
        regu = list(csv.DictReader(fh))

    def score(cell, family):
        hit = [r for r in regu if r["cell"] == cell and r["family"] == family]
        assert len(hit) == 1, (cell, family)
        return float(hit[0]["score"])

    screened = {}
    for family in ("parallel", "cyclic"):
        hit = [r["cell"] for r in regu
               if r["cell"].startswith("blend_") and r["family"] == family]
        assert len(hit) == 1, (family, hit)
        screened[family] = hit[0]

    assert tables.BLEND_TEST_LEADING == "hybrid_seq_mix0p75"
    for macro, family in (("nBlendScreenGapPar", "parallel"),
                          ("nBlendScreenGapCyc", "cyclic")):
        gap = (score(tables.BLEND_TEST_LEADING, family)
               - score(screened[family], family))
        assert registry[macro][0] == "%.2f" % (100.0 * gap), macro
        assert float(registry[macro][0]) > 0, macro

    taken = ("hybrid_mix0p5", "cyclic")
    gap = score(screened["cyclic"], "cyclic") - score(*taken)
    assert registry["nBlendScreenVsSelectedCyc"][0] == "%.2f" % (100.0 * gap)
    assert taken[0] in registry["nBlendScreenVsSelectedCyc"][1]


#: Which reporting view each selecting stage's test columns are joined onto,
#: and the column names that view calls them by.  `t_selection_axis` prints the
#: REPORTED row rather than the view's own reading of the same runs, so this is
#: the join the table actually makes.
SELECTION_REPORTS = {"aggregation": "agg-winners.csv",
                     "regularisation": "reg-winners.csv",
                     "combination": "combos.csv"}

#: The one pair of DISTINCT arms whose three test numbers round to the same
#: figures the tables print, named here so a coincidence cannot be mistaken for
#: a join and a join cannot hide inside the rounding.  The parallel reading of
#: the cyclic-tuned composite at m = 0.25 comes from
#: d01_regfull_hybrid_seq_mix0p25_fold{1..5}/concurrent_delta and the screened
#: composite on the cyclic schedule from
#: d01_regfull_sequential_blend_lam0p1_T0p25_mix0p5_fold{1..5}/sequential_weights:
#: two stems, ten folders, no folder and no per-fold value in common.  Their
#: agreement stops at the fourth digit, which is what the test asserts; runs
#: read under the wrong arm's name would agree at every digit instead.
ROUNDED_TWINS = frozenset({(("blend_lam0p1_T0p25_mix0p5", "cyclic"),
                            ("hybrid_seq_mix0p25", "parallel"))})


def test_every_selection_axis_arm_joins_one_run_set_and_no_arm_appears_twice():
    """
    The selection-axis view joins on (cell, family), and a join on a key two
    arms can share reports one arm's runs under the other's name --- silently,
    because both columns still look like accuracies.  Two things are checked
    here.  Every key of the view is unique and lands on exactly one reporting
    row whose numbers are the view's own to the precision the view stores them
    at, over one set of five folds.  And no two DISTINCT arms of the shipped
    views carry all three test numbers: exactly is a failure, and to the four
    figures the tables print it is allowed only for the one pair the records
    really do put that close, which is named above.
    """
    import make_paper_tables as tables

    def read(name):
        with (PAPER / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    view = read("selection_axis.csv")
    keys = [(r["stage"], r["cell"], r["schedule"]) for r in view]
    assert len(set(keys)) == len(keys), "a (stage, cell, schedule) twice"

    stages = {stage for stage, _, _ in tables.SELECTION_STAGES}
    assert set(SELECTION_REPORTS) == stages, sorted(stages)

    for row in view:
        reported = [r for r in read(SELECTION_REPORTS[row["stage"]])
                    if r["cell"] == row["cell"]
                    and r["family"] == row["schedule"]]
        assert len(reported) == 1, (row["stage"], row["cell"], row["schedule"],
                                    len(reported))
        got = reported[0]
        assert got["folds"] == "5", (row["cell"], got["folds"])
        for mine, theirs in (("test_adaptation", "adaptation"),
                             ("test_preservation", "preservation"),
                             ("test_score", "score")):
            assert abs(float(row[mine]) - float(got[theirs])) < 5e-7, (
                row["stage"], row["cell"], row["schedule"], mine)

    arms = [("selection_axis/" + r["stage"], r["cell"], r["schedule"],
             float(r["test_adaptation"]), float(r["test_preservation"]),
             float(r["test_score"])) for r in view]
    for name in sorted(set(SELECTION_REPORTS.values())):
        arms += [(name, r["cell"], r["family"], float(r["adaptation"]),
                  float(r["preservation"]), float(r["score"]))
                 for r in read(name)]

    def collisions(round_it):
        seen = {}
        for source, cell, family, a, p, s in arms:
            key = (("%.4f" % a, "%.4f" % p, "%.2f" % (100 * s)) if round_it
                   else (a, p, s))
            seen.setdefault(key, set()).add((cell, family))
        return {tuple(sorted(v)) for v in seen.values() if len(v) > 1}

    assert not collisions(False), (
        "two arms with identical test numbers: %s" % sorted(collisions(False)))
    assert collisions(True) == ROUNDED_TWINS, sorted(collisions(True))

    for pair in ROUNDED_TWINS:
        numbers = [next((a, p, s) for _, cell, family, a, p, s in arms
                        if (cell, family) == one) for one in pair]
        assert numbers[0] != numbers[1], pair


def test_the_carried_fairness_range_leaves_the_controls_out():
    """
    Two macros price what the carrying bought the worst-served client, and the
    two beside them price the same block with its controls in.  The pair is
    easy to swap and the swap is invisible --- both are a range of the same
    column of the same five rows --- so the arms are named here rather than
    only the values: plain FedAvg lifts its worst client least of the five, so
    a carried range that had picked it up would read low at the bottom and
    right at the top.
    """
    import make_numbers

    registry = make_numbers.build()
    with (PAPER / "fairness_c10d10.csv").open(newline="") as handle:
        fair = list(csv.DictReader(handle))

    carried = [r for r in fair if r["cell"] != "control"]
    assert sorted(r["cell"] for r in carried) == ["balanced", "sequential",
                                                  "winner"]
    assert sorted(r["cell"] for r in fair if r["cell"] == "control") == \
        ["control", "control"]

    lifts = [float(r["lift_worst"]) for r in carried]
    assert registry["nFairWorstMinGainCarried"][0] == "%.2f" % (100 * min(lifts))
    assert registry["nFairWorstMaxGainCarried"][0] == "%.2f" % (100 * max(lifts))
    assert registry["nFairWorstMinGainCarried"][0] == "11.48"
    assert registry["nFairWorstMaxGainCarried"][0] == "12.17"

    # the controls are outside it, and the wider pair is not
    over_all = [float(r["lift_worst"]) for r in fair]
    assert min(over_all) < min(lifts), "a control lifts more than a carried arm"
    assert registry["nFairWorstMinGain"][0] == "%.2f" % (100 * min(over_all))
    assert registry["nFairWorstMaxGain"][0] == "%.2f" % (100 * max(over_all))
    for name in ("nFairWorstMinGainCarried", "nFairWorstMaxGainCarried"):
        assert "non-control" in registry[name][1], name
        assert "fairness_c10d10.csv" in registry[name][1], name


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


def test_both_selection_axis_rank_columns_are_over_one_population():
    """
    The block header states a denominator and both Rank columns use it.

    ``export_selection_axis`` ranks one population twice, once per axis, so
    the test rank is over the arms the record ranked and NOT over the rows
    the reporting table prints -- which on the penalties is a larger set,
    because the screened composite joined the finals after the screen had
    ranked these.  A reader checking a test rank against the finals table
    lands a place out for every arm below it, so both counts are stated on
    the page and both are pinned here.
    """
    import make_paper_tables as tables

    with (PAPER / "selection_axis.csv").open(newline="") as fh:
        view = list(csv.DictReader(fh))
    with (PAPER / "reg-winners.csv").open(newline="") as fh:
        regu = list(csv.DictReader(fh))

    for stage in {r["stage"] for r in view}:
        for family, _ in tables.SCHEDULE_BLOCKS:
            block = [r for r in view if r["stage"] == stage
                     and r["schedule"] == family]
            ranked = {int(r["ranked"]) for r in block}
            assert ranked == {len(block)}, (stage, family, ranked)
            for column in ("val_score_rank", "test_score_rank"):
                places = sorted(int(r[column]) for r in block)
                assert places == list(range(1, len(block) + 1)), (stage,
                                                                  family,
                                                                  column)

    singles, mixes, finals = tables.penalty_population(regu)
    penalties = [r for r in view if r["stage"] == "regularisation"]
    assert {int(r["ranked"]) for r in penalties} == {singles + mixes}
    assert finals == singles + mixes + 1, (singles, mixes, finals)
    # The screened composite is the one final outside the ranking, and the
    # counts the two notes state are these: 13 ranked, 14 finals, 28 in all.
    screened = [r["cell"] for r in regu if r["cell"].startswith("blend_")]
    assert len(screened) == len(tables.SCHEDULE_BLOCKS), screened
    assert not [r for r in penalties if r["cell"].startswith("blend_")]
    assert len(regu) == len(tables.SCHEDULE_BLOCKS) * finals, len(regu)


def test_the_notes_that_name_a_baseline_name_it_by_macro():
    """
    Three table notes state which cohort's shipped model a block is scored
    against, and a retyped 0.8225 beside a regenerated view is exactly the
    drift numbers.tex exists to stop.  So the notes carry the macro and the
    macro carries the provenance; this is what says they still do.
    """
    import make_paper_tables as tables

    sizes = {key: tables.rows("sizes_%s.csv" % key)
             for key in ("c10d10", "five", "c20d10", "c20d20")}
    scaling = tables.t_scaling(sizes, tables.rows("cost_stages.csv"))
    for macro in ("nGZeroCohortAcc", "nGZeroCohortAccFive",
                  "nGZeroCohortAccTwenty"):
        assert "\\%s{}" % macro in scaling, macro

    plateau = tables.t_plateau(tables.rows("plateau_stages.csv"),
                               tables.rows("plateau_holdout_protocols.csv"))
    for macro in ("nGZeroCohortAcc", "nGZeroCohortAccExtreme"):
        assert "\\%s{}" % macro in plateau, macro
    # A printed accuracy beside the macro would be the retyped copy.
    for text in (scaling, plateau):
        assert "0.8225" not in text and "0.7590" not in text


def test_every_plateau_stage_row_names_the_baseline_its_level_is_read_against():
    """
    tab:plateau stacks four cohorts and its note used to name two of them.

    A reader who takes the rows the note leaves out for the baseline it does
    name reads the five-client and twenty-client levels against a population
    those arms never saw -- silently, because the levels are points either
    way.  So the partition is checked from both ends: every stage the table
    prints has a baseline, that baseline is the one report_tables assigns the
    stage's own run folders, and the note names all four of them.
    """
    import make_paper_tables as tables
    import report_tables

    macro_of = {"c10": "nGZeroCohortAcc", "c5": "nGZeroCohortAccFive",
                "c20": "nGZeroCohortAccTwenty",
                "extreme": "nGZeroCohortAccExtreme"}
    stages = [stage for stage, _ in tables.STAGE_LABEL]
    printed = {row["stage"] for row in tables.rows("plateau_stages.csv")}
    assert set(stages) <= printed, sorted(set(stages) - printed)
    assert set(stages) == set(tables.STAGE_BASELINE), sorted(
        set(stages) ^ set(tables.STAGE_BASELINE))

    plateau = tables.t_plateau(tables.rows("plateau_stages.csv"),
                               tables.rows("plateau_holdout_protocols.csv"))
    for stage in stages:
        macro = tables.STAGE_BASELINE[stage]
        assert macro == macro_of[report_tables.cohort_of(stage + "_")], stage
        assert "\\%s{}" % macro in plateau, stage
    assert set(tables.STAGE_BASELINE.values()) == set(macro_of.values())


def test_the_extreme_scores_close_from_the_accuracy_pairs_beside_them():
    """
    Section 7 quotes each extreme arrangement's score in points beside the two
    accuracies it is made of, and a reader closes the one against the other:
    ``(adaptation - A0) - (P0 - preservation)``, with the extremes' own
    shipped-model accuracy as A0.  A pair read off a neighbouring round, or
    off the test table rather than the per-round trace the score came from,
    still looks like a pair -- it just does not close.  Four printed digits
    per accuracy put the recomputation within a hundredth of a point, which is
    the tolerance here and the reason the pairs are quoted to four.
    """
    import make_numbers

    registry = make_numbers.build()
    a0 = float(registry["nGZeroCohortAccExtreme"][0])
    source = float(registry["nGZeroSrcAcc"][0])
    quoted = (("dual", "nDualOracleScore", "nDualOracleAdapt", "nDualOraclePres"),
              ("double", "nDoubleOracleScore", "nDoubleOracleAdapt",
               "nDoubleOraclePres"),
              ("dual", "nDualFinalScore", "nDualFinalAdapt", "nDualFinalPres"),
              ("double", "nDoubleFinalScore", "nDoubleFinalAdapt",
               "nDoubleFinalPres"))
    for arm, score, adapt, pres in quoted:
        closed = 100 * ((float(registry[adapt][0]) - a0)
                        - (source - float(registry[pres][0])))
        assert round(abs(closed - float(registry[score][0])), 6) <= 0.01, score
        for macro in (score, adapt, pres):
            assert ", row %s" % arm in registry[macro][1], (macro, arm)
