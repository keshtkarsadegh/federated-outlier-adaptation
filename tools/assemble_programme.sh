#!/bin/bash
# Assemble the complete programme of Digits_study01 into a fresh results root.
#
# Nothing is inherited: the run starts from by_write.zip and ends at the paper's
# tables. The task files of the stages that already have them are reused
# verbatim - they are committed and checksummed, and every path in them is
# $FOA_STUDY_DIR, so pointing that at a new root is the whole of the change.
#
# What this script adds, because the old tree had it only by hand:
#   * a producer for cohort_worst5.json. Nothing in jobs/v4 wrote it; the
#     five-client stage read it as though it appeared on its own, so a run from
#     zero would have stopped there.
#   * uniform --outer-workers 1 --inner-workers 1 on every line, so that
#     seconds-per-round finally measures the algorithm and not the launch.
#
# What it deliberately drops:
#   p04            the v1 selection, superseded by p05v2
#   p07            retrains g-0, which p05v2 already does; its evaluations are
#                  in p05v2 too
#   p08 p09 p10    the ten-client references, superseded by the per-size
#                  reference files which cover 5, 10, 20 and the extremes
#   p20_runs_f1    the fold-1 subset of p20_runs
set -euo pipefail

P=/user/keshtkar/u25763/.project/dir.lustre-grete/payman_uni
SRC=$P/jobs/v4
NEW=$P/results_v5/studies/Digits_study01
J=$NEW/jobs
mkdir -p "$J" "$NEW/logs" "$NEW/tables" "$NEW/outliers" "$NEW/fold_books"

norm() {  # one worker setting for every line in the programme
    sed -E 's/--outer-workers [0-9]+ --inner-workers [0-9]+/--outer-workers 1 --inner-workers 1/'
}

# ---------------------------------------------------------------- front end
norm < "$SRC/d01_p02.txt"    > "$J/s01_detector.txt"
norm < "$SRC/d01_p05v2.txt"  > "$J/s02_selection.txt"

# the cohorts the later points need, and the one nothing produced
{
  cat <<'HDR'
# s03_cohorts.txt - the five- and twenty-client cohorts.
#
# THE FIVE-CLIENT COHORT HAD NO PRODUCER. jobs/v4 contains no line that writes
# cohort_worst5.json, yet the five-client stage reads it; it had been made by
# hand. A programme that cannot rebuild its own inputs is not reproducible, so
# the line is here now, cut from the same ranking and by the same rule as the
# ten- and twenty-client cohorts.
#
# The five worst are a subset of the ten worst, so they share the ten-client
# fold book: same writers, same rows, same splits, and the two points stay
# directly comparable.
HDR
  echo 'foa select-outliers --results-dir $FOA_STUDY_DIR --resolution 28 --classes digits --mode worst --k 5 --scores $FOA_STUDY_DIR/outliers/bad_acc_on_g0.json --require-trainable --tag digits_cohort5 --force --no-accuracy-table --out $FOA_STUDY_DIR/outliers/cohort_worst5.json'
  grep '^foa' "$SRC/d01_p20_setup.txt" | norm
} > "$J/s03_cohorts.txt"

# ------------------------------------------------------------------ grids
norm < "$SRC/d01_p11.txt"                    > "$J/s05_agg_screen.txt"
norm < "$SRC/d01_p11_ext.txt"                > "$J/s06_agg_ext.txt"
norm < "$SRC/d01_p12.txt"                    > "$J/s07_agg_full.txt"
norm < "$SRC/d01_p12_trimmed_patch.txt"      > "$J/s08_agg_patch.txt"
norm < "$SRC/d01_p13.txt"                    > "$J/s09_reg_screen.txt"
norm < "$SRC/d01_p13_ext.txt"                > "$J/s10_reg_ext.txt"
norm < "$SRC/d01_p13_hybrid.txt"             > "$J/s11_reg_hybrid.txt"
norm < "$SRC/d01_p14.txt"                    > "$J/s12_reg_full.txt"
cat "$SRC"/d01_p14_*_patch.txt | grep '^foa' | norm > "$J/s13_reg_patch.txt"
norm < "$SRC/d01_p15.txt"                    > "$J/s14_combos.txt"
norm < "$SRC/d01_p16.txt"                    > "$J/s15_extremes.txt"

# ------------------------------------------------------- the other points
norm < "$SRC/d01_p18_five.txt"    > "$J/s16_five.txt"
norm < "$SRC/d01_p19_drop20.txt"  > "$J/s17_drop20.txt"
norm < "$SRC/d01_p20_runs.txt"    > "$J/s18_c20.txt"

echo "assembled into $J"
printf "%-28s %6s\n" "stage" "tasks"
tot=0
for f in "$J"/s*.txt; do
    n=$(grep -c '^foa' "$f")
    printf "%-28s %6s\n" "$(basename "$f")" "$n"
    tot=$((tot + n))
done
printf "%-28s %6s\n" "-- assembled subtotal" "$tot"
echo
echo "references are generated separately, once the cohorts exist."
