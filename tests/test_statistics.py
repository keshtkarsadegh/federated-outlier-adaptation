"""
The statistics the report layer decides with.

Every function is checked against a value that can be looked up or computed by
hand: the t quantiles of a printed table, the closed forms of the incomplete
beta function, an exactly enumerable Wilcoxon null distribution, and the
step-down arithmetic of the Holm correction.
"""

from __future__ import annotations

import math

import pytest

from federated_outlier_adaptation.analysis import statistics as stats


# --------------------------------------------------------------------------- #
# distributions
# --------------------------------------------------------------------------- #
def test_incomplete_beta_matches_its_closed_forms():
    # I_x(a, 1) = x^a, and the uniform case is the identity.
    assert stats.regularised_incomplete_beta(2.0, 1.0, 0.25) == pytest.approx(0.0625)
    assert stats.regularised_incomplete_beta(1.0, 1.0, 0.5) == pytest.approx(0.5)
    assert stats.regularised_incomplete_beta(3.0, 3.0, 0.5) == pytest.approx(0.5)
    assert stats.regularised_incomplete_beta(2.0, 5.0, 0.0) == 0.0
    assert stats.regularised_incomplete_beta(2.0, 5.0, 1.0) == 1.0


def test_the_t_distribution_is_symmetric_around_zero():
    for df in (1, 4, 30):
        assert stats.student_t_cdf(0.0, df) == pytest.approx(0.5)
        assert stats.student_t_cdf(-1.7, df) == pytest.approx(1.0 - stats.student_t_cdf(1.7, df))


@pytest.mark.parametrize(
    "df,expected",
    [(1, 12.706205), (2, 4.302653), (4, 2.776445), (9, 2.262157), (30, 2.042272)],
)
def test_t_quantiles_match_the_printed_table(df, expected):
    assert stats.student_t_quantile(0.975, df) == pytest.approx(expected, abs=1e-5)


def test_the_quantile_inverts_the_cdf():
    for df in (2, 7, 25):
        for p in (0.05, 0.5, 0.9, 0.99):
            assert stats.student_t_cdf(stats.student_t_quantile(p, df), df) == pytest.approx(p)


def test_the_quantile_rejects_impossible_probabilities():
    with pytest.raises(ValueError):
        stats.student_t_quantile(0.0, 5)
    with pytest.raises(ValueError):
        stats.student_t_cdf(1.0, 0)


def test_normal_cdf_at_the_usual_points():
    assert stats.normal_cdf(0.0) == pytest.approx(0.5)
    assert stats.normal_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def test_a_summary_is_mean_std_and_a_t_interval():
    values = [0.90, 0.92, 0.94, 0.96, 0.98]
    summary = stats.summarise(values)
    assert summary.n == 5
    assert summary.mean == pytest.approx(0.94)
    expected_std = math.sqrt(sum((v - 0.94) ** 2 for v in values) / 4)
    assert summary.std == pytest.approx(expected_std)
    assert summary.ci == pytest.approx(2.776445 * expected_std / math.sqrt(5), rel=1e-5)
    assert summary.low == pytest.approx(summary.mean - summary.ci)
    assert summary.high == pytest.approx(summary.mean + summary.ci)


def test_a_single_observation_has_no_interval():
    summary = stats.summarise([0.5])
    assert (summary.n, summary.mean) == (1, 0.5)
    assert summary.std is None and summary.ci is None
    assert summary.low is None and summary.high is None


def test_an_empty_sample_is_reported_as_empty():
    summary = stats.summarise([None, float("nan"), "x"])
    assert summary.n == 0 and summary.mean is None
    assert summary.as_dict()["mean"] is None


def test_booleans_are_not_numbers():
    assert stats.finite([True, False, 1.5]) == [1.5]


# --------------------------------------------------------------------------- #
# paired tests
# --------------------------------------------------------------------------- #
def test_the_paired_t_test_reproduces_a_hand_computed_example():
    a = [0.72, 0.75, 0.80, 0.78]
    b = [0.70, 0.74, 0.76, 0.75]
    result = stats.paired_t_test(a, b)
    differences = [x - y for x, y in zip(a, b)]
    mean = sum(differences) / 4
    std = math.sqrt(sum((d - mean) ** 2 for d in differences) / 3)
    assert result["n"] == 4
    assert result["difference"] == pytest.approx(mean)
    assert result["t"] == pytest.approx(mean / (std / 2.0))
    assert result["df"] == 3
    assert 0.0 < result["p"] < 0.05


def test_a_constant_difference_is_certain_and_no_difference_is_not():
    assert stats.paired_t_test([1.0, 2.0], [0.0, 1.0])["p"] == 0.0
    assert stats.paired_t_test([1.0, 2.0], [1.0, 2.0])["p"] == 1.0


def test_too_few_pairs_report_no_p_value():
    assert stats.paired_t_test([1.0], [0.5])["p"] is None
    assert stats.paired_t_test([], [])["n"] == 0


def test_wilcoxon_is_exact_on_the_smallest_decisive_sample():
    # Six strictly positive differences: only one of the 64 sign assignments is
    # at least as extreme, on either side.
    result = stats.wilcoxon_signed_rank([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0])
    assert result["method"] == "exact"
    assert result["statistic"] == 0
    assert result["p"] == pytest.approx(2.0 / 64.0)


def test_wilcoxon_counts_both_tails():
    result = stats.wilcoxon_signed_rank([1, 2, 3, -4, 5], [0, 0, 0, 0, 0])
    # Ranks 1..5, the negative one carries rank 4, so the statistic is 4 and
    # the subsets of {1..5} summing to at most 4 are 0,1,2,3,4,1+2,1+3 = 7.
    assert result["statistic"] == 4
    assert result["p"] == pytest.approx(2.0 * 7.0 / 32.0)


def test_wilcoxon_drops_zero_differences():
    result = stats.wilcoxon_signed_rank([1, 1, 2], [1, 1, 0])
    assert result["n"] == 1


def test_wilcoxon_falls_back_to_the_normal_approximation_with_ties():
    result = stats.wilcoxon_signed_rank([1, 1, 2, 3], [0, 0, 0, 0])
    assert result["method"] == "normal"
    assert 0.0 < result["p"] <= 1.0


def test_wilcoxon_without_any_pair():
    assert stats.wilcoxon_signed_rank([], [])["p"] is None


def test_average_ranks_share_the_tie():
    assert stats.average_ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_cohens_d_is_the_mean_difference_over_its_own_spread():
    a, b = [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]
    differences = [x - y for x, y in zip(a, b)]
    mean = sum(differences) / 4
    std = math.sqrt(sum((d - mean) ** 2 for d in differences) / 3)
    assert stats.cohens_d_paired(a, b) == pytest.approx(mean / std)
    assert stats.cohens_d_paired([1.0], [0.0]) is None
    assert stats.cohens_d_paired([1.0, 1.0], [0.0, 0.0]) is None


# --------------------------------------------------------------------------- #
# multiplicity
# --------------------------------------------------------------------------- #
def test_holm_steps_down_and_stays_monotone():
    assert stats.holm([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    # The order of the input does not matter, only the ranks.
    assert stats.holm([0.03, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_holm_never_exceeds_one():
    assert stats.holm([0.4, 0.6, 0.9]) == pytest.approx([1.0, 1.0, 1.0])


def test_holm_ignores_the_comparisons_that_were_not_made():
    adjusted = stats.holm([0.01, None, 0.02])
    assert adjusted[1] is None
    # Only two tests take part, so the tightest is doubled, not tripled.
    assert adjusted[0] == pytest.approx(0.02)
    assert adjusted[2] == pytest.approx(0.02)
    assert stats.holm([None, None]) == [None, None]


@pytest.mark.parametrize(
    "p,expected",
    [(0.0005, "***"), (0.005, "**"), (0.04, "*"), (0.2, ""), (None, "")],
)
def test_significance_markers(p, expected):
    assert stats.significance_marker(p) == expected
