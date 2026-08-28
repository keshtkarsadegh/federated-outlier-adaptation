"""
Small-sample statistics of the report layer.

The study repeats every configuration over a handful of seeds, so every number
it reports is a mean over three to five observations.  At that size the
interesting quantities are the ones that say how much the mean can be trusted -
a confidence interval, a paired test against the baseline of the same setting,
an effect size - and the multiplicity correction that keeps a table of forty
such tests honest.

Everything here is implemented in the standard library on purpose.  The package
already depends on a scientific stack, but a statistic that decides what a paper
claims is worth having in a form that can be read, and the closed forms below
are short:

    - Student's t distribution through the regularised incomplete beta function
      (continued fraction, Lentz's method), which gives both the CDF used by the
      t-test and, by bisection, the quantile used by the confidence interval;
    - the Wilcoxon signed-rank test, exact for the sample sizes this study
      actually produces and with the tie-corrected normal approximation beyond
      them;
    - Cohen's ``d_z`` for paired samples;
    - the Holm step-down correction, applied within each table.

All functions ignore non-finite entries and return ``None`` rather than raising
when a sample is too small, so a partially populated results tree produces a
table with gaps instead of a traceback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

#: Confidence level of every reported interval.
DEFAULT_LEVEL = 0.95

#: Sample size up to which the Wilcoxon null distribution is enumerated exactly.
EXACT_WILCOXON_MAX_N = 25

#: Thresholds of the significance markers, tightest first.
SIGNIFICANCE_LEVELS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


def finite(values: Iterable) -> list[float]:
    """The finite numbers of ``values``, in order; booleans are not numbers."""
    out: list[float] = []
    for value in values or []:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            continue
        out.append(float(value))
    return out


def mean(values: Iterable) -> Optional[float]:
    """Arithmetic mean of the finite entries, or ``None`` when there are none."""
    data = finite(values)
    return sum(data) / len(data) if data else None


def sample_std(values: Iterable) -> Optional[float]:
    """Sample standard deviation (``n - 1``), or ``None`` below two points."""
    data = finite(values)
    if len(data) < 2:
        return None
    average = sum(data) / len(data)
    variance = sum((value - average) ** 2 for value in data) / (len(data) - 1)
    return math.sqrt(variance)


# --------------------------------------------------------------------------- #
# Student's t
# --------------------------------------------------------------------------- #
def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz evaluation of the continued fraction of the incomplete beta."""
    tiny = 1e-30
    epsilon = 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, 300):
        m2 = 2 * m
        for numerator in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + numerator * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + numerator / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)``, the regularised incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    mirrored = math.exp(b * math.log1p(-x) + a * math.log(x) - _log_beta(b, a))
    return 1.0 - mirrored * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """Cumulative distribution function of Student's t with ``df`` degrees."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if not math.isfinite(t):
        return 0.0 if t < 0 else 1.0
    x = df / (df + t * t)
    tail = 0.5 * regularised_incomplete_beta(df / 2.0, 0.5, x)
    return tail if t <= 0 else 1.0 - tail


def student_t_quantile(p: float, df: float) -> float:
    """Inverse CDF of Student's t, found by bisection on the CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    low, high = -1e6, 1e6
    for _ in range(200):
        middle = 0.5 * (low + high)
        if student_t_cdf(middle, df) < p:
            low = middle
        else:
            high = middle
        if high - low < 1e-12 * max(1.0, abs(low)):
            break
    return 0.5 * (low + high)


def normal_cdf(z: float) -> float:
    """Standard normal CDF, used by the Wilcoxon approximation."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Summary:
    """Mean of a metric over the seeds of one configuration."""

    n: int
    mean: Optional[float] = None
    std: Optional[float] = None
    sem: Optional[float] = None
    ci: Optional[float] = None
    level: float = DEFAULT_LEVEL

    @property
    def low(self) -> Optional[float]:
        if self.mean is None or self.ci is None:
            return None
        return self.mean - self.ci

    @property
    def high(self) -> Optional[float]:
        if self.mean is None or self.ci is None:
            return None
        return self.mean + self.ci

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "sem": self.sem,
            "ci": self.ci,
            "low": self.low,
            "high": self.high,
        }


def summarise(values: Iterable, level: float = DEFAULT_LEVEL) -> Summary:
    """
    Mean, standard deviation and half-width of the ``level`` interval.

    A single observation has a mean and no spread, which is reported as
    ``n = 1`` with ``std``/``ci`` unset rather than as a fake zero interval.
    """
    data = finite(values)
    if not data:
        return Summary(n=0, level=level)
    average = sum(data) / len(data)
    if len(data) == 1:
        return Summary(n=1, mean=average, level=level)
    std = sample_std(data)
    sem = std / math.sqrt(len(data))
    half_width = student_t_quantile(0.5 + level / 2.0, len(data) - 1) * sem
    return Summary(n=len(data), mean=average, std=std, sem=sem, ci=half_width, level=level)


# --------------------------------------------------------------------------- #
# paired comparisons
# --------------------------------------------------------------------------- #
def _paired(a: Sequence, b: Sequence) -> list[float]:
    """Finite differences of index-aligned pairs."""
    differences: list[float] = []
    for left, right in zip(a, b):
        if isinstance(left, bool) or isinstance(right, bool):
            continue
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            continue
        if not math.isfinite(left) or not math.isfinite(right):
            continue
        differences.append(float(left) - float(right))
    return differences


def paired_t_test(a: Sequence, b: Sequence) -> dict:
    """
    Two-sided paired t-test of ``a - b``.

    Returns:
        dict: ``{"n", "difference", "t", "df", "p"}``.  ``p`` is ``None`` when
        fewer than two pairs survive; a sample with zero spread reports
        ``p = 0`` for a non-zero difference and ``p = 1`` for a zero one, which
        is the limit of the statistic rather than an undefined value.
    """
    differences = _paired(a, b)
    n = len(differences)
    if n == 0:
        return {"n": 0, "difference": None, "t": None, "df": None, "p": None}
    average = sum(differences) / n
    if n < 2:
        return {"n": n, "difference": average, "t": None, "df": None, "p": None}
    std = sample_std(differences)
    if not std:
        return {
            "n": n,
            "difference": average,
            "t": None,
            "df": n - 1,
            "p": 1.0 if average == 0.0 else 0.0,
        }
    t = average / (std / math.sqrt(n))
    return {
        "n": n,
        "difference": average,
        "t": t,
        "df": n - 1,
        "p": 2.0 * (1.0 - student_t_cdf(abs(t), n - 1)),
    }


def average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks of ``values``, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return ranks


def _exact_signed_rank_p(statistic: float, n: int) -> float:
    """Two-sided exact p-value from the enumerated null distribution."""
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for value in range(total, rank - 1, -1):
            counts[value] += counts[value - rank]
    at_or_below = sum(counts[: int(math.floor(statistic)) + 1])
    return min(1.0, 2.0 * at_or_below / float(2**n))


def wilcoxon_signed_rank(a: Sequence, b: Sequence) -> dict:
    """
    Two-sided Wilcoxon signed-rank test of ``a - b``.

    Zero differences are dropped (the classic Wilcoxon treatment).  The null
    distribution is enumerated exactly for up to
    :data:`EXACT_WILCOXON_MAX_N` untied pairs and approximated by the
    tie-corrected normal otherwise.

    Returns:
        dict: ``{"n", "statistic", "p", "method"}``.
    """
    differences = [value for value in _paired(a, b) if value != 0.0]
    n = len(differences)
    if n == 0:
        return {"n": 0, "statistic": None, "p": None, "method": "none"}

    magnitudes = [abs(value) for value in differences]
    ranks = average_ranks(magnitudes)
    positive = sum(rank for rank, value in zip(ranks, differences) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, differences) if value < 0)
    statistic = min(positive, negative)

    tied = len(set(magnitudes)) != n
    if n <= EXACT_WILCOXON_MAX_N and not tied:
        return {
            "n": n,
            "statistic": statistic,
            "p": _exact_signed_rank_p(statistic, n),
            "method": "exact",
        }

    expected = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    groups: dict[float, int] = {}
    for magnitude in magnitudes:
        groups[magnitude] = groups.get(magnitude, 0) + 1
    variance -= sum(size**3 - size for size in groups.values()) / 48.0
    if variance <= 0:
        return {"n": n, "statistic": statistic, "p": 1.0, "method": "normal"}
    z = (statistic - expected + 0.5) / math.sqrt(variance)
    return {
        "n": n,
        "statistic": statistic,
        "p": min(1.0, 2.0 * normal_cdf(z)),
        "method": "normal",
    }


def cohens_d_paired(a: Sequence, b: Sequence) -> Optional[float]:
    """
    Cohen's ``d_z``: the mean difference in units of its own spread.

    The paired form is the one the seed-matched design calls for; ``None`` when
    fewer than two pairs survive or the differences are constant.
    """
    differences = _paired(a, b)
    if len(differences) < 2:
        return None
    std = sample_std(differences)
    if not std:
        return None
    return (sum(differences) / len(differences)) / std


# --------------------------------------------------------------------------- #
# multiplicity
# --------------------------------------------------------------------------- #
def holm(pvalues: Sequence[Optional[float]]) -> list[Optional[float]]:
    """
    Holm step-down adjusted p-values, in the order they were given.

    Entries that are ``None`` (a comparison that could not be made) take no part
    in the correction and stay ``None``, so a table with gaps is not penalised
    for tests it never ran.
    """
    indexed = [(index, p) for index, p in enumerate(pvalues) if p is not None]
    adjusted: list[Optional[float]] = [None] * len(pvalues)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 0.0
    for position, (index, p) in enumerate(indexed):
        running = max(running, min(1.0, (m - position) * p))
        adjusted[index] = running
    return adjusted


def significance_marker(p: Optional[float]) -> str:
    """``***``/``**``/``*`` for the usual thresholds, empty string otherwise."""
    if p is None:
        return ""
    for threshold, marker in SIGNIFICANCE_LEVELS:
        if p < threshold:
            return marker
    return ""
