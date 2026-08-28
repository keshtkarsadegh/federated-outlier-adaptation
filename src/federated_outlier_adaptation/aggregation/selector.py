
"""

Purpose:
    Centralized selector for aggregation method classes in federated adaptive learning.

Provides:
    - CLASS_MAP: dictionary mapping (scenario, metadata) to the correct aggregation class.
    - select_class: safe lookup function that returns the appropriate class or raises.

Scenarios:
    - "sequential" vs "concurrent"
Metadata:
    - "weights" vs "delta"

"""


from federated_outlier_adaptation.aggregation.concurrent_methods import \
    ConcurrentWeightsAGGMETHODS,ConcurrentDeltaAGGMETHODS
from federated_outlier_adaptation.aggregation.sequential_methods import \
    SequentialWeightsAGGMETHODS, SequentialDeltaAGGMETHODS


# Example: map by (module_id, class_id)
CLASS_MAP = {
    "sequential": {"weights": SequentialWeightsAGGMETHODS, "delta":SequentialDeltaAGGMETHODS},
    "concurrent": {"weights": ConcurrentWeightsAGGMETHODS, "delta":ConcurrentDeltaAGGMETHODS}
}

def select_class(scenario, metadata):
    try:
        return CLASS_MAP[scenario][metadata]
    except KeyError:
        raise ValueError(f"Invalid selection: {scenario}, {metadata}")


#: Sentinel returned by :func:`resolve_aggregation` when the requested rule
#: does not exist in a family, so the caller can skip that job.
SKIP_FAMILY = "__skip__"

# --------------------------------------------------------------------------- #
# The FedAvg pair
# --------------------------------------------------------------------------- #
# Of the four (scenario, metadata) families, the ``fedavg`` variant used to name
# one rule in each - and two of those four were algebraically the same update as
# the other two: ``con_weighted_cw`` is ``con_delta_weighted_cgd`` written on
# weights instead of deltas, and ``seq_fedavg_update`` is
# ``seq_delta_fedavg_update`` written the same way.  Running all four therefore
# spent half the GPU time reproducing run-to-run noise and reported it as two
# settings.  The variant now runs **two** families, one per schedule, and the
# rule of each is a single constant so that a sweep and the final it feeds are
# executed with the identical update:
#
#     concurrent  theta + sum_k (n_k/N) (theta_k - theta)       (server step 1)
#     sequential  theta <- (1 - n_k/N) theta + (n_k/N) theta_k
#
#: Family and rule of the concurrent schedule: the delta form with eta_s = 1.
FEDAVG_CONCURRENT_FAMILY = ("concurrent", "delta")
FEDAVG_CONCURRENT_RULE = "con_delta_weighted_cgd"

#: Family and rule of the sequential schedule.
FEDAVG_SEQUENTIAL_FAMILY = ("sequential", "weights")
FEDAVG_SEQUENTIAL_RULE = "seq_fedavg_update"

#: The two families the ``fedavg`` variant runs, with their rule.
FEDAVG_FAMILIES = {
    FEDAVG_CONCURRENT_FAMILY: FEDAVG_CONCURRENT_RULE,
    FEDAVG_SEQUENTIAL_FAMILY: FEDAVG_SEQUENTIAL_RULE,
}

#: The two families it skips, because their rule duplicates one of the above.
FEDAVG_DUPLICATE_FAMILIES = (("concurrent", "weights"), ("sequential", "delta"))


def _fedavg_variant() -> dict:
    """The ``fedavg`` variant map: two rules, two skipped duplicates."""
    variant = {family: SKIP_FAMILY for family in FEDAVG_DUPLICATE_FAMILIES}
    variant.update(FEDAVG_FAMILIES)
    return variant


#: Named aggregation variants: one representative rule per family.
#:
#: ``all`` keeps the published behaviour of running every rule of the family.
#: The other keys pick a single comparable rule per family so that a run of the
#: four (scenario, metadata) jobs answers one question instead of sixteen.
AGG_VARIANTS = {
    "all": None,
    "fedavg": _fedavg_variant(),
    "anchored": {
        ("concurrent", "weights"): "con_weighted_cgw",
        ("concurrent", "delta"): "con_delta_weighted_cgd",
        ("sequential", "weights"): "seq_fedavg_update",
        ("sequential", "delta"): "seq_delta_fedavg_update",
    },
    "capped": {
        ("concurrent", "weights"): "con_capped_cgw",
        ("concurrent", "delta"): "con_delta_capped_cgd",
        ("sequential", "weights"): "seq_fedavg_update",
        ("sequential", "delta"): "seq_delta_fedavg_update",
    },
    # The cyclic question - in which order and with which schedule the clients
    # are visited one after another - lives entirely in the sequential families,
    # so this key runs every schedule of those two and skips the parallel ones.
    # One task therefore answers the whole question instead of four.
    "cyclic": {
        ("concurrent", "weights"): SKIP_FAMILY,
        ("concurrent", "delta"): SKIP_FAMILY,
        ("sequential", "weights"): None,
        ("sequential", "delta"): None,
    },
}


def resolve_aggregation(scenario: str, metadata: str, aggregation=None, extended: bool = False):
    """
    Resolve ``--aggregation`` for one (scenario, metadata) family.

    Args:
        scenario, metadata: The family the job belongs to.
        aggregation: ``None``/``"all"``/``"none"`` runs every rule of the
            family (the published behaviour).  A key of :data:`AGG_VARIANTS`
            picks the family's representative.  Anything else is treated as a
            concrete rule name.
        extended: Allow the opt-in server-optimiser rules to be named.

    Returns:
        str | None: The rule name, ``None`` for "every rule of the family", or
        :data:`SKIP_FAMILY` when a concrete rule does not exist here.
    """
    if aggregation in (None, "", "none", "all"):
        return None
    if aggregation in AGG_VARIANTS:
        return AGG_VARIANTS[aggregation][(scenario, metadata)]
    cls = select_class(scenario, metadata)
    if not hasattr(cls, aggregation):
        return SKIP_FAMILY
    if not extended and aggregation in getattr(cls, "EXTENDED_METHODS", ()):
        # Naming an opt-in rule explicitly is enough to enable it.
        return aggregation
    return aggregation


def list_method_names(cls, extended: bool = False):
    """
    Aggregation method names of a container class.

    Methods listed in the class attribute ``EXTENDED_METHODS`` (currently the
    server-optimiser rules) are opt-in: they are withheld unless ``extended`` is
    True, so the published sweeps keep the exact method set they were run with.

    Args:
        cls: One of the four aggregation container classes.
        extended (bool): Include the opt-in methods.

    Returns:
        list[str]: Method names, in the order the container reports them.
    """
    names, _ = cls.list_methods()
    if extended:
        return list(names)
    withheld = set(getattr(cls, "EXTENDED_METHODS", ()))
    return [name for name in names if name not in withheld]


def accepts_server_state(method) -> bool:
    """Whether an aggregation function takes the persistent ``server_state``."""
    import inspect

    try:
        return "server_state" in inspect.signature(method).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
