from copy import deepcopy

import torch

"""
Concurrent aggregation strategies for federated learning.

This module implements several variants:
- *_cw  : client-only weighted/scaled/capped averaging
- *_cgw : client + global mixed averaging
- *_cgd : delta-based updates (client update relative to global)

Grouped into:
    - ConcurrentWeightsAGGMETHODS: operates directly on weights
    - ConcurrentDeltaAGGMETHODS : operates on deltas
"""


def con_weighted_cw(global_weight, clients_weights_list, client_sample_counts):
    """
        Concurrent Weighted Aggregation (only client weights).

        Each client model is weighted by its sample count relative to the total,
        and then averaged to form the new model. No global mixing.

        Args:
            global_weight (dict): Current global model state_dict.
            clients_weights_list (list[dict]): List of client state_dicts.
            client_sample_counts (list[int]): Number of samples per client.

        Returns:
            dict: Averaged model weights.
        """
    avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    total = sum(client_sample_counts)
    for key in avg:
        for w, cnt in zip(clients_weights_list, client_sample_counts):
            avg[key] += w[key] * (cnt / total)
    return avg

def con_weighted_cgw(global_weight, clients_weights_list, client_sample_counts):
    """
    Concurrent Weighted Aggregation (clients and global model).

    Clients are aggregated weighted by sample counts (like con_weighted_cw),
    then the result is mixed 50/50 with the existing global model.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    client_avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    total = sum(client_sample_counts)
    for key in client_avg:
        for w, cnt in zip(clients_weights_list, client_sample_counts):
            client_avg[key] += w[key] * (cnt / total)
    # mix with global
    avg = {}
    for key in global_weight:
        avg[key] = 0.5 * global_weight[key] + 0.5 * client_avg[key]
    return avg


def con_scaled_cw(global_weight, clients_weights_list, client_sample_counts):
    """
    Concurrent scaled Aggregation (client weights).

    Each client contributes equally regardless of data size.
    Average is uniform across clients. No global mixing.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """

    K = len(client_sample_counts)
    avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    scale = 1.0 / K
    for key in avg:
        for w in clients_weights_list:
            avg[key] += w[key] * (scale )
    return avg

def con_scaled_cgw(global_weight, clients_weights_list, client_sample_counts):
    """
    Concurrent scaled Aggregation (client and global weights).

    Clients are averaged uniformly (equal weights), then result is blended
    50/50 with the current global model.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    K = len(client_sample_counts)
    # compute scaled client avg
    client_avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    scale = 1.0 / K
    for key in client_avg:
        for w in clients_weights_list:
            client_avg[key] += w[key] * (scale )
    # mix with global
    avg = {}
    for key in global_weight:
        avg[key] = 0.5* global_weight[key] + 0.5 * client_avg[key]
    return avg


def con_capped_cw(global_weight, clients_weights_list, client_sample_counts):
    """
     Concurrent Capped Aggregation (client weights).

     Client contributions are capped so no single client dominates:
    - Proportions = min(n_i / total, 1/K)
    - Then renormalized and used for weighted averaging.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    K = len(client_sample_counts)
    total = sum(client_sample_counts)
    # Step 1: cap proportions
    raw = [min(cnt/total, 1.0/K) for cnt in client_sample_counts]
    # Step 2: renormalize
    Z = sum(raw)
    q = [r / Z for r in raw]
    # aggregate
    avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    for key in avg:
        for w, qk in zip(clients_weights_list, q):
            avg[key] += w[key] * qk
    return avg


def con_capped_cgw(global_weight, clients_weights_list, client_sample_counts):
    """
     Concurrent Capped Aggregation (client and global weights).

     Client contributions are capped so no single client dominates:
    - Proportions = min(n_i / total, 1/K)
    - Then renormalized and used for weighted averaging.
    - then blend 50/50 with the global model.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    K = len(client_sample_counts)
    total = sum(client_sample_counts)
    raw = [min(cnt/total, 1.0/K) for cnt in client_sample_counts]
    Z = sum(raw)
    q = [r / Z for r in raw]
    # client contribution
    client_part = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    for key in client_part:
        for w, qk in zip(clients_weights_list, q):
            client_part[key] += w[key] * qk
    avg = {}
    for key in global_weight:
        avg[key] = 0.5 * global_weight[key] + 0.5 * client_part[key]
    return avg
class ConcurrentWeightsAGGMETHODS:
    """
    Container class exposing concurrent aggregation methods
    operating directly on client weights (no deltas).
    """
    con_weighted_cw=staticmethod(con_weighted_cw)
    con_weighted_cgw=staticmethod(con_weighted_cgw)
    con_scaled_cw=staticmethod(con_scaled_cw)
    con_scaled_cgw=staticmethod(con_scaled_cgw)
    con_capped_cw=staticmethod(con_capped_cw)
    con_capped_cgw=staticmethod(con_capped_cgw)
    def __iter__(self):
        for name in dir(self):
            if name.startswith("con_"):
                method = getattr(self, name)
                if callable(method):
                    yield method

    @classmethod
    def list_methods(cls):
        """
        Returns:
            (list[str], list[function]):
            Names and function handles of all con_* methods.
        """
        names, fns = [], []
        for name in dir(cls):
            if name.startswith("con_"):
                # getattr on a staticmethod returns the underlying function
                fn = getattr(cls, name)
                if callable(fn):
                    names.append(name)
                    fns.append(fn)
        return names, fns






def con_delta_weighted_cgd(global_weights, clients_weights_list, client_sample_counts):
    """
    Concurrent Delta Aggregation (Weighted).

    Clients contribute deltas (difference from global).
    Each delta is weighted by client sample proportion,
    then added to the global model.
    Args:
        global_weights (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    total = sum(client_sample_counts)
    new_w = deepcopy(global_weights)

    for key in global_weights:
        delta_sum = torch.zeros_like(global_weights[key])

        for w_k, n_k in zip(clients_weights_list, client_sample_counts):
            delta = w_k[key] - global_weights[key]
            delta_sum += (n_k / total) * delta

        # Weighted combination: 0.7 * global + 0.3 * delta_sum
        new_w[key] =  global_weights[key] + delta_sum

    return new_w


def con_delta_scaled_cgd(global_weights, clients_weights_list, client_sample_counts):
    """
    Concurrent Delta Aggregation (Scaled).

    Each client contributes equally (1/K factor),
    deltas are averaged uniformly and added to global.
    Args:
        global_weights (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    K = len(client_sample_counts)
    factor = 1.0 / (K )

    new_w = deepcopy(global_weights)
    for key in global_weights:
        delta_sum = torch.zeros_like(global_weights[key])
        for w_k in clients_weights_list:
            delta = w_k[key] - global_weights[key]
            delta_sum += factor  * delta
        new_w[key] =  global_weights[key] +  delta_sum


    return new_w

def con_delta_capped_cgd(global_weights, clients_weights_list, client_sample_counts):
    """
    Concurrent Delta Aggregation (Capped).

    Client proportions are capped at 1/K to prevent domination.
    Normalized capped proportions weight the client deltas,
    which are then added to the global model.
    Args:
        global_weights (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    total = sum(client_sample_counts)
    K = len(client_sample_counts)
    max_cap = 1.0 / K

    # Step 1: compute raw proportions and cap
    raw = [min(n_k / total, max_cap) for n_k in client_sample_counts]
    # Step 2: renormalize
    Z = sum(raw)
    q = [r / Z for r in raw]

    new_w = deepcopy(global_weights)
    for key in global_weights:
        delta_sum = torch.zeros_like(global_weights[key])
        for w_k, q_k in zip(clients_weights_list, q):
            delta = w_k[key] - global_weights[key]
            delta_sum +=   q_k * delta
        new_w[key] =  global_weights[key] + delta_sum


    return new_w

# --------------------------------------------------------------------------- #
# Extended parallel rules
# --------------------------------------------------------------------------- #
# The rules above are stateless: every round starts from the current global
# weights and nothing is carried over, and each of them hard-codes one server
# step size and one weighting.  The rules below make those choices explicit and
# add robust, anchored and optimiser-based servers.  They need state that lives
# longer than one call - the frozen initial model, a momentum buffer, the
# configured step size - so the runner owns a :class:`ServerState` and hands it
# to the aggregation function through the keyword ``server_state``.  The
# published rules do not accept that keyword and are called exactly as before.
#
# Every parallel rule can be written as
#
#     theta_{t+1} = theta_t + eta_s * A({Delta_k}, {p_k}) - lambda_s (theta_t - theta_g)
#
# with the client update Delta_k = theta_k - theta_t, an aggregator A (weighted
# mean, coordinate-wise median, trimmed mean, or an optimiser step), weights p_k
# (proportional / uniform / capped) and an optional pull lambda_s towards the
# frozen global model theta_g.

#: Defaults of Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
SERVER_LR = 1e-2
SERVER_BETA1 = 0.9
SERVER_BETA2 = 0.99
SERVER_TAU = 1e-3
#: Server momentum of FedAvgM, applied with a server step size of 1.0.
SERVER_MOMENTUM = 0.9
FEDAVGM_LR = 1.0
#: Fraction trimmed from each end per coordinate by the trimmed-mean rule.
TRIM_FRACTION = 0.2

#: Client weighting schemes shared by the extended rules.
WEIGHTINGS = ("proportional", "uniform", "capped")

#: Default pull of the parameterised server anchor, matching the fixed rule
#: ``con_delta_anchor_lam01`` that predates it.
ANCHOR_LAMBDA = 0.1


class ServerState:
    """
    Everything a parallel aggregation rule needs beyond the current round.

    The published rules read their coefficients from the module constants; the
    same coefficients are attributes here so that a run can sweep them.  Every
    default is the constant, so a state constructed without arguments - which is
    what a rule called with ``server_state=None`` builds - behaves exactly as
    before.  The literature grids these exist for are listed in
    ``grid_ranges_from_literature.md`` section 3: FedAvgM's server momentum
    beta in {0.7, 0.9, 0.97, 0.99, 0.997}, FedAdam/FedYogi's server learning
    rate and adaptivity tau on a log grid, and the trimmed mean's trim fraction
    in {10 %, 20 %}.

    Attributes:
        momentum: First-moment buffer per parameter key (FedAvgM, FedOpt).
        second: Second-moment buffer per parameter key (FedAdam, FedYogi).
        step: Number of aggregation steps applied so far.
        frozen_global: State dict of the model the run started from, used by
            the anchored rule.  The runner fills it in before the first round.
        weighting: Client weighting scheme of the extended rules.
        eta: Server step size of the robust and anchored rules.
        tau: Adaptivity constant of FedAdam/FedYogi (also the initial value of
            the second moment, as in Reddi et al.).
        server_lr: Server learning rate of FedAdam/FedYogi.
        beta1, beta2: Moment decays of FedAdam/FedYogi.
        momentum_beta: Server momentum of FedAvgM.
        fedavgm_lr: Server step size of FedAvgM.
        trim_fraction: Fraction dropped at each end per coordinate by the
            trimmed mean.
    """

    def __init__(
        self,
        tau: float = SERVER_TAU,
        weighting: str = "proportional",
        eta: float = 1.0,
        server_lr: float = SERVER_LR,
        beta1: float = SERVER_BETA1,
        beta2: float = SERVER_BETA2,
        momentum_beta: float = SERVER_MOMENTUM,
        fedavgm_lr: float = FEDAVGM_LR,
        trim_fraction: float = TRIM_FRACTION,
        anchor_lambda: float = ANCHOR_LAMBDA,
    ):
        if weighting not in WEIGHTINGS:
            raise ValueError(f"Unknown weighting {weighting!r}; expected one of {WEIGHTINGS}")
        if not 0.0 <= float(trim_fraction) < 0.5:
            raise ValueError(
                f"trim_fraction must lie in [0, 0.5); got {trim_fraction!r}"
            )
        self.momentum = {}
        self.second = {}
        self.step = 0
        self.tau = float(tau)
        self.weighting = weighting
        self.eta = float(eta)
        self.server_lr = float(server_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.momentum_beta = float(momentum_beta)
        self.fedavgm_lr = float(fedavgm_lr)
        self.trim_fraction = float(trim_fraction)
        if float(anchor_lambda) < 0.0:
            raise ValueError(
                f"anchor_lambda is a pull towards theta_g and cannot be "
                f"negative; got {anchor_lambda!r}"
            )
        self.anchor_lambda = float(anchor_lambda)
        self.frozen_global = None

    def hyperparameters(self) -> dict:
        """The configured server coefficients, for the provenance record."""
        return {
            "weighting": self.weighting,
            "server_eta": self.eta,
            "server_lr": self.server_lr,
            "server_tau": self.tau,
            "server_beta1": self.beta1,
            "server_beta2": self.beta2,
            "server_momentum_beta": self.momentum_beta,
            "fedavgm_lr": self.fedavgm_lr,
            "trim_fraction": self.trim_fraction,
            "anchor_lambda": self.anchor_lambda,
        }

    def reset(self):
        """Clear the optimiser buffers; configuration and anchor are kept."""
        self.momentum.clear()
        self.second.clear()
        self.step = 0

    def momentum_for(self, key, reference):
        if key not in self.momentum:
            self.momentum[key] = torch.zeros_like(reference, dtype=torch.float32)
        return self.momentum[key]

    def second_for(self, key, reference):
        if key not in self.second:
            # Reddi et al. initialise the second moment at tau^2.
            self.second[key] = torch.full_like(
                reference, self.tau ** 2, dtype=torch.float32
            )
        return self.second[key]


#: Backwards-compatible alias of the original, optimiser-only name.
ServerOptimizerState = ServerState


def client_weights(client_sample_counts, weighting: str = "proportional"):
    """
    Normalised client weights ``p_k`` of one round.

    ``proportional`` is ``n_k / N`` (FedAvg), ``uniform`` gives every client
    ``1 / K``, and ``capped`` caps the proportional share at ``1 / K`` and
    renormalises, which is the scheme the published ``*_capped_*`` rules use.
    """
    counts = list(client_sample_counts)
    K = len(counts)
    if K == 0:
        return []
    if weighting == "uniform":
        return [1.0 / K] * K
    total = sum(counts) or 1
    raw = [count / total for count in counts]
    if weighting == "capped":
        raw = [min(value, 1.0 / K) for value in raw]
    Z = sum(raw) or 1.0
    return [value / Z for value in raw]


def _state(server_state):
    return server_state if server_state is not None else ServerState()


def _client_deltas(global_weights, clients_weights_list, key):
    """Stacked client updates of one parameter: ``[K, *shape]``."""
    reference = global_weights[key].to(torch.float32)
    return torch.stack([w[key].to(torch.float32) - reference for w in clients_weights_list])


def _weighted_delta(global_weights, clients_weights_list, client_sample_counts, weighting):
    """``Delta = sum_k p_k (theta_k - theta)`` for every parameter."""
    p = client_weights(client_sample_counts, weighting)
    delta = {}
    for key in global_weights:
        accumulated = torch.zeros_like(global_weights[key], dtype=torch.float32)
        for weights, weight in zip(clients_weights_list, p):
            accumulated += weight * (
                weights[key].to(torch.float32) - global_weights[key].to(torch.float32)
            )
        delta[key] = accumulated
    return delta


def _proportional_delta(global_weights, clients_weights_list, client_sample_counts):
    """
    Average client update with FedAvg weights.

    ``Delta`` is the step plain FedAvg would take; the FedOpt pseudo-gradient is
    ``-Delta``.
    """
    return _weighted_delta(
        global_weights, clients_weights_list, client_sample_counts, "proportional"
    )


def _apply(global_weights, delta, eta):
    """``theta + eta * delta`` for every key."""
    new_weights = deepcopy(global_weights)
    for key in global_weights:
        new_weights[key] = global_weights[key] + eta * delta[key]
    return new_weights


# --- H1: explicit server step size ----------------------------------------- #
def _eta_rule(global_weights, clients_weights_list, client_sample_counts, server_state, eta):
    state = _state(server_state)
    delta = _weighted_delta(
        global_weights, clients_weights_list, client_sample_counts, state.weighting
    )
    state.step += 1
    return _apply(global_weights, delta, eta)


def con_delta_eta025(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """``theta + 0.25 * sum_k p_k Delta_k`` - a conservative server step."""
    return _eta_rule(
        global_weights, clients_weights_list, client_sample_counts, server_state, 0.25
    )


def con_delta_eta05(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """
    ``theta + 0.5 * sum_k p_k Delta_k``.

    With proportional weights this is exactly ``con_weighted_cgw``: a half step
    towards the client average is the same as blending the average 50/50 with
    the global model.
    """
    return _eta_rule(
        global_weights, clients_weights_list, client_sample_counts, server_state, 0.5
    )


def con_delta_eta1(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """``theta + 1.0 * sum_k p_k Delta_k`` - plain FedAvg."""
    return _eta_rule(
        global_weights, clients_weights_list, client_sample_counts, server_state, 1.0
    )


def con_delta_eta(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """
    ``theta + eta_s * sum_k p_k Delta_k`` with ``eta_s`` read from the state.

    The three fixed rules above pin eta_s at 0.25, 0.5 and 1.0, which is a rule
    per value and no way to ask for 0.1 or 0.75.  This one takes the step size
    from ``--server-eta`` and the weighting from ``--weighting``, so a whole row
    of a screening grid is one rule with different coefficients rather than a
    new function per cell.  ``--server-eta 1`` with proportional weights is
    plain FedAvg exactly.
    """
    state = _state(server_state)
    return _eta_rule(
        global_weights, clients_weights_list, client_sample_counts, server_state, state.eta
    )


# --- H3: robust aggregators -------------------------------------------------- #
def con_delta_median(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """
    Coordinate-wise median of the client updates.

    ``theta + eta_s * median_k(Delta_k)``, evaluated independently per
    coordinate, so a minority of extreme clients cannot drag the server.
    """
    state = _state(server_state)
    delta = {}
    for key in global_weights:
        stacked = _client_deltas(global_weights, clients_weights_list, key)
        delta[key] = stacked.median(dim=0).values
    state.step += 1
    return _apply(global_weights, delta, state.eta)


def con_delta_trimmed_mean(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """
    Coordinate-wise trimmed mean of the client updates.

    ``state.trim_fraction`` of the values is dropped at each end per coordinate
    before averaging; with too few clients to trim, this reduces to the plain
    mean.  Yin et al. experiment at 10 % and warn that an overly large fraction
    is sub-optimal, so the value is swept rather than fixed.
    """
    state = _state(server_state)
    delta = {}
    for key in global_weights:
        stacked = _client_deltas(global_weights, clients_weights_list, key)
        K = stacked.shape[0]
        trim = int(state.trim_fraction * K)
        ordered = stacked.sort(dim=0).values
        if trim > 0 and K - 2 * trim > 0:
            ordered = ordered[trim: K - trim]
        delta[key] = ordered.mean(dim=0)
    state.step += 1
    return _apply(global_weights, delta, state.eta)


# --- H4: anchoring to the frozen global model -------------------------------- #
def _anchor_rule(
    global_weights, clients_weights_list, client_sample_counts, server_state, lambda_s
):
    state = _state(server_state)
    delta = _weighted_delta(
        global_weights, clients_weights_list, client_sample_counts, state.weighting
    )
    frozen = state.frozen_global
    new_weights = deepcopy(global_weights)
    for key in global_weights:
        updated = global_weights[key] + state.eta * delta[key]
        if frozen is not None and key in frozen:
            anchor = frozen[key].to(global_weights[key].dtype).to(global_weights[key].device)
            updated = updated - lambda_s * (global_weights[key] - anchor)
        new_weights[key] = updated
    state.step += 1
    return new_weights


def con_delta_anchor_lam01(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """
    Server-side anchoring with ``lambda_s = 0.1``.

        theta_{t+1} = theta_t + eta_s sum_k p_k Delta_k - lambda_s (theta_t - theta_g)

    ``theta_g`` is the model the run started from; the runner stores it on the
    server state.  Without an anchor the rule degenerates to the plain step.
    """
    return _anchor_rule(
        global_weights, clients_weights_list, client_sample_counts, server_state, 0.1
    )


def con_delta_anchor_lam05(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """Server-side anchoring with ``lambda_s = 0.5``; see :func:`con_delta_anchor_lam01`."""
    return _anchor_rule(
        global_weights, clients_weights_list, client_sample_counts, server_state, 0.5
    )


def con_delta_anchor_lam(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """
    Server anchoring with ``lambda_s`` read from the state (``--server-anchor``).

    Same rule as :func:`con_delta_anchor_lam01`, with the pull towards
    ``theta_g`` swept instead of pinned at 0.1 or 0.5:

        theta_{t+1} = theta_t + eta_s sum_k p_k Delta_k - lambda_s (theta_t - theta_g)

    ``lambda_s = 0`` is the plain step, which makes the row's low end a genuine
    control rather than a near-miss.
    """
    state = _state(server_state)
    return _anchor_rule(
        global_weights,
        clients_weights_list,
        client_sample_counts,
        server_state,
        state.anchor_lambda,
    )


def con_delta_anchor(
    global_weights,
    clients_weights_list,
    client_sample_counts,
    server_state=None,
    lambda_s: float = 0.1,
):
    """Parameterised form of the anchored rule, for direct calls and tests."""
    return _anchor_rule(
        global_weights, clients_weights_list, client_sample_counts, server_state, lambda_s
    )


# --- H5: server optimisers --------------------------------------------------- #
def con_delta_fedavgm(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """
    FedAvgM: server momentum over the averaged client update.

        m <- beta_s * m + Delta
        theta <- theta + eta_s * m        (defaults beta_s = 0.9, eta_s = 1.0;
                                           beta_s is swept from the server state)

    Args:
        server_state (ServerState): Buffers carried across rounds.  A missing
            state degenerates to plain FedAvg for the first round.

    Returns:
        dict: Updated global model weights.
    """
    state = _state(server_state)
    delta = _weighted_delta(
        global_weights, clients_weights_list, client_sample_counts, state.weighting
    )

    new_weights = deepcopy(global_weights)
    for key, step in delta.items():
        momentum = state.momentum_for(key, step)
        momentum.mul_(state.momentum_beta).add_(step)
        new_weights[key] = global_weights[key] + state.fedavgm_lr * momentum
    state.step += 1
    return new_weights


def _fedopt(global_weights, clients_weights_list, client_sample_counts, state, yogi):
    """Shared body of FedAdam and FedYogi; ``yogi`` selects the second moment."""
    delta = _weighted_delta(
        global_weights, clients_weights_list, client_sample_counts, state.weighting
    )

    new_weights = deepcopy(global_weights)
    for key, step in delta.items():
        momentum = state.momentum_for(key, step)
        second = state.second_for(key, step)

        momentum.mul_(state.beta1).add_((1.0 - state.beta1) * step)
        squared = step * step
        if yogi:
            second.sub_((1.0 - state.beta2) * squared * torch.sign(second - squared))
        else:
            second.mul_(state.beta2).add_((1.0 - state.beta2) * squared)

        update = state.server_lr * momentum / (second.sqrt() + state.tau)
        new_weights[key] = global_weights[key] + update
    state.step += 1
    return new_weights


def con_delta_fedadam(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """
    FedAdam: Adam on the server over the averaged client update.

        m <- b1 m + (1-b1) Delta
        v <- b2 v + (1-b2) Delta^2
        theta <- theta + eta * m / (sqrt(v) + tau)

    with the defaults ``eta = 1e-2``, ``b1 = 0.9``, ``b2 = 0.99``,
    ``tau = 1e-3``; ``eta`` and ``tau`` are swept from the server state.
    """
    return _fedopt(
        global_weights, clients_weights_list, client_sample_counts, _state(server_state), False
    )


def con_delta_fedyogi(
    global_weights, clients_weights_list, client_sample_counts, server_state=None
):
    """
    FedYogi: Yogi on the server, i.e. FedAdam with a sign-controlled second moment.

        v <- v - (1-b2) Delta^2 sign(v - Delta^2)

    which prevents the effective step size from decaying as fast as Adam's.
    """
    return _fedopt(
        global_weights, clients_weights_list, client_sample_counts, _state(server_state), True
    )


class ConcurrentDeltaAGGMETHODS:
    """
    Container class exposing concurrent aggregation methods
    that operate on deltas (client update relative to global).

    The three published rules are stateless.  The FedOpt rules keep server-side
    optimiser state and are listed as *extended* methods: they are excluded from
    ``list_methods()`` sweeps unless ``extended=True`` is requested, so the
    aggregation comparison and every grid search keep the published method set.
    """
    con_delta_weighted_cgd=staticmethod(con_delta_weighted_cgd)
    con_delta_scaled_cgd=staticmethod(con_delta_scaled_cgd)
    con_delta_capped_cgd=staticmethod(con_delta_capped_cgd)

    # --- extended rules, opt-in (see EXTENDED_METHODS) ---
    con_delta_eta025=staticmethod(con_delta_eta025)
    con_delta_eta05=staticmethod(con_delta_eta05)
    con_delta_eta1=staticmethod(con_delta_eta1)
    con_delta_eta=staticmethod(con_delta_eta)
    con_delta_median=staticmethod(con_delta_median)
    con_delta_trimmed_mean=staticmethod(con_delta_trimmed_mean)
    con_delta_anchor_lam01=staticmethod(con_delta_anchor_lam01)
    con_delta_anchor_lam05=staticmethod(con_delta_anchor_lam05)
    con_delta_anchor_lam=staticmethod(con_delta_anchor_lam)
    con_delta_fedavgm=staticmethod(con_delta_fedavgm)
    con_delta_fedadam=staticmethod(con_delta_fedadam)
    con_delta_fedyogi=staticmethod(con_delta_fedyogi)

    #: Names withheld from the default sweeps, grouped by hypothesis.
    EXTENDED_METHODS = (
        # H1 server step size
        "con_delta_eta025",
        "con_delta_eta05",
        "con_delta_eta1",
        "con_delta_eta",
        # H3 robust aggregators
        "con_delta_median",
        "con_delta_trimmed_mean",
        # H4 anchoring to the frozen global model
        "con_delta_anchor_lam01",
        "con_delta_anchor_lam05",
        "con_delta_anchor_lam",
        # H5 server optimisers
        "con_delta_fedavgm",
        "con_delta_fedadam",
        "con_delta_fedyogi",
    )

    def __iter__(self):
        for name in dir(self):
            if name.startswith("con_delta_"):
                method = getattr(self, name)
                if callable(method):
                    yield method

    @classmethod
    def list_methods(cls):
        """
        Returns:
            (list[str], list[function]):
            Names and function handles of all con_delta_* methods.
        """
        names, fns = [], []
        for name in dir(cls):
            if name.startswith("con_delta_"):
                # getattr on a staticmethod returns the underlying function
                fn = getattr(cls, name)
                if callable(fn):
                    names.append(name)
                    fns.append(fn)
        return names, fns