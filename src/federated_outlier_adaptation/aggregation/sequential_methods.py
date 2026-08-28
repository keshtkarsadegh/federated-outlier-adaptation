from copy import deepcopy

"""
Sequential aggregation strategies for federated learning.

This module implements sequential (one-by-one) update rules where each
client update is integrated into the global model in order of arrival.
Two categories are provided:

- Weight-based updates (SequentialWeightsAGGMETHODS):
    * seq_fixed_ratio_update        : fixed 70/30 global-client blend
    * seq_equal_update              : equal incremental averaging (α = 1/(i+1))
    * seq_fedavg_update             : weighted by client sample proportion
    * seq_incremental_update        : progressive weighting based on index
    * seq_mix_alpha                 : the mixing weight as a swept scalar

- Delta-based updates (SequentialDeltaAGGMETHODS):
    * seq_delta_fedavg_update       : FedAvg using deltas weighted by sample proportion
    * seq_delta_scaled              : deltas scaled equally by 1/K
    * seq_delta_capped              : deltas capped at 1/K to prevent domination
    * seq_delta_progressive_update  : progressive deltas with α increasing over time

The first three of those weight-based rules hard-code a mixing coefficient -
0.7/0.3, 1/(i+1), i/(K+i) - that no reference justifies.  ``seq_mix_alpha``
replaces the three constants with one swept scalar, so the cyclic schedule's
mixing weight becomes an axis with a range instead of three arbitrary points.
The published rules are untouched and still run under ``--aggregation all``.

These methods support keyword arguments such as:
    - index: position of the client in sequential order (int)
    - num_clients: total number of clients (int)
    - client_sample_count: number of samples for this client (int)
    - all_clients_samples: total samples across all clients (int)
    - alpha: the swept mixing weight (float), used by seq_mix_alpha

They are grouped into containers (SequentialWeightsAGGMETHODS and
SequentialDeltaAGGMETHODS) for easy iteration and discovery.
"""

def seq_fixed_ratio_update(global_weights, client_weight,**kwargs):
    """
    Sequential Fixed-Ratio Update.

    Blends the global model with the current client model
    using a fixed ratio: 70% global, 30% client.

    Args:
        global_weights (dict): Current global model state_dict.
        client_weight (dict): State_dict from the current client.
        **kwargs: Unused extra arguments.

    Returns:
        dict: Updated global model weights.
    """

    new_w = {}
    for key in global_weights:
        new_w[key] =0.7* global_weights[key] +0.3  * client_weight[key]
    return new_w
def seq_equal_update(global_weights, client_weight, **kwargs):
    """
    Sequential Equal Update.

    Performs incremental averaging where each client is weighted equally
    in order of arrival. The update weight α decreases with index:
        α = 1 / (index + 1)

    Args:
        global_weights (dict): Current global model state_dict.
        client_weight (dict): State_dict from the current client.
        **kwargs:
            index (int): Sequential index of the client update (0-based).

    Returns:
        dict: Updated global model weights.
    """

    index = kwargs.get("index", 0)
    alpha = 1 / (index + 1)

    aggregated = {}
    for k in global_weights:
        aggregated[k] = (1 - alpha) * global_weights[k] + alpha * client_weight[k]
    return aggregated
def seq_fedavg_update(global_weights, client_weight, **kwargs):
    """
    Sequential FedAvg Update.

    Weights the client update proportional to its sample count
    relative to the total number of samples across all clients.

    α = n_k / N

    Args:
        global_weights (dict): Current global model state_dict.
        client_weight (dict): State_dict from the current client.
        **kwargs:
            client_sample_count (int): Number of samples for this client.
            all_clients_samples (int): Total number of samples across all clients.

    Returns:
        dict: Updated global model weights.
    """

    client_sample_count = kwargs.get("client_sample_count", 0)
    all_clients_samples = kwargs.get("all_clients_samples", 0)
    alpha = client_sample_count / all_clients_samples if all_clients_samples > 0 else 0

    aggregated = {}
    for k in global_weights:
        aggregated[k] = (1 - alpha) * global_weights[k] + alpha * client_weight[k]
    return aggregated

def seq_incremental_update(global_weights, client_weight, **kwargs):
    """
    Sequential Incremental Update.

    Uses a progressive weighting factor that increases with index:
        α = index / (num_clients + index)

    Later clients contribute more strongly to the update.

    Args:
        global_weights (dict): Current global model state_dict.
        client_weight (dict): State_dict from the current client.
        **kwargs:
            num_clients (int): Total number of clients.
            index (int): Sequential index of the client update (0-based).

    Returns:
        dict: Updated global model weights.
    """

    num_clients = kwargs.get("num_clients", 0)
    index = kwargs.get("index", 0)
    alpha = index / (num_clients + index)

    aggregated = {}
    for k in global_weights:
        aggregated[k] = (1 - alpha) * global_weights[k] + alpha * client_weight[k]
    return aggregated


#: Mixing weight of :func:`seq_mix_alpha` when the runner passes none.
DEFAULT_MIX_ALPHA = 0.5


def seq_mix_alpha(global_weights, client_weight, **kwargs):
    """
    Sequential Mixing with a Swept Weight.

    The cyclic update with the mixing coefficient as an explicit scalar::

        theta <- (1 - alpha) * theta + alpha * theta_k

    ``alpha = n_k / N`` reproduces :func:`seq_fedavg_update`, ``alpha = 0.3``
    reproduces :func:`seq_fixed_ratio_update`, and ``alpha = 1 / (i + 1)``
    reproduces :func:`seq_equal_update`; the point of the rule is that the
    coefficient is swept instead of being one of those three constants.

    Args:
        global_weights (dict): Current global model state_dict.
        client_weight (dict): State_dict from the current client.
        **kwargs:
            alpha (float): Mixing weight in [0, 1].  Defaults to
                :data:`DEFAULT_MIX_ALPHA`.

    Returns:
        dict: Updated global model weights.
    """
    alpha = kwargs.get("alpha")
    alpha = DEFAULT_MIX_ALPHA if alpha is None else float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"The sequential mixing alpha must lie in [0, 1]; got {alpha!r}")

    aggregated = {}
    for k in global_weights:
        aggregated[k] = (1 - alpha) * global_weights[k] + alpha * client_weight[k]
    return aggregated


class SequentialWeightsAGGMETHODS:
    seq_fixed_ratio_update=staticmethod(seq_fixed_ratio_update)
    seq_equal_update=staticmethod(seq_equal_update)
    seq_fedavg_update=staticmethod(seq_fedavg_update)
    seq_incremental_update=staticmethod(seq_incremental_update)

    # --- extended rules, opt-in (see EXTENDED_METHODS) ---
    seq_mix_alpha=staticmethod(seq_mix_alpha)

    #: Names withheld from the default sweeps, so the published method set of
    #: every existing sweep is unchanged.
    EXTENDED_METHODS = ("seq_mix_alpha",)

    def __iter__(self):
        for name in dir(self):
            if name.startswith("seq_"):
                method = getattr(self, name)
                if callable(method):
                    yield method

    @classmethod
    def list_methods(cls):
        names, fns = [], []
        for name in dir(cls):
            if name.startswith("seq_"):
                # getattr on a staticmethod returns the underlying function
                fn = getattr(cls, name)
                if callable(fn):
                    names.append(name)
                    fns.append(fn)
        return names, fns




def seq_delta_fedavg_update(global_weights, client_weight,**kwargs):
    """
    Sequential Delta FedAvg Update.

    Applies the FedAvg principle but in delta form:
    the difference between client and global is scaled
    by the client’s relative sample proportion.

    Args:
        global_weights (dict): Current global model state_dict.
        client_weight (dict): State_dict from the current client.
        **kwargs:
            client_sample_count (int): Number of samples for this client.
            all_clients_samples (int): Total number of samples across all clients.

    Returns:
        dict: Updated global model weights.
    """

    client_sample_count=kwargs.get("client_sample_count",0)
    all_clients_samples=kwargs.get("all_clients_samples",0)
    total_samples = all_clients_samples
    new_weights = deepcopy(global_weights)
    for key in global_weights:
        delta = client_weight[key] - global_weights[key]
        weighted_delta = (client_sample_count / total_samples) * delta
        new_weights[key] = global_weights[key] + weighted_delta
    return new_weights


def seq_delta_scaled(global_weights, client_weight,**kwargs):
    """
    Sequential Delta Scaled Update.

    Each client contributes equally, regardless of sample size.
    The delta is scaled by 1 / num_clients.

    Args:
        global_weights (dict): Current global model state_dict.
        client_weight (dict): State_dict from the current client.
        **kwargs:
            num_clients (int): Total number of clients.

    Returns:
        dict: Updated global model weights.
    """

    num_clients=kwargs.get("num_clients",0)
    scale = 1.0 / num_clients
    new_weights = deepcopy(global_weights)

    for key in global_weights:
        delta = client_weight[key] - global_weights[key]
        weighted_delta = scale  * delta
        new_weights[key] = global_weights[key] + weighted_delta

    return new_weights


def seq_delta_capped(global_weights, client_weight,**kwargs):
    """
    Sequential Delta Capped Update.

    Client contribution is capped at 1 / num_clients,
    ensuring no single client dominates the update.

    Args:
        global_weights (dict): Current global model state_dict.
        client_weight (dict): State_dict from the current client.
        **kwargs:
            client_sample_count (int): Number of samples for this client.
            all_clients_samples (int): Total number of samples across all clients.
            num_clients (int): Total number of clients.

    Returns:
        dict: Updated global model weights.
    """

    client_sample_count=kwargs.get("client_sample_count",0)
    all_clients_samples=kwargs.get("all_clients_samples",0)
    num_clients=kwargs.get("num_clients",0)
    total_samples = all_clients_samples
    max_allowed = 1.0 / num_clients
    new_weights = deepcopy(global_weights)

    # p_k capped at 1/K
    norm_weight = min(client_sample_count / total_samples, max_allowed)

    for key in global_weights:
        delta = client_weight[key] - global_weights[key]
        weighted_delta =norm_weight * delta
        new_weights[key] = global_weights[key] + weighted_delta

    return new_weights
def seq_delta_progressive_update(global_weights, client_weight,**kwargs):
    """
    Sequential Delta Progressive Update.

    Applies a progressively increasing weight to each client’s delta:
        α = index / (num_clients + index)

    Later updates are emphasized more than earlier ones.

    Args:
        global_weights (dict): Current global model state_dict.
        client_weight (dict): State_dict from the current client.
        **kwargs:
            index (int): Sequential index of the client update (0-based).
            num_clients (int): Total number of clients.

    Returns:
        dict: Updated global model weights.
    """

    index=kwargs.get("index",0)
    num_clients=kwargs.get("num_clients",0)
    alpha = index / (num_clients+index)
    new_weights = deepcopy(global_weights)

    for key in global_weights:
        delta = client_weight[key] - global_weights[key]
        new_weights[key] = global_weights[key] + alpha * delta
    return new_weights



class SequentialDeltaAGGMETHODS:
    seq_delta_fedavg_update=staticmethod(seq_delta_fedavg_update)
    seq_delta_scaled=staticmethod(seq_delta_scaled)
    seq_delta_capped=staticmethod(seq_delta_capped)
    seq_delta_progressive_update=staticmethod(seq_delta_progressive_update)
    def __iter__(self):
        for name in dir(self):
            if name.startswith("seq_"):
                method = getattr(self, name)
                if callable(method):
                    yield method

    @classmethod
    def list_methods(cls):
        names, fns = [], []
        for name in dir(cls):
            if name.startswith("seq_"):
                # getattr on a staticmethod returns the underlying function
                fn = getattr(cls, name)
                if callable(fn):
                    names.append(name)
                    fns.append(fn)
        return names, fns
