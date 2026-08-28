"""
Client population bookkeeping shared by the two runners.

Three concerns live here so that the concurrent and the sequential loop stay in
step:

    - **Participants with multiplicity.**  The published client restriction is
      an intersection, so a writer id listed twice contributed a single
      participant.  A restriction that repeats an id is now expanded into that
      many participants, each with its own local training run, its own sample
      count and its own entry in the aggregation, while the client evaluation
      loader is still built over the *unique* ids.  Restrictions without
      duplicates keep the previous ``provider.restrict`` semantics exactly.

    - **Per-round selection.**  Each round asks a
      :class:`~federated_outlier_adaptation.runners.client_sampler.ClientSampler`
      for its participants.  With the defaults (``participation=1.0``,
      ``policy="all"``) the sampler returns the full participant list in order,
      so the loop is unchanged.

    - **Per-client tracking.**  When the sampler is active, or when tracking is
      requested explicitly, every client of the pool is evaluated on its own
      data each round.  The accuracies feed the ``worst_first`` policy and are
      reported next to the round accuracies.

    - **Held-out client accuracy.**  The published ``accuracies`` entries
      evaluate the clients on their *full* data, which includes the 60% the
      clients train on.  That series is kept exactly as it is; in addition the
      aggregated global model is scored each round on the union of the whole
      pool's 40% evaluation splits (the same ``seed=42`` split the local
      training uses, so the held-out set is stable across rounds).  The result
      is reported under the new key ``heldout_client_accuracies``; when the
      sampler is active the same quantity restricted to the round's
      participants is reported as ``heldout_participant_accuracies``.

Evaluation cost
---------------
All of the above used to run through one ``DataLoader`` per client and per
split, decoding every image again in every round.  When the run hands in a
:class:`~federated_outlier_adaptation.utils.eval_cache.GpuEvalCache`, the three
pool sets are materialised once and each round costs one batched forward pass
per set: the pooled numbers, the per-client numbers and the forgetting signals
are all read off the same pass.  The arithmetic is unchanged - a per-client
accuracy is still ``correct / count`` and a pooled number is still the
sample-count weighted mean of those - so the recorded series are the same as on
the loader path.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Mapping, Optional, Sequence

from torch.utils.data import DataLoader, Subset

from federated_outlier_adaptation.runners.client_sampler import ClientSampler
from federated_outlier_adaptation.utils.eval_cache import (
    POOL_CLIENTS_SET,
    POOL_TEST_SET,
    POOL_VAL_SET,
)

#: Split seed of the published pipeline; the validation/test halves reuse it
#: so the division is identical in every run.
SPLIT_SEED = 42


def booked_split(provider, client_id: str, batch_size: int, loader_seed):
    """
    The book's own validation and test loaders for one client, or ``None``.

    A fold book *is* the split.  It recorded which rows are train, which are
    validation and which are test, and the third of those is what a reported
    number is supposed to be measured on.  Without this the run would take the
    book's validation rows, cut them in half a second time here, and call one
    half "test" - so a book fold's actual test partition would never be read at
    all and the reported adaptation number would be measured on rows the
    early-stopping rule had already seen.

    ``None`` when no book is configured or the book does not cover this client,
    in which case the caller keeps the seeded halves and nothing changes.
    """
    dataset = getattr(provider, "dataset", None)
    book_split = getattr(dataset, "book_split", None)
    if book_split is None:
        return None
    from federated_outlier_adaptation.data.merged_clients import members
    from federated_outlier_adaptation.utils.seeding import configured_fold

    # A merged client is covered exactly when each of its members is.
    if book_split(members(client_id), configured_fold()) is None:
        return None
    _, val_loader, test_loader = provider.build_dataset(
        client_id,
        train_rate=0.6,
        eval_rate=0.4,
        batch_size=batch_size,
        loader_seed=loader_seed,
    )
    return val_loader, test_loader


def stratified_halves(dataset, seed: int = SPLIT_SEED):
    """
    Split a dataset's indices into two label-balanced halves.

    Deterministic for a given seed and independent of the sample order, so
    the validation and test halves of a client are the same in every round
    and in every run.

    Returns:
        tuple[list[int], list[int]]: ``(validation indices, test indices)``.
    """
    by_label: dict[Any, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        label = dataset[index][1]
        by_label[int(label)].append(index)

    # The 40 % held-out half is cut into validation and test here, so this is the
    # second half of the 60/20/20 split and follows the fold with the first.
    from federated_outlier_adaptation.utils.seeding import configured_fold, fold_seed

    generator = random.Random(fold_seed(seed, configured_fold()))
    validation: list[int] = []
    test: list[int] = []
    for label in sorted(by_label):
        indices = list(by_label[label])
        generator.shuffle(indices)
        half = len(indices) // 2
        validation += indices[:half]
        test += indices[half:]
    return sorted(validation), sorted(test)


def _subset_loader(dataset, indices, batch_size):
    """Evaluation loader over a fixed subset; no shuffling, so it is stable."""
    if not indices:
        return None
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)


def _loader_dataset(loader):
    """The dataset behind an evaluation loader, or ``None``."""
    return None if loader is None else loader.dataset


class EmptyPopulationError(ValueError):
    """
    A run ended up with no participant at all.

    Nothing downstream checks for that: the pool evaluation loader is then built
    over an empty sample list, ``build_dataset`` returns ``None``, and the run
    dies several hundred lines later inside ``BaseTrainer.evaluate`` with
    ``TypeError: 'NoneType' object is not iterable``.  Raising here says what
    was asked for and what was available instead.
    """


class UnknownClientError(ValueError):
    """
    A run named client ids the dataset does not have.

    Raised when an explicit client list - ``foa extreme --clients ...`` - holds
    an id that is not among ``provider.all_client_ids()``: a writer of another
    dataset, a hard-coded id from an earlier setting, or a typo.
    """


class UntrainableClientError(ValueError):
    """
    A participant holds too little data to form a local training split.

    The per-client split is stratified and cuts *per label*
    (``int(n_label * train_rate)``), so a writer with a single sample in a class
    contributes nothing to that class's train and validation halves.  A writer
    with one sample in *every* class therefore has an empty train split and an
    empty validation split, and ``build_dataset`` returns ``None`` for both.
    That is not rare at 62 classes: the same writer that held ~18 samples per
    class in the ten-class setting holds ~1 here.

    Without this the run reached ``BaseTrainer.train(None, None, ...)`` and died
    with ``TypeError: 'NoneType' object is not iterable`` in the middle of a
    round, naming neither the client nor the reason.
    """


def untrainable_message(client_id, train_loader, eval_loader, provider=None) -> str:
    """The message of :class:`UntrainableClientError`, with the split sizes."""
    sizes = {
        name: (0 if loader is None else len(loader.dataset))
        for name, loader in (("train", train_loader), ("validation", eval_loader))
    }
    try:
        held = int(provider.sample_count(client_id)) if provider is not None else None
    except (AttributeError, KeyError, TypeError):  # pragma: no cover - exotic provider
        held = None
    return (
        f"Client {client_id!r} cannot be trained: its local split is empty "
        f"({sizes}, {held} sample(s) in total). A stratified 60/40 split cuts "
        "per label, so a writer holding one sample per class has neither a "
        "train nor a validation half. Exclude such writers from the pool - "
        "'foa select-outliers --require-trainable' does exactly that - or pick "
        "a client population that does not contain them."
    )


def _requested_ids(single_outlier: Any) -> list:
    """The restriction as a list of ids, preserving repeats; ``[]`` for none."""
    if not single_outlier or single_outlier == "None":
        return []
    if isinstance(single_outlier, str):
        return [single_outlier]
    return list(single_outlier)


def resolve_participants(provider, selected: Sequence[str], single_outlier: Any):
    """
    Split a client restriction into unique clients and participants.

    Two readings of a client restriction meet here, and which one applies
    depends on whether the requested ids are in the current selection:

    * **Restriction** (the published reading, unchanged).  Every requested id is
      in ``selected``, so the restriction narrows the run's selection to those
      clients.  The order is the selection's, and a repeated id yields that many
      participants over the same data.  This is the branch the published
      extreme cases take - their ids are members of the frozen five-writer
      selection - so their participant lists are bit for bit what they were.

    * **Explicit participant list.**  The requested ids are *not* (all) in the
      current selection.  An intersection would then be empty, which is how a
      run used to end up with no clients at all.  But naming ids explicitly is
      not a request to narrow a selection - it is a statement of who the
      participants are - so they are validated against the whole population
      (``provider.all_client_ids()``) and used as given, in the requested order.
      That makes ``--clients`` independent of whichever selection file happens
      to be current: a pool written by ``select-outliers``, the frozen
      ``selected_outliers.json``, or nothing at all.

    Returns:
        tuple[list[str], list[str]]: ``(unique_clients, participants)``.  The
        two lists differ only when the restriction repeats an id.

    Raises:
        UnknownClientError: An explicit id is not a client of this dataset.
        EmptyPopulationError: There is no participant to run with.
    """
    requested = _requested_ids(single_outlier)
    unique = provider.restrict(selected, single_outlier)

    if not requested:
        # No restriction: the selection is the population, as before.
        if not unique:
            raise EmptyPopulationError(
                "The run has no participants: its client selection is empty. "
                "Check --outliers-file / --pool-frac, or the selection file the "
                "provider resolves by default."
            )
        return unique, list(unique)

    if unique and set(unique) == set(requested):
        # Every requested id is in the selection - the published reading.
        if len(requested) != len(set(requested)):
            allowed = set(unique)
            return unique, [cid for cid in requested if cid in allowed]
        return unique, list(unique)

    # The requested ids are not all in the current selection, so they are taken
    # as the participant list itself and checked against the whole population.
    known = set(provider.all_client_ids())
    # A merged client ("w1+w2") is never in the population itself; it is valid
    # exactly when every writer it names is, so validation asks about members.
    from federated_outlier_adaptation.data.merged_clients import unknown_members

    missing = unknown_members(dict.fromkeys(requested), known)
    if missing:
        available = sorted(known)
        raise UnknownClientError(
            f"Unknown client id(s) {missing}: this dataset has "
            f"{len(available)} client(s)"
            + (f", such as {available[:5]}" if available else " (it has none)")
            + ". The extreme-case defaults in constants.EXTREME_CASE_CLIENTS "
            "are writers of the published 128x128 digit setting and do not "
            "exist in every dataset; pass 'foa extreme --clients ...' to name "
            "clients of this one."
        )
    return list(dict.fromkeys(requested)), list(requested)


class ClientPopulation:
    """
    The clients of one run: who exists, who trains this round, how they score.

    Args:
        provider: Dataset provider used to build the per-client loaders.
        selected: Selected client list of the run (before any restriction).
        single_outlier: Optional restriction, possibly with duplicates.
        participation: Fraction of the participants drawn per round.
        policy: Selection policy, see :mod:`.client_sampler`.
        sampler_seed: Seed of the sampler's random number generator.
        track_clients: Force per-client evaluation even for the default policy.
        loader_seed: Seed handed to ``provider.build_dataset`` for the
            per-client evaluation loaders.
    """

    def __init__(
        self,
        provider,
        selected: Sequence[str],
        single_outlier: Any = None,
        participation: float = 1.0,
        policy: str = "all",
        sampler_seed: Optional[int] = None,
        track_clients: bool = False,
        loader_seed: Optional[int] = None,
        clients_per_round: Optional[int] = None,
        insample_every: int = 1,
    ):
        self.provider = provider
        self.unique_clients, self.participants = resolve_participants(
            provider, selected, single_outlier
        )
        self.sampler = ClientSampler(
            self.participants,
            participation=participation,
            policy=policy,
            seed=sampler_seed,
            clients_per_round=clients_per_round,
        )
        self.track_clients = bool(track_clients)
        self.loader_seed = loader_seed
        self.insample_every = max(1, int(insample_every))
        #: Rounds in which the in-sample metrics were actually measured.
        self.insample_rounds: list[int] = []
        #: Cached evaluation sets of the pool; ``None`` keeps the loader path.
        self.cache = None

        self.client_accuracies: list[dict[str, float]] = []
        self.participants_per_round: list[list[str]] = []
        self.latest_accuracies: dict[str, float] = {}
        self.heldout_accuracies: list[Optional[float]] = []
        self.heldout_per_client: list[dict[str, float]] = []
        self.heldout_participant_accuracies: list[Optional[float]] = []
        self.pool_val_accuracies: list[Optional[float]] = []
        self.pool_test_accuracies: list[Optional[float]] = []
        self.client_val_accuracies: list[dict[str, float]] = []
        self.client_test_accuracies: list[dict[str, float]] = []
        self._loaders: dict[str, Any] = {}
        self._heldout_loaders: dict[str, Any] = {}
        self._split_loaders: dict[str, Any] = {}

    # ------------------------------------------------------------------- state
    @property
    def active(self) -> bool:
        """Whether this run deviates from the published every-client-every-round loop."""
        return self.sampler.is_active or self.track_clients

    @property
    def has_duplicates(self) -> bool:
        return len(self.participants) != len(set(self.participants))

    # ------------------------------------------------------------------ rounds
    def select(self, round_index: int) -> list[str]:
        """Participants of ``round_index``, recorded for the result payload."""
        participants = self.sampler.select(round_index, self.latest_accuracies)
        if self.active:
            self.participants_per_round.append(list(participants))
        return participants

    def client_loader(self, client_id: str, batch_size: int):
        """
        Evaluation loader over a client's own data.

        Built once per client and cached; ``train_rate=0`` / ``eval_rate=0``
        yields the whole split as the test loader, the same convention the
        all-clients evaluation loader uses.
        """
        loader = self._loaders.get(client_id)
        if loader is None:
            loader = self.provider.build_dataset(
                client_id,
                train_rate=0.0,
                eval_rate=0.0,
                batch_size=batch_size,
                loader_seed=self.loader_seed,
            )[2]
            self._loaders[client_id] = loader
        return loader

    # ------------------------------------------------------------------ cache
    def prepare_cache(self, cache, batch_size: int) -> dict[str, bool]:
        """
        Materialise the pool's evaluation sets into ``cache``.

        Three sets are registered, each as one concatenation carrying the
        per-client segment boundaries: the clients' own data (only when the run
        tracks clients), and the validation and test halves of their held-out
        split.  A set that cannot be materialised is simply absent, and the
        corresponding evaluation keeps its loaders, so a provider without a
        compact form loses nothing but the speed-up.

        Returns:
            dict[str, bool]: which sets were registered.
        """
        self.cache = cache
        registered: dict[str, bool] = {}
        if cache is None:
            return registered

        if self.active:
            registered[POOL_CLIENTS_SET] = cache.add_segments(
                POOL_CLIENTS_SET,
                [
                    (client_id, _loader_dataset(self.client_loader(client_id, batch_size)))
                    for client_id in self.unique_clients
                ],
            )
        for name, half in ((POOL_VAL_SET, 0), (POOL_TEST_SET, 1)):
            registered[name] = cache.add_segments(
                name,
                [
                    (client_id, _loader_dataset(self.split_loaders(client_id, batch_size)[half]))
                    for client_id in self.unique_clients
                ],
            )
        return registered

    def _cached(self, name: str) -> bool:
        return self.cache is not None and self.cache.has(name)

    def measures_insample(self, round_index: int, is_last: bool = False) -> bool:
        """
        Whether the in-sample metrics are measured in this round.

        ``insample_every=1`` (the default) measures every round, which is the
        published behaviour.  A larger value measures every ``N``-th round and
        the last one; the series keep their length and their alignment with
        ``accuracies``, with the entries of the skipped rounds repeating the
        most recent measurement.  ``insample_rounds`` records which rounds were
        measured.
        """
        if self.insample_every <= 1:
            return True
        return bool(is_last) or (int(round_index) % self.insample_every == 0)

    def track(
        self,
        evaluate: Callable[[Any], float],
        batch_size: int,
        measure: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate every pool client on its own data and store the accuracies.

        Args:
            evaluate: Callable taking a loader and returning an accuracy; the
                runner passes its trainer's ``evaluate`` with the current global
                model already loaded.  Unused when the run has a cached
                in-sample set, which produces every client's accuracy from one
                forward pass.
            batch_size: Batch size of the evaluation loaders.
            measure: ``False`` on a round that ``--insample-every`` skips; the
                series then repeats the most recent measurement so it stays
                aligned with ``accuracies``.

        Returns:
            dict[str, float]: ``{client: accuracy}`` of this round.
        """
        if not self.active:
            return {}
        if not measure:
            self.client_accuracies.append(dict(self.latest_accuracies))
            return dict(self.latest_accuracies)

        if self._cached(POOL_CLIENTS_SET):
            accuracies = self.cache.segment_accuracies(POOL_CLIENTS_SET)
        else:
            accuracies = {}
            for client_id in self.unique_clients:
                loader = self.client_loader(client_id, batch_size)
                if loader is None:  # pragma: no cover - client without data
                    continue
                accuracies[client_id] = float(evaluate(loader))
        self.latest_accuracies = accuracies
        self.client_accuracies.append(dict(accuracies))
        return accuracies

    # ----------------------------------------------------------------- held-out
    def heldout_loader(self, client_id: str, batch_size: int):
        """
        The client's 40% evaluation split, i.e. the part it does not train on.

        Built with the same ``train_rate=0.6`` / ``eval_rate=0.4`` and the same
        default split seed as the local training loaders, so the held-out
        samples are exactly the complement of the training samples and do not
        move between rounds.
        """
        loader = self._heldout_loaders.get(client_id)
        if loader is None:
            loader = self.provider.build_dataset(
                client_id,
                train_rate=0.6,
                eval_rate=0.4,
                batch_size=batch_size,
                loader_seed=self.loader_seed,
            )[1]
            self._heldout_loaders[client_id] = loader
        return loader

    def split_loaders(self, client_id: str, batch_size: int):
        """
        Deterministic validation/test halves of a client's held-out split.

        The 40% the client never trains on is divided once, stratified by label
        and seeded with :data:`SPLIT_SEED`, so model selection (validation) and
        the reported numbers (test) never share a sample.  Both loaders are
        built once per client and cached.

        When a fold book covers this client the book's own validation and test
        partitions are returned instead - see :func:`booked_split`.  The book
        made that division already, and it is the division every other process
        reading the same fold uses.

        Returns:
            tuple: ``(validation loader, test loader)``; either may be ``None``
            when the client has too few samples to split.
        """
        cached = self._split_loaders.get(client_id)
        if cached is not None:
            return cached
        booked = booked_split(self.provider, client_id, batch_size, self.loader_seed)
        if booked is not None:
            # The book already named the three parts; re-cutting them here
            # would put the model-selection rows into the reported number.
            self._split_loaders[client_id] = booked
            return booked
        heldout = self.heldout_loader(client_id, batch_size)
        if heldout is None:
            self._split_loaders[client_id] = (None, None)
            return None, None
        dataset = heldout.dataset
        val_index, test_index = stratified_halves(dataset, seed=SPLIT_SEED)
        pair = (
            _subset_loader(dataset, val_index, batch_size),
            _subset_loader(dataset, test_index, batch_size),
        )
        self._split_loaders[client_id] = pair
        return pair

    def evaluate_pool(
        self,
        evaluate: Callable[[Any], float],
        batch_size: int,
        participants: Sequence[str],
    ) -> dict[str, Any]:
        """
        Score the current global model on the pool's held-out data.

        Three pooled numbers are produced, all weighted by sample count so they
        are accuracies over a union of splits rather than means of per-client
        accuracies:

        - ``pool_val_acc``  - the validation halves, for model selection,
        - ``pool_test_acc`` - the test halves, for the reported numbers,
        - ``heldout_client_accuracies`` - both halves together, i.e. the whole
          40% the clients never train on.

        The headline numbers always cover the **whole pool**, so the adaptation
        curve is comparable no matter how many clients trained in a round; when
        the sampler is active the same quantity restricted to the round's
        participants is recorded alongside.

        Returns:
            dict: the pooled numbers of this round.
        """
        per_val: dict[str, float] = {}
        per_test: dict[str, float] = {}
        counts: dict[str, tuple[int, int]] = {}
        val_correct = val_total = 0.0
        test_correct = test_total = 0.0

        cached_val = self.cache.segment_accuracies(POOL_VAL_SET) if self._cached(POOL_VAL_SET) else None
        cached_test = (
            self.cache.segment_accuracies(POOL_TEST_SET) if self._cached(POOL_TEST_SET) else None
        )

        for client_id in self.unique_clients:
            val_loader, test_loader = self.split_loaders(client_id, batch_size)
            val_count = len(val_loader.dataset) if val_loader is not None else 0
            test_count = len(test_loader.dataset) if test_loader is not None else 0
            counts[client_id] = (val_count, test_count)
            if val_count:
                accuracy = (
                    cached_val[client_id]
                    if cached_val is not None
                    else float(evaluate(val_loader))
                )
                per_val[client_id] = accuracy
                val_correct += accuracy * val_count
                val_total += val_count
            if test_count:
                accuracy = (
                    cached_test[client_id]
                    if cached_test is not None
                    else float(evaluate(test_loader))
                )
                per_test[client_id] = accuracy
                test_correct += accuracy * test_count
                test_total += test_count

        pool_val = (val_correct / val_total) if val_total else None
        pool_test = (test_correct / test_total) if test_total else None
        combined_total = val_total + test_total
        pooled = ((val_correct + test_correct) / combined_total) if combined_total else None

        self.heldout_accuracies.append(pooled)
        self.pool_val_accuracies.append(pool_val)
        self.pool_test_accuracies.append(pool_test)

        if self.active:
            per_client = {
                cid: (
                    (per_val.get(cid, 0.0) * counts[cid][0]
                     + per_test.get(cid, 0.0) * counts[cid][1])
                    / (counts[cid][0] + counts[cid][1])
                )
                for cid in counts
                if counts[cid][0] + counts[cid][1]
            }
            self.heldout_per_client.append(per_client)
            self.client_val_accuracies.append(dict(per_val))
            self.client_test_accuracies.append(dict(per_test))

            selected = list(dict.fromkeys(participants))
            weighted, total = 0.0, 0
            for cid in selected:
                val_count, test_count = counts.get(cid, (0, 0))
                weighted += per_val.get(cid, 0.0) * val_count
                weighted += per_test.get(cid, 0.0) * test_count
                total += val_count + test_count
            self.heldout_participant_accuracies.append(
                (weighted / total) if total else None
            )

        return {
            "heldout": pooled,
            "pool_val_acc": pool_val,
            "pool_test_acc": pool_test,
        }

    # ------------------------------------------------------------------ report
    def info(self) -> dict[str, Any]:
        """
        Additive payload keys describing the population.

        The three pooled evaluation series are always reported; the per-client
        and selection keys only appear when the run actually deviates from the
        published every-client-every-round loop, so a default result file gains
        the evaluation series and nothing else.
        """
        info: dict[str, Any] = {
            "heldout_client_accuracies": list(self.heldout_accuracies),
            "pool_val_accuracies": list(self.pool_val_accuracies),
            "pool_test_accuracies": list(self.pool_test_accuracies),
        }
        if self.insample_every > 1:
            # Only a run that asked to thin the in-sample metrics reports this,
            # so a default result file gains no key at all.
            info["insample_every"] = self.insample_every
            info["insample_rounds"] = list(self.insample_rounds)
        if not self.active:
            return info
        info.update(
            {
                "client_accuracies": [dict(entry) for entry in self.client_accuracies],
                "client_heldout_accuracies": [dict(e) for e in self.heldout_per_client],
                "heldout_participant_accuracies": list(self.heldout_participant_accuracies),
                "client_val_accuracies": [dict(e) for e in self.client_val_accuracies],
                "client_test_accuracies": [dict(e) for e in self.client_test_accuracies],
                "participants": [list(entry) for entry in self.participants_per_round],
                "participation": self.sampler.participation,
                "policy": self.sampler.policy,
                "sampler_seed": self.sampler.seed,
                "pool_size": len(self.participants),
            }
        )
        return info
