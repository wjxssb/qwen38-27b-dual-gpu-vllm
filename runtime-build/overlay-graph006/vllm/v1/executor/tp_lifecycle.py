# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation fencing primitives for tensor-parallel request lifecycles.

This module intentionally owns only lifecycle metadata.  It never frees a
transport, CUDA, or model-state resource itself: callers must prove a
request-owned resource quiescent before the registry can retire its record.
That makes a failed proof fail closed instead of reusing potentially live TP
resources.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def is_enabled() -> bool:
    """Return whether integration should enable generation lifecycle fencing.

    Keeping the core importable without configuration lets the frozen runtime
    retain its existing behavior until the independent stability campaign has
    completed its no-GPU and fault-injection gates.
    """
    return os.getenv("VLLM_TP_GENERATION_LIFECYCLE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class GenerationLifecycleError(RuntimeError):
    """Raised when a caller attempts an unsafe lifecycle transition."""


class TPGroupPoisoned(GenerationLifecycleError):
    """Raised when a caller tries to use a TP group marked unsafe to reuse."""


LifecycleError = GenerationLifecycleError


class OwnershipError(GenerationLifecycleError):
    """Raised when request ownership cannot be proven."""


class GenerationPhase(str, Enum):
    """A TP request generation's legal lifecycle phases."""

    IDLE = "idle"
    BEGIN = "begin"
    RUNNING = "running"
    COMPLETING = "completing"
    ABORTING = "aborting"
    QUIESCING = "quiescing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    BASELINE_VERIFY = "baseline_verify"
    CLEAN_COMPLETE = "clean_complete"
    POISONED = "poisoned"


class ResourceType(str, Enum):
    """Request-scoped resource types whose ownership can be established."""

    RPC = "rpc"
    FUTURE = "future"
    SHM_SLOT = "shm_slot"
    SHM_MESSAGE = "shm_message"
    REQUEST_STATE = "request_state"
    MTP_REQUEST_STATE = "mtp_request_state"
    GDN_REQUEST_STATE = "gdn_request_state"
    CUDA_EVENT_HANDLE = "cuda_event_handle"
    OTHER_REQUEST_SCOPED = "other_request_scoped"


class ResourceState(str, Enum):
    """Metadata state, not the underlying resource's implementation state."""

    ACTIVE = "active"
    QUIESCENT = "quiescent"
    RELEASED = "released"


class WatchdogClassification(str, Enum):
    """Progress-aware watchdog outcomes."""

    PROGRESSING = "progressing"
    SLOW_PROGRESS = "slow_progress"
    NO_PROGRESS = "no_progress"
    RANK_SKEW = "rank_skew"


class RecoveryState(str, Enum):
    """Availability state for a complete TP executor process group."""

    READY = "ready"
    POISONED = "poisoned"
    RECOVERING = "recovering"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GenerationToken:
    """Identifies a request generation and the TP executor serving it."""

    generation: int
    request_id: str
    executor_identity: str


@dataclass
class ResourceRecord:
    """Bounded metadata for a resource proven to belong to one request."""

    generation: int
    request_id: str
    rank: int
    resource_type: ResourceType
    resource_id: str
    state: ResourceState
    created_timestamp: float
    last_transition_timestamp: float


@dataclass(frozen=True)
class OwnershipSummary:
    """Compact counts for one generation's provably owned metadata."""

    generation: int
    rank: int | None
    active_count: int
    quiescent_count: int
    released_count: int


@dataclass(frozen=True)
class RankFenceState:
    """A rank's compact quiescence proof at a lifecycle boundary."""

    rank: int
    executor_identity: str
    last_completed_generation: int
    active_generation: int | None = None
    active_request_count: int = 0
    active_generation_rpc_count: int = 0
    unresolved_generation_future_count: int = 0
    owned_generation_shm_count: int = 0
    known_request_owned_state_count: int = 0
    worker_alive: bool = True
    poisoned: bool = False
    collective_sequence: int | None = None
    candidate_swap_bytes: int = 0
    persistent_baseline_violations: tuple[str, ...] = ()

    def quiescence_failures(self, generation: int) -> tuple[str, ...]:
        """Return the fields that prevent safe reuse after ``generation``."""
        failures: list[str] = []
        if self.active_generation is not None:
            failures.append("active_generation")
        if self.active_request_count:
            failures.append("active_request_count")
        if self.active_generation_rpc_count:
            failures.append("active_generation_rpc_count")
        if self.unresolved_generation_future_count:
            failures.append("unresolved_generation_future_count")
        if self.owned_generation_shm_count:
            failures.append("owned_generation_shm_count")
        if self.known_request_owned_state_count:
            failures.append("known_request_owned_state_count")
        if not self.worker_alive:
            failures.append("worker_not_alive")
        if self.poisoned:
            failures.append("rank_poisoned")
        if self.candidate_swap_bytes:
            failures.append("candidate_swap")
        failures.extend(
            f"persistent_baseline_{violation}"
            for violation in self.persistent_baseline_violations
        )
        if self.last_completed_generation > generation:
            failures.append("completed_generation_ahead")
        return tuple(failures)


@dataclass(frozen=True)
class LifecycleEvent:
    """A compact, bounded diagnostic event with correlation keys."""

    timestamp: float
    event: str
    generation: int | None = None
    request_id: str | None = None
    rank: int | None = None
    rpc_id: str | None = None
    phase: GenerationPhase | None = None
    computed_tokens: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


class BoundedEventWindow:
    """A fixed-size event window that never blocks an inference caller."""

    def __init__(self, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("event capacity must be positive")
        self._events: deque[LifecycleEvent] = deque(maxlen=capacity)
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Return events evicted after the fixed capacity was reached."""
        return self._dropped

    def append(self, event: LifecycleEvent) -> None:
        """Append an event without I/O or unbounded allocation."""
        if len(self._events) == self._events.maxlen:
            self._dropped += 1
        self._events.append(event)

    def dump(self) -> tuple[LifecycleEvent, ...]:
        """Return a stable diagnostic snapshot."""
        return tuple(self._events)


class GenerationOwnershipRegistry:
    """Tracks only proven request-owned metadata for bounded generations.

    A long request can legitimately issue more RPCs than there are detailed
    records available.  Once a record has been proven quiescent, retaining its
    individual identifier until the generation-wide commit provides no further
    liveness protection.  Such records are therefore compacted into per-rank,
    per-resource-type counters and a fixed diagnostic tail.  They are still
    logical ownership until commit; compaction never changes an underlying
    transport, future, or device resource.
    """

    def __init__(
        self,
        capacity_per_generation: int = 1024,
        diagnostic_history_per_generation: int = 64,
    ) -> None:
        if capacity_per_generation <= 0:
            raise ValueError("registry capacity must be positive")
        if diagnostic_history_per_generation <= 0:
            raise ValueError("diagnostic history capacity must be positive")
        self._capacity_per_generation = capacity_per_generation
        self._records: dict[tuple[int, int, ResourceType, str], ResourceRecord] = {}
        self._diagnostic_history_per_generation = diagnostic_history_per_generation
        self._compacted_quiescent: dict[tuple[int, int, ResourceType], int] = {}
        self._compacted_released: dict[tuple[int, int, ResourceType], int] = {}
        self._compacted_history: dict[int, deque[ResourceRecord]] = {}

    @staticmethod
    def _key(
        generation: int,
        rank: int,
        resource_type: ResourceType,
        resource_id: str,
    ) -> tuple[int, int, ResourceType, str]:
        return generation, rank, resource_type, resource_id

    def register(
        self,
        *,
        generation: int,
        request_id: str,
        rank: int,
        resource_type: ResourceType,
        resource_id: str,
        timestamp: float | None = None,
    ) -> ResourceRecord:
        """Register an explicitly request-owned resource.

        Duplicate registrations are idempotent only if every ownership field
        agrees.  There is deliberately no API for global or persistent model
        resources.
        """
        if not isinstance(resource_type, ResourceType):
            raise OwnershipError(
                "only known request-scoped resource types may register"
            )
        if generation <= 0 or not request_id or not resource_id or rank < 0:
            raise OwnershipError(
                "resource ownership requires a valid generation and id"
            )
        key = self._key(generation, rank, resource_type, resource_id)
        existing = self._records.get(key)
        if existing is not None:
            if existing.request_id != request_id:
                raise OwnershipError("resource id already belongs to another request")
            return existing
        if self._detailed_count(generation) >= self._capacity_per_generation:
            raise OwnershipError("per-generation ownership registry is full")
        now = time.monotonic() if timestamp is None else timestamp
        record = ResourceRecord(
            generation=generation,
            request_id=request_id,
            rank=rank,
            resource_type=resource_type,
            resource_id=resource_id,
            state=ResourceState.ACTIVE,
            created_timestamp=now,
            last_transition_timestamp=now,
        )
        self._records[key] = record
        return record

    def mark_quiescent(
        self,
        *,
        generation: int,
        rank: int,
        resource_type: ResourceType,
        resource_id: str,
        timestamp: float | None = None,
    ) -> bool:
        """Mark a resource quiescent after its real owner has proved it safe."""
        record = self._get(generation, rank, resource_type, resource_id)
        if record.state is ResourceState.RELEASED:
            return False
        if record.state is ResourceState.QUIESCENT:
            return False
        record.state = ResourceState.QUIESCENT
        record.last_transition_timestamp = (
            time.monotonic() if timestamp is None else timestamp
        )
        return True

    def compact_quiescent(
        self,
        generation: int,
        *,
        rank: int | None = None,
    ) -> int:
        """Compact already-quiescent logical metadata before TP-wide commit.

        The caller must have completed the real resource transition first.
        Active records are deliberately untouched.  The aggregate and bounded
        diagnostic history remain part of the generation until
        :meth:`release_quiescent` is called by the TP-wide commit.
        """
        compacted_keys = [
            key
            for key, record in self._records.items()
            if record.generation == generation
            and (rank is None or record.rank == rank)
            and record.state is ResourceState.QUIESCENT
        ]
        if not compacted_keys:
            return 0
        history = self._compacted_history.setdefault(
            generation,
            deque(maxlen=self._diagnostic_history_per_generation),
        )
        for key in compacted_keys:
            record = self._records.pop(key)
            aggregate_key = (record.generation, record.rank, record.resource_type)
            self._compacted_quiescent[aggregate_key] = (
                self._compacted_quiescent.get(aggregate_key, 0) + 1
            )
            history.append(
                ResourceRecord(
                    generation=record.generation,
                    request_id=record.request_id,
                    rank=record.rank,
                    resource_type=record.resource_type,
                    resource_id=record.resource_id,
                    state=record.state,
                    created_timestamp=record.created_timestamp,
                    last_transition_timestamp=record.last_transition_timestamp,
                )
            )
        return len(compacted_keys)

    def transition(
        self,
        *,
        generation: int,
        rank: int,
        resource_type: ResourceType,
        resource_id: str,
        state: ResourceState,
        timestamp: float | None = None,
    ) -> bool:
        """Apply a safe ownership transition without force-releasing records."""
        if state is ResourceState.QUIESCENT:
            return self.mark_quiescent(
                generation=generation,
                rank=rank,
                resource_type=resource_type,
                resource_id=resource_id,
                timestamp=timestamp,
            )
        record = self._get(generation, rank, resource_type, resource_id)
        if state is ResourceState.ACTIVE and record.state is ResourceState.ACTIVE:
            return False
        raise LifecycleError(
            "only a TP-wide commit may transition quiescent records to released"
        )

    def release_quiescent(self, generation: int, timestamp: float | None = None) -> int:
        """Retire metadata only for resources already proven quiescent.

        This intentionally raises instead of force-releasing active records.
        """
        active = self.count(generation=generation, state=ResourceState.ACTIVE)
        if active:
            raise LifecycleError(
                f"generation {generation} has {active} active owned resources"
            )
        now = time.monotonic() if timestamp is None else timestamp
        released = 0
        for record in self._records.values():
            if record.generation != generation:
                continue
            if record.state is ResourceState.QUIESCENT:
                record.state = ResourceState.RELEASED
                record.last_transition_timestamp = now
                released += 1
        compacted_keys = [
            key for key in self._compacted_quiescent if key[0] == generation
        ]
        for key in compacted_keys:
            count = self._compacted_quiescent.pop(key)
            self._compacted_released[key] = self._compacted_released.get(key, 0) + count
            released += count
        return released

    def retire_released(self, generation: int) -> int:
        """Forget released lifecycle metadata after successful commit."""
        stale_keys = [
            key
            for key, record in self._records.items()
            if record.generation == generation
            and record.state is ResourceState.RELEASED
        ]
        for key in stale_keys:
            del self._records[key]
        compacted_keys = [
            key for key in self._compacted_released if key[0] == generation
        ]
        compacted_retired = sum(
            self._compacted_released.pop(key) for key in compacted_keys
        )
        self._compacted_history.pop(generation, None)
        return len(stale_keys) + compacted_retired

    def retire(self, generation: int) -> int:
        """Integration-friendly alias for retiring committed metadata."""
        return self.retire_released(generation)

    def summary(self, generation: int, rank: int | None = None) -> OwnershipSummary:
        """Return compact ownership counts without exposing resource contents."""
        return OwnershipSummary(
            generation=generation,
            rank=rank,
            active_count=self.count(
                generation=generation, rank=rank, state=ResourceState.ACTIVE
            ),
            quiescent_count=self.count(
                generation=generation,
                rank=rank,
                state=ResourceState.QUIESCENT,
            ),
            released_count=self.count(
                generation=generation,
                rank=rank,
                state=ResourceState.RELEASED,
                include_released=True,
            ),
        )

    def count(
        self,
        *,
        generation: int,
        rank: int | None = None,
        state: ResourceState | None = None,
        include_released: bool = False,
    ) -> int:
        """Count logical records, including compacted quiescent metadata."""
        detailed_count = sum(
            record.generation == generation
            and (rank is None or record.rank == rank)
            and (state is None or record.state is state)
            and (include_released or record.state is not ResourceState.RELEASED)
            for record in self._records.values()
        )

        def aggregate_count(
            aggregate: Mapping[tuple[int, int, ResourceType], int],
        ) -> int:
            return sum(
                value
                for (record_generation, record_rank, _), value in aggregate.items()
                if record_generation == generation
                and (rank is None or record_rank == rank)
            )

        if state is ResourceState.ACTIVE:
            return detailed_count
        if state is ResourceState.QUIESCENT:
            return detailed_count + aggregate_count(self._compacted_quiescent)
        if state is ResourceState.RELEASED:
            return detailed_count + (
                aggregate_count(self._compacted_released) if include_released else 0
            )
        return (
            detailed_count
            + aggregate_count(self._compacted_quiescent)
            + (aggregate_count(self._compacted_released) if include_released else 0)
        )

    def records(self, generation: int) -> tuple[ResourceRecord, ...]:
        """Return the detailed (not compacted) ownership snapshot."""
        return tuple(
            record
            for record in self._records.values()
            if record.generation == generation
        )

    def compacted_diagnostics(self, generation: int) -> tuple[ResourceRecord, ...]:
        """Return the bounded tail of quiescent records retained for diagnosis."""
        return tuple(self._compacted_history.get(generation, ()))

    def _detailed_count(self, generation: int) -> int:
        """Count only records that consume the fixed detailed-record budget."""
        return sum(record.generation == generation for record in self._records.values())

    def _get(
        self,
        generation: int,
        rank: int,
        resource_type: ResourceType,
        resource_id: str,
    ) -> ResourceRecord:
        try:
            return self._records[
                self._key(generation, rank, resource_type, resource_id)
            ]
        except KeyError as exc:
            raise OwnershipError("resource ownership was not registered") from exc


OwnershipRegistry = GenerationOwnershipRegistry


class RankFenceValidator:
    """Validates a compact, TP-wide quiescence proof before reuse."""

    @staticmethod
    def failures(
        *,
        ranks: Iterable[int],
        executor_identity: str,
        expected_completed_generation: int,
        rank_states: Mapping[int, RankFenceState],
    ) -> list[str]:
        """Return all mismatches that must block the next request."""
        expected_ranks = frozenset(ranks)
        failures: list[str] = []
        if set(rank_states) != expected_ranks:
            failures.append("missing_or_unexpected_ranks")
        for rank in sorted(expected_ranks):
            state = rank_states.get(rank)
            if state is None:
                continue
            if state.rank != rank:
                failures.append(f"rank_{rank}_identity_mismatch")
            if state.executor_identity != executor_identity:
                failures.append(f"rank_{rank}_executor_identity")
            if state.last_completed_generation != expected_completed_generation:
                failures.append(f"rank_{rank}_completed_generation")
            for failure in state.quiescence_failures(expected_completed_generation):
                failures.append(f"rank_{rank}_{failure}")
        # A collective sequence is optional because not every runtime exposes
        # one safely.  When every TP rank does expose it, disagreement is a
        # concrete proof that this process group must not be reused.
        collective_sequences = [
            rank_states[rank].collective_sequence
            for rank in expected_ranks
            if rank in rank_states and rank_states[rank].collective_sequence is not None
        ]
        if (
            len(collective_sequences) == len(expected_ranks)
            and len(set(collective_sequences)) != 1
        ):
            failures.append("collective_sequence_mismatch")
        return failures


@dataclass(frozen=True)
class ResponseEnvelope:
    """The minimum correlation metadata required to fence an async response."""

    generation: int
    request_id: str
    executor_identity: str
    rpc_id: str


@dataclass(frozen=True)
class WatchdogSample:
    """One rank's bounded progress report."""

    generation: int
    rank: int
    phase: GenerationPhase
    computed_tokens: int
    timestamp: float
    rpc_id: str | None = None


@dataclass(frozen=True)
class WatchdogAssessment:
    """Progress classification and its bounded evidence window."""

    classification: WatchdogClassification
    generation: int
    reason: str
    events: tuple[LifecycleEvent, ...]


class ProgressWatchdog:
    """Classifies no-progress separately from legitimate slow progress."""

    def __init__(
        self,
        ranks: Iterable[int],
        *,
        diagnostic_interval_seconds: float,
        event_capacity: int = 256,
    ) -> None:
        self._ranks = frozenset(ranks)
        if not self._ranks:
            raise ValueError("watchdog needs at least one rank")
        if diagnostic_interval_seconds <= 0:
            raise ValueError("diagnostic interval must be positive")
        self._interval = diagnostic_interval_seconds
        self._samples: dict[tuple[int, int], WatchdogSample] = {}
        self._last_progress: dict[tuple[int, int], float] = {}
        self._events = BoundedEventWindow(event_capacity)

    @property
    def dropped_events(self) -> int:
        """Return bounded diagnostic-event eviction count."""
        return self._events.dropped

    def observe(
        self,
        *,
        generation: int,
        rank: int,
        phase: GenerationPhase,
        computed_tokens: int,
        timestamp: float,
        rpc_id: str | None = None,
    ) -> None:
        """Record a compact heartbeat without changing timeout policy."""
        if rank not in self._ranks:
            raise LifecycleError(f"unexpected TP rank {rank}")
        if computed_tokens < 0:
            raise LifecycleError("computed token count cannot be negative")
        key = generation, rank
        previous = self._samples.get(key)
        made_progress = (
            previous is None
            or computed_tokens > previous.computed_tokens
            or phase is not previous.phase
        )
        sample = WatchdogSample(
            generation=generation,
            rank=rank,
            phase=phase,
            computed_tokens=computed_tokens,
            timestamp=timestamp,
            rpc_id=rpc_id,
        )
        self._samples[key] = sample
        if made_progress:
            self._last_progress[key] = timestamp
        self._events.append(
            LifecycleEvent(
                timestamp=timestamp,
                event="progress",
                generation=generation,
                rank=rank,
                rpc_id=rpc_id,
                phase=phase,
                computed_tokens=computed_tokens,
            )
        )

    def assess(self, generation: int, timestamp: float) -> WatchdogAssessment:
        """Return a diagnostic classification; it never extends a deadline."""
        samples = {rank: self._samples.get((generation, rank)) for rank in self._ranks}
        missing = [rank for rank, sample in samples.items() if sample is None]
        if missing:
            return self._assessment(
                WatchdogClassification.NO_PROGRESS,
                generation,
                f"missing heartbeat from ranks {sorted(missing)}",
            )

        assert all(sample is not None for sample in samples.values())
        progress_ages = {
            rank: timestamp - self._last_progress[(generation, rank)]
            for rank in self._ranks
        }
        recent_progress = {
            rank for rank, age in progress_ages.items() if age <= self._interval
        }
        if not recent_progress:
            return self._assessment(
                WatchdogClassification.NO_PROGRESS,
                generation,
                "no rank made progress during the diagnostic interval",
            )
        if recent_progress != self._ranks:
            return self._assessment(
                WatchdogClassification.RANK_SKEW,
                generation,
                "some TP ranks progressed while others stalled",
            )

        token_counts = [sample.computed_tokens for sample in samples.values() if sample]
        if max(token_counts) != min(token_counts):
            return self._assessment(
                WatchdogClassification.RANK_SKEW,
                generation,
                "TP ranks report divergent computed-token progress",
            )
        sample_ages = [
            timestamp - sample.timestamp for sample in samples.values() if sample
        ]
        classification = (
            WatchdogClassification.PROGRESSING
            if max(sample_ages) <= self._interval / 2
            else WatchdogClassification.SLOW_PROGRESS
        )
        return self._assessment(
            classification, generation, "TP progress is still moving"
        )

    def retire_completed_generations(self, generation: int) -> None:
        """Discard per-generation samples after a clean lifecycle commit.

        The event window is already bounded, but these lookup maps are keyed
        by generation.  Keeping completed generations would make a healthy
        long-running service accumulate diagnostic metadata indefinitely.
        This removes only diagnostic samples; it never changes a transport,
        future, or model resource.
        """
        self._samples = {
            key: sample for key, sample in self._samples.items() if key[0] > generation
        }
        self._last_progress = {
            key: timestamp
            for key, timestamp in self._last_progress.items()
            if key[0] > generation
        }

    def _assessment(
        self,
        classification: WatchdogClassification,
        generation: int,
        reason: str,
    ) -> WatchdogAssessment:
        return WatchdogAssessment(
            classification=classification,
            generation=generation,
            reason=reason,
            events=self._events.dump(),
        )


class TPGenerationLifecycle:
    """Coordinates a fail-closed request lifecycle for a fixed TP group."""

    def __init__(
        self,
        *,
        ranks: Iterable[int],
        executor_identity: str,
        registry_capacity: int = 1024,
        event_capacity: int = 256,
    ) -> None:
        self._ranks = frozenset(ranks)
        if not self._ranks:
            raise ValueError("TP lifecycle needs at least one rank")
        if not executor_identity:
            raise ValueError("executor identity is required")
        self.executor_identity = executor_identity
        self.registry = GenerationOwnershipRegistry(registry_capacity)
        self.events = BoundedEventWindow(event_capacity)
        self.phase = GenerationPhase.IDLE
        self.active: GenerationToken | None = None
        self.last_completed_generation = 0
        self._prepared: dict[int, RankFenceState] = {}
        self._poison_reason: str | None = None

    @property
    def poison_reason(self) -> str | None:
        """Return why this process group became unavailable, if it did."""
        return self._poison_reason

    @property
    def can_accept_requests(self) -> bool:
        """Return whether a new generation may safely begin."""
        return self.phase in {GenerationPhase.IDLE, GenerationPhase.CLEAN_COMPLETE}

    def begin_generation(
        self, request_id: str, timestamp: float | None = None
    ) -> GenerationToken:
        """Allocate exactly one new monotonic generation for ``request_id``."""
        if not self.can_accept_requests:
            raise LifecycleError(f"cannot begin while lifecycle is {self.phase.value}")
        if not request_id:
            raise LifecycleError("request id is required")
        generation = self.last_completed_generation + 1
        self.active = GenerationToken(generation, request_id, self.executor_identity)
        self.phase = GenerationPhase.BEGIN
        self._prepared.clear()
        self._event("generation_begin", timestamp=timestamp)
        return self.active

    def mark_running(self, generation: int, timestamp: float | None = None) -> None:
        """Mark the active generation running after request dispatch begins."""
        self._require_active(generation, GenerationPhase.BEGIN)
        self.phase = GenerationPhase.RUNNING
        self._event("generation_running", timestamp=timestamp)

    def mark_completing(self, generation: int, timestamp: float | None = None) -> None:
        """Begin normal completion; quiescence still must be proven."""
        self._require_active(generation, GenerationPhase.RUNNING)
        self.phase = GenerationPhase.COMPLETING
        self._event("generation_completing", timestamp=timestamp)

    def mark_aborting(self, generation: int, timestamp: float | None = None) -> None:
        """Begin abort handling without releasing unknown resources."""
        self._require_active(generation, GenerationPhase.RUNNING)
        self.phase = GenerationPhase.ABORTING
        self._event("generation_aborting", timestamp=timestamp)

    def begin_quiesce(self, generation: int, timestamp: float | None = None) -> None:
        """Stop new work for a completed or aborted generation."""
        self._require_active(
            generation, GenerationPhase.COMPLETING, GenerationPhase.ABORTING
        )
        self.phase = GenerationPhase.QUIESCING
        self._event("generation_quiescing", timestamp=timestamp)

    def prepare_rank(
        self,
        generation: int,
        state: RankFenceState,
        timestamp: float | None = None,
    ) -> bool:
        """Record one rank's quiescence proof for phase one cleanup.

        A duplicate matching prepare is harmless.  A bad proof poisons the
        group because allowing the next request to reuse the TP group would be
        unsafe.
        """
        self._require_active(
            generation, GenerationPhase.QUIESCING, GenerationPhase.PREPARED
        )
        if state.rank not in self._ranks:
            return self._poison("prepare from unknown rank", timestamp)
        if state.executor_identity != self.executor_identity:
            return self._poison("prepare executor identity mismatch", timestamp)
        if state.quiescence_failures(generation):
            failures = ",".join(state.quiescence_failures(generation))
            return self._poison(
                f"rank {state.rank} not quiescent: {failures}",
                timestamp,
            )
        if self.registry.count(
            generation=generation, rank=state.rank, state=ResourceState.ACTIVE
        ):
            return self._poison(
                f"rank {state.rank} still owns lifecycle resources", timestamp
            )
        existing = self._prepared.get(state.rank)
        if existing is not None and existing != state:
            return self._poison(
                f"rank {state.rank} sent conflicting prepare", timestamp
            )
        self._prepared[state.rank] = state
        if len(self._prepared) == len(self._ranks):
            self.phase = GenerationPhase.PREPARED
            self._event("all_ranks_prepared", timestamp=timestamp)
        else:
            self._event("rank_prepared", rank=state.rank, timestamp=timestamp)
        return True

    def commit_cleanup(self, generation: int, timestamp: float | None = None) -> bool:
        """Commit only unanimously prepared, already-quiescent ownership.

        Repeated commit calls after the first one are intentionally no-ops.
        """
        self._require_active(
            generation,
            GenerationPhase.PREPARED,
            GenerationPhase.BASELINE_VERIFY,
        )
        if self.phase is GenerationPhase.BASELINE_VERIFY:
            return True
        if set(self._prepared) != self._ranks:
            return self._poison("commit without every rank prepared", timestamp)
        self.phase = GenerationPhase.COMMITTING
        try:
            released = self.registry.release_quiescent(generation, timestamp)
            retired = self.registry.retire_released(generation)
        except LifecycleError as exc:
            return self._poison(f"commit could not prove quiescence: {exc}", timestamp)
        self.phase = GenerationPhase.BASELINE_VERIFY
        self._event(
            "cleanup_committed",
            timestamp=timestamp,
            details={"released": released, "retired": retired},
        )
        return True

    def verify_post_request_baseline(
        self,
        generation: int,
        rank_states: Mapping[int, RankFenceState],
        timestamp: float | None = None,
    ) -> bool:
        """Require a TP-wide clean baseline before advancing generation."""
        self._require_active(generation, GenerationPhase.BASELINE_VERIFY)
        failures = self.rank_fence_failures(rank_states, generation)
        if self.registry.count(generation=generation, include_released=True):
            failures.append("registry_not_empty")
        if failures:
            return self._poison(
                f"post-request baseline failed: {','.join(failures)}", timestamp
            )
        self.last_completed_generation = generation
        self.active = None
        self.phase = GenerationPhase.CLEAN_COMPLETE
        self._event("post_request_baseline_clean", timestamp=timestamp)
        return True

    def rank_fence_failures(
        self,
        rank_states: Mapping[int, RankFenceState],
        generation: int,
    ) -> list[str]:
        """Return fence discrepancies that block the following generation."""
        return RankFenceValidator.failures(
            ranks=self._ranks,
            executor_identity=self.executor_identity,
            expected_completed_generation=generation,
            rank_states=rank_states,
        )

    def accept_response(
        self,
        response: ResponseEnvelope,
        timestamp: float | None = None,
    ) -> bool:
        """Fence late replies so they cannot mutate a later request's state."""
        active = self.active
        is_current = (
            active is not None
            and self.phase
            in {
                GenerationPhase.BEGIN,
                GenerationPhase.RUNNING,
                GenerationPhase.COMPLETING,
                GenerationPhase.ABORTING,
            }
            and response.generation == active.generation
            and response.request_id == active.request_id
            and response.executor_identity == active.executor_identity
        )
        if is_current:
            self._event(
                "response_accepted", rpc_id=response.rpc_id, timestamp=timestamp
            )
            return True
        self.record_stale_response(
            generation=response.generation,
            request_id=response.request_id,
            rpc_id=response.rpc_id,
            timestamp=timestamp,
            reason="response generation is no longer active",
        )
        return False

    def record_stale_response(
        self,
        *,
        generation: int | None,
        request_id: str | None,
        rpc_id: str | None,
        timestamp: float | None = None,
        rank: int | None = None,
        executor_identity: str | None = None,
        reason: str,
    ) -> None:
        """Record a rejected response without treating it as accepted.

        Transport-side validation knows more than :meth:`accept_response`:
        notably the expected RPC method and response queue rank.  It must be
        able to preserve that evidence without routing a same-generation but
        otherwise mismatched response through the accepted-response path.
        """
        active = self.active
        self._event(
            "stale_response",
            generation=generation,
            request_id=request_id,
            rank=rank,
            rpc_id=rpc_id,
            timestamp=timestamp,
            details={
                "reason": reason,
                "active_generation": active.generation if active else None,
                "active_request_id": active.request_id if active else None,
                "observed_executor_identity": executor_identity,
            },
        )

    def poison(self, reason: str, timestamp: float | None = None) -> None:
        """Fail closed without freeing resources whose liveness is unknown."""
        self._poison(reason, timestamp)

    def diagnostic_dump(self) -> tuple[LifecycleEvent, ...]:
        """Return only the bounded lifecycle metadata diagnostic window."""
        return self.events.dump()

    def _poison(self, reason: str, timestamp: float | None) -> bool:
        self._poison_reason = reason
        self.phase = GenerationPhase.POISONED
        self._event(
            "tp_group_poisoned", timestamp=timestamp, details={"reason": reason}
        )
        return False

    def _require_active(self, generation: int, *phases: GenerationPhase) -> None:
        if self.active is None or self.active.generation != generation:
            raise LifecycleError(f"generation {generation} is not active")
        if self.phase not in phases:
            expected = ", ".join(phase.value for phase in phases)
            raise LifecycleError(
                f"generation {generation} is {self.phase.value}, expected {expected}"
            )

    def _event(
        self,
        event: str,
        *,
        timestamp: float | None = None,
        generation: int | None = None,
        request_id: str | None = None,
        rank: int | None = None,
        rpc_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        active = self.active
        self.events.append(
            LifecycleEvent(
                timestamp=time.monotonic() if timestamp is None else timestamp,
                event=event,
                generation=(active.generation if active else generation),
                request_id=(active.request_id if active else request_id),
                rank=rank,
                rpc_id=rpc_id,
                phase=self.phase,
                details={} if details is None else details,
            )
        )


class TPGroupRecoveryController:
    """Bounds whole-group TP executor rebuilds after a poison decision."""

    def __init__(
        self,
        *,
        ranks: Iterable[int],
        executor_identity: str,
        max_attempts: int,
    ) -> None:
        self._ranks = frozenset(ranks)
        if not self._ranks:
            raise ValueError("recovery controller needs at least one rank")
        if not executor_identity:
            raise ValueError("executor identity is required")
        if max_attempts <= 0:
            raise ValueError("max recovery attempts must be positive")
        self.executor_identity = executor_identity
        self.max_attempts = max_attempts
        self.attempts = 0
        self.state = RecoveryState.READY
        self.poisoned_generation: int | None = None
        self.reason: str | None = None

    @property
    def can_accept_requests(self) -> bool:
        """Return true only after a verified full-group recovery."""
        return self.state is RecoveryState.READY

    def poison(self, generation: int, reason: str) -> None:
        """Record that the old TP group must never be reused."""
        if generation <= 0 or not reason:
            raise LifecycleError("poison needs a generation and reason")
        self.poisoned_generation = generation
        self.reason = reason
        self.state = RecoveryState.POISONED

    def begin_rebuild(self, ranks_to_rebuild: Iterable[int]) -> bool:
        """Start one bounded, all-rank-only executor rebuild attempt."""
        if self.state is not RecoveryState.POISONED:
            return False
        if frozenset(ranks_to_rebuild) != self._ranks:
            self.state = RecoveryState.UNAVAILABLE
            self.reason = "partial TP rank recovery is forbidden"
            return False
        if self.attempts >= self.max_attempts:
            self.state = RecoveryState.UNAVAILABLE
            self.reason = "recovery attempt budget exhausted"
            return False
        self.attempts += 1
        self.state = RecoveryState.RECOVERING
        return True

    def complete_rebuild(
        self,
        *,
        executor_identity: str,
        recovery_generation: int,
        rank_states: Mapping[int, RankFenceState],
    ) -> bool:
        """Accept only a fresh identity and a clean full TP-group baseline."""
        if self.state is not RecoveryState.RECOVERING:
            return False
        if executor_identity == self.executor_identity or not executor_identity:
            return self._candidate_rebuild_failed("executor identity did not change")
        if (
            self.poisoned_generation is None
            or recovery_generation <= self.poisoned_generation
        ):
            return self._candidate_rebuild_failed("recovery generation did not advance")
        if set(rank_states) != self._ranks:
            return self._candidate_rebuild_failed(
                "replacement did not rebuild every TP rank"
            )
        for rank in self._ranks:
            state = rank_states[rank]
            if (
                state.rank != rank
                or state.executor_identity != executor_identity
                or state.last_completed_generation != recovery_generation - 1
                or not state.worker_alive
                or state.poisoned
                or state.active_generation is not None
                or state.active_request_count
                or state.active_generation_rpc_count
                or state.unresolved_generation_future_count
                or state.owned_generation_shm_count
                or state.known_request_owned_state_count
                or state.candidate_swap_bytes
                or state.persistent_baseline_violations
            ):
                return self._candidate_rebuild_failed(
                    "replacement baseline is not clean"
                )
        self.executor_identity = executor_identity
        self.state = RecoveryState.READY
        self.reason = None
        return True

    def report_rebuild_failure(self, reason: str) -> bool:
        """Record a failed replacement attempt without ever returning READY.

        A retry is possible only while the configured attempt budget remains.
        The old TP group stays poisoned throughout; callers must create a new
        complete process group for every later attempt.
        """
        if self.state is not RecoveryState.RECOVERING:
            return False
        return self._candidate_rebuild_failed(reason)

    def fail_rebuild(self, reason: str) -> None:
        """Permanently fail closed when safe old-executor teardown is unproven."""
        self._fail_permanently(reason)

    def _candidate_rebuild_failed(self, reason: str) -> bool:
        if self.attempts < self.max_attempts:
            self.state = RecoveryState.POISONED
            self.reason = reason
            return False
        self._fail_permanently(reason)
        return False

    def _fail_permanently(self, reason: str) -> None:
        self.state = RecoveryState.UNAVAILABLE
        self.reason = reason


class RankLifecycleParticipant:
    """Rank-local ownership bookkeeping for a WorkerProc integration.

    The participant tracks only the local RPC, future, and SHM-message records
    explicitly handed to it.  Unknown runner, model, and persistent state is
    represented only by the caller-provided count in ``prepare_summary`` and
    is never cleared by this helper.
    """

    _LOCAL_RESOURCE_TYPES = frozenset(
        {ResourceType.RPC, ResourceType.FUTURE, ResourceType.SHM_MESSAGE}
    )

    def __init__(
        self,
        *,
        rank: int,
        executor_identity: str,
        registry_capacity: int = 256,
        initial_completed_generation: int = 0,
    ) -> None:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if not executor_identity:
            raise ValueError("executor identity is required")
        if initial_completed_generation < 0:
            raise ValueError("initial completed generation cannot be negative")
        self.rank = rank
        self.executor_identity = executor_identity
        self.registry = OwnershipRegistry(registry_capacity)
        self.phase = GenerationPhase.IDLE
        self.active: GenerationToken | None = None
        self.last_completed_generation = initial_completed_generation

    def begin(self, token: GenerationToken) -> None:
        """Adopt an EngineCore-issued generation before processing local work."""
        if token.executor_identity != self.executor_identity:
            raise TPGroupPoisoned("rank received a token for another executor")
        if self.active is not None or self.phase not in {
            GenerationPhase.IDLE,
            GenerationPhase.CLEAN_COMPLETE,
        }:
            raise GenerationLifecycleError("rank still has an active generation")
        if token.generation != self.last_completed_generation + 1:
            raise GenerationLifecycleError(
                "rank generation did not advance exactly once"
            )
        self.active = token
        self.phase = GenerationPhase.RUNNING

    def begin_rpc(
        self,
        envelope: ResponseEnvelope,
        *,
        future_id: str | None = None,
        shm_message_id: str | None = None,
    ) -> bool:
        """Register local, request-owned transport records for one RPC."""
        if not self._accept_envelope(envelope):
            return False
        self._register(envelope, ResourceType.RPC, envelope.rpc_id)
        if future_id is not None:
            self._register(envelope, ResourceType.FUTURE, future_id)
        if shm_message_id is not None:
            self._register(envelope, ResourceType.SHM_MESSAGE, shm_message_id)
        return True

    def complete_rpc(
        self,
        envelope: ResponseEnvelope,
        *,
        future_id: str | None = None,
        shm_message_id: str | None = None,
    ) -> bool:
        """Mark only caller-proven local records quiescent; never free them."""
        if not self._accept_envelope(envelope):
            return False
        self._mark_quiescent(envelope, ResourceType.RPC, envelope.rpc_id)
        if future_id is not None:
            self._mark_quiescent(envelope, ResourceType.FUTURE, future_id)
        if shm_message_id is not None:
            self._mark_quiescent(envelope, ResourceType.SHM_MESSAGE, shm_message_id)
        # The actual transport/future transitions above are the proof.  This
        # compacts only their logical metadata; commit still owns retirement.
        self.registry.compact_quiescent(envelope.generation, rank=self.rank)
        return True

    def prepare_summary(
        self,
        generation: int,
        *,
        worker_alive: bool,
        active_request_count: int = 0,
        known_request_owned_state_count: int = 0,
        collective_sequence: int | None = None,
        candidate_swap_bytes: int = 0,
        persistent_baseline_violations: tuple[str, ...] = (),
    ) -> RankFenceState:
        """Return a rank-local phase-one proof without touching unknown state."""
        self._require_generation(generation)
        if (
            active_request_count < 0
            or known_request_owned_state_count < 0
            or candidate_swap_bytes < 0
        ):
            raise ValueError("request-owned counts cannot be negative")
        active_rpc_count = self._count_active(generation, ResourceType.RPC)
        active_future_count = self._count_active(generation, ResourceType.FUTURE)
        active_shm_count = self._count_active(generation, ResourceType.SHM_MESSAGE)
        has_active_local_state = bool(
            active_request_count
            or active_rpc_count
            or active_future_count
            or active_shm_count
            or known_request_owned_state_count
        )
        if not has_active_local_state:
            self.phase = GenerationPhase.PREPARED
        return RankFenceState(
            rank=self.rank,
            executor_identity=self.executor_identity,
            last_completed_generation=self.last_completed_generation,
            active_generation=generation if has_active_local_state else None,
            active_request_count=active_request_count,
            active_generation_rpc_count=active_rpc_count,
            unresolved_generation_future_count=active_future_count,
            owned_generation_shm_count=active_shm_count,
            known_request_owned_state_count=known_request_owned_state_count,
            worker_alive=worker_alive,
            collective_sequence=collective_sequence,
            candidate_swap_bytes=candidate_swap_bytes,
            persistent_baseline_violations=persistent_baseline_violations,
        )

    def commit(self, generation: int) -> bool:
        """Retire only locally quiescent ownership after TP-wide prepare."""
        if (
            self.phase is GenerationPhase.CLEAN_COMPLETE
            and self.last_completed_generation == generation
        ):
            return True
        self._require_generation(generation)
        if self.phase is not GenerationPhase.PREPARED:
            raise GenerationLifecycleError("rank commit requires a prepared summary")
        self.registry.release_quiescent(generation)
        self.registry.retire_released(generation)
        self.last_completed_generation = generation
        self.active = None
        self.phase = GenerationPhase.CLEAN_COMPLETE
        return True

    def baseline_summary(
        self,
        *,
        worker_alive: bool,
        active_request_count: int = 0,
        known_request_owned_state_count: int = 0,
        collective_sequence: int | None = None,
        candidate_swap_bytes: int = 0,
        persistent_baseline_violations: tuple[str, ...] = (),
    ) -> RankFenceState:
        """Report the post-commit rank fence without reopening an old request."""
        if self.active is not None:
            raise GenerationLifecycleError("rank cannot report baseline while active")
        if (
            active_request_count < 0
            or known_request_owned_state_count < 0
            or candidate_swap_bytes < 0
        ):
            raise ValueError("request-owned counts cannot be negative")
        return RankFenceState(
            rank=self.rank,
            executor_identity=self.executor_identity,
            last_completed_generation=self.last_completed_generation,
            active_request_count=active_request_count,
            known_request_owned_state_count=known_request_owned_state_count,
            worker_alive=worker_alive,
            collective_sequence=collective_sequence,
            candidate_swap_bytes=candidate_swap_bytes,
            persistent_baseline_violations=persistent_baseline_violations,
        )

    def _accept_envelope(self, envelope: ResponseEnvelope) -> bool:
        active = self.active
        return bool(
            active
            and self.phase is GenerationPhase.RUNNING
            and envelope.generation == active.generation
            and envelope.request_id == active.request_id
            and envelope.executor_identity == active.executor_identity
        )

    def _register(
        self,
        envelope: ResponseEnvelope,
        resource_type: ResourceType,
        resource_id: str,
    ) -> None:
        if resource_type not in self._LOCAL_RESOURCE_TYPES:
            raise OwnershipError("rank participant cannot own that resource type")
        self.registry.register(
            generation=envelope.generation,
            request_id=envelope.request_id,
            rank=self.rank,
            resource_type=resource_type,
            resource_id=resource_id,
        )

    def _mark_quiescent(
        self,
        envelope: ResponseEnvelope,
        resource_type: ResourceType,
        resource_id: str,
    ) -> None:
        self.registry.mark_quiescent(
            generation=envelope.generation,
            rank=self.rank,
            resource_type=resource_type,
            resource_id=resource_id,
        )

    def _count_active(self, generation: int, resource_type: ResourceType) -> int:
        return sum(
            record.resource_type is resource_type
            and record.state is ResourceState.ACTIVE
            for record in self.registry.records(generation)
        )

    def _require_generation(self, generation: int) -> None:
        if self.active is None or self.active.generation != generation:
            raise GenerationLifecycleError("rank generation is not active")


class GenerationCoordinator(TPGenerationLifecycle):
    """Integration-facing generation coordinator with bounded TP recovery.

    The executor calls this object at its actual request and worker lifecycle
    boundaries.  The coordinator makes no transport or GPU calls itself, so a
    timeout cannot accidentally force-free an unknown live resource.
    """

    def __init__(
        self,
        *,
        ranks: Iterable[int],
        executor_identity: str,
        registry_capacity: int = 1024,
        event_capacity: int = 256,
        max_recovery_attempts: int = 2,
    ) -> None:
        self._registry_capacity = registry_capacity
        super().__init__(
            ranks=ranks,
            executor_identity=executor_identity,
            registry_capacity=registry_capacity,
            event_capacity=event_capacity,
        )
        self.recovery = TPGroupRecoveryController(
            ranks=self._ranks,
            executor_identity=executor_identity,
            max_attempts=max_recovery_attempts,
        )

    def begin(self, request_id: str, timestamp: float | None = None) -> GenerationToken:
        """Allocate the next generation only when the TP group is verified ready."""
        if (
            self.phase is GenerationPhase.POISONED
            or not self.recovery.can_accept_requests
        ):
            raise TPGroupPoisoned(
                self._poison_reason or self.recovery.reason or "TP group unavailable"
            )
        return self.begin_generation(request_id, timestamp)

    def prepare(
        self,
        generation: int,
        state: RankFenceState,
        timestamp: float | None = None,
    ) -> bool:
        """Integration-friendly alias for one rank's phase-one proof."""
        return self.prepare_rank(generation, state, timestamp)

    def commit(self, generation: int, timestamp: float | None = None) -> bool:
        """Integration-friendly alias for phase-two cleanup commit."""
        return self.commit_cleanup(generation, timestamp)

    def baseline(
        self,
        generation: int,
        rank_states: Mapping[int, RankFenceState],
        timestamp: float | None = None,
    ) -> bool:
        """Verify a clean TP baseline before admitting the next generation."""
        return self.verify_post_request_baseline(generation, rank_states, timestamp)

    def begin_rebuild(self, ranks_to_rebuild: Iterable[int]) -> bool:
        """Begin a bounded all-rank executor rebuild after poison."""
        return self.recovery.begin_rebuild(ranks_to_rebuild)

    def rebuild(
        self,
        *,
        executor_identity: str,
        recovery_generation: int,
        rank_states: Mapping[int, RankFenceState],
        old_executor_destroyed: bool,
        timestamp: float | None = None,
    ) -> bool:
        """Adopt only a verified, whole-group replacement executor.

        Args:
            executor_identity: Identity assigned to the newly created executor.
            recovery_generation: First generation reserved for that executor.
            rank_states: Clean-baseline proofs from all replacement ranks.
            old_executor_destroyed: Confirmation that the poisoned group was
                fully torn down before its metadata is discarded.
            timestamp: Optional monotonic timestamp for the diagnostic window.
        """
        if not old_executor_destroyed:
            self.recovery.fail_rebuild("old TP executor was not fully destroyed")
            return False
        if not self.recovery.complete_rebuild(
            executor_identity=executor_identity,
            recovery_generation=recovery_generation,
            rank_states=rank_states,
        ):
            return False
        self.executor_identity = executor_identity
        self.registry = OwnershipRegistry(self._registry_capacity)
        self.active = None
        self.last_completed_generation = recovery_generation - 1
        self._prepared.clear()
        self._poison_reason = None
        self.phase = GenerationPhase.IDLE
        self._event(
            "tp_executor_rebuilt",
            timestamp=timestamp,
            generation=recovery_generation,
            details={"executor_identity": executor_identity},
        )
        return True

    def _poison(self, reason: str, timestamp: float | None) -> bool:
        result = super()._poison(reason, timestamp)
        generation = (
            self.active.generation
            if self.active
            else self.last_completed_generation + 1
        )
        self.recovery.poison(generation, reason)
        return result
