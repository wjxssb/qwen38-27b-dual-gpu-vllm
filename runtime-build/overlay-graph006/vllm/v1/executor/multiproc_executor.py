# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import uuid
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, cast

import cloudpickle
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_inner_dp_world_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.stability_telemetry import configure_process as configure_stability_process
from vllm.stability_telemetry import emit as emit_stability_telemetry
from vllm.stability_telemetry import (
    emit_failure_window as emit_stability_failure_window,
)
from vllm.stability_telemetry import is_enabled as stability_telemetry_enabled
from vllm.stability_telemetry import lifecycle_scope as stability_lifecycle_scope
from vllm.stability_telemetry import rpc_scope as stability_rpc_scope
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.utils import numa_utils
from vllm.utils.network_utils import (
    aiter_requires_tcp_store,
    get_distributed_init_method,
    get_file_store_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.ompmultiprocessing import OMPProcessManager
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.utils.torch_utils import (
    OMP_NUM_THREADS_SET_BY_VLLM,
    set_torch_threads_for_runtime,
    startup_omp_num_threads,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.executor.tp_lifecycle import (
    GenerationCoordinator,
    GenerationLifecycleError,
    GenerationPhase,
    GenerationToken,
    ProgressWatchdog,
    RankFenceState,
    RankLifecycleParticipant,
    ResourceState,
    ResourceType,
    ResponseEnvelope,
    TPGroupPoisoned,
    WatchdogClassification,
)
from vllm.v1.executor.tp_lifecycle import (
    is_enabled as stability_lifecycle_enabled,
)
from vllm.v1.executor.vllm_net_devices import set_worker_net_device
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


_STABILITY_CONTROL_RPC = "__stability_tp_lifecycle_control__"


def _stability_positive_int_env(name: str, default: int) -> int:
    """Read a bounded lifecycle setting without making startup permissive."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return parsed


def _stability_candidate_swap_snapshot() -> tuple[int, tuple[str, ...]]:
    """Read the candidate cgroup's swap counter at a lifecycle boundary.

    The stability profile is deliberately cgroup-v2-only for swap accounting.
    Treat an absent or malformed counter as an unproven baseline rather than
    silently reporting zero.  This executes only in the bounded control RPC,
    never in model execution or queue hot paths.
    """
    path = "/sys/fs/cgroup/memory.swap.current"
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read().strip()
    except OSError:
        return 0, ("candidate_swap_accounting_unavailable",)
    try:
        value = int(raw)
    except ValueError:
        return 0, ("candidate_swap_accounting_malformed",)
    if value < 0:
        return 0, ("candidate_swap_accounting_negative",)
    return value, ()


def _stability_scheduler_summary(args: tuple[Any, ...]) -> dict[str, Any]:
    if not args or not isinstance(args[0], SchedulerOutput):
        return {}
    scheduler_output = args[0]
    progress: list[dict[str, Any]] = []
    for request in scheduler_output.scheduled_new_reqs:
        progress.append(
            {
                "request_id": request.req_id,
                "num_computed_tokens": request.num_computed_tokens,
                "scheduled_tokens": scheduler_output.num_scheduled_tokens.get(
                    request.req_id, 0
                ),
            }
        )
    cached = scheduler_output.scheduled_cached_reqs
    for request_id, num_computed_tokens in zip(
        cached.req_ids, cached.num_computed_tokens
    ):
        progress.append(
            {
                "request_id": request_id,
                "num_computed_tokens": num_computed_tokens,
                "scheduled_tokens": scheduler_output.num_scheduled_tokens.get(
                    request_id, 0
                ),
            }
        )
    return {
        "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
        "request_progress": progress,
        "finished_request_ids": sorted(scheduler_output.finished_req_ids),
    }


class FutureWrapper(Future):
    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        get_response: Callable[[], Any],
        aggregate: Callable = lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
        super().__init__()
        self.futures_queue.appendleft(self)

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        # Drain any futures ahead of us in the queue.
        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()

    def _wait_for_response(self):
        try:
            response = self.aggregate(self.get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


class MultiprocExecutor(Executor):
    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        self.monitor_workers = monitor_workers
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        configure_stability_process("engine-core")
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None
        self._stability_next_rpc_id = 1
        self._stability_active_rpcs: dict[int, dict[str, Any]] = {}
        self._stability_next_sample_context: dict[str, Any] | None = None
        self._stability_lifecycle: GenerationCoordinator | None = None
        self._stability_executor_identity: str | None = None
        self._stability_generation_by_request: dict[str, int] = {}
        self._stability_pending_finalization: set[str] = set()
        self._stability_recovery_required = False
        self._stability_last_rank_states: dict[int, RankFenceState] = {}
        self._stability_progress_watchdog: ProgressWatchdog | None = None
        self._stability_watchdog_stop = threading.Event()
        self._stability_watchdog_thread: Thread | None = None
        self._stability_last_watchdog_diagnostic: dict[int, float] = {}
        # This is deliberately a best-effort, bounded CPU IPC queue.  It is
        # not a vLLM MessageQueue, does not carry model work, and is never
        # used to initiate a collective or lifecycle transition.
        self._stability_heartbeat_queue: Any | None = None
        self._stability_heartbeat_capacity = 0
        self._stability_heartbeat_drain_limit = 0
        self._stability_heartbeat_received = 0
        self._stability_heartbeat_rejected = 0
        self._stability_heartbeat_stale = 0
        self._stability_heartbeat_transport_errors = 0
        self._stability_heartbeat_last_by_rank: dict[int, dict[str, Any]] = {}

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        if stability_lifecycle_enabled():
            # This feature gate is deliberately narrow.  It is the frozen
            # campaign topology, not a general distributed-runtime rewrite.
            if (
                self.world_size != 2
                or tp_size != 2
                or pp_size != 1
                or pcp_size != 1
                or self.vllm_config.scheduler_config.max_num_seqs != 1
            ):
                raise RuntimeError(
                    "VLLM_TP_GENERATION_LIFECYCLE is limited to the "
                    "TP2/PP1/DCP1/max_num_seqs=1 stability profile"
                )
            self._stability_executor_identity = uuid.uuid4().hex
            self._stability_lifecycle = GenerationCoordinator(
                ranks=range(self.world_size),
                executor_identity=self._stability_executor_identity,
                registry_capacity=_stability_positive_int_env(
                    "VLLM_TP_LIFECYCLE_REGISTRY_CAPACITY", 1024
                ),
                max_recovery_attempts=_stability_positive_int_env(
                    "VLLM_TP_LIFECYCLE_MAX_RECOVERY_ATTEMPTS", 2
                ),
            )
            self._stability_progress_watchdog = ProgressWatchdog(
                range(self.world_size),
                diagnostic_interval_seconds=float(
                    _stability_positive_int_env(
                        "VLLM_TP_LIFECYCLE_WATCHDOG_INTERVAL_S", 15
                    )
                ),
            )
            emit_stability_telemetry(
                "tp_lifecycle_initialized",
                executor_identity=self._stability_executor_identity,
                world_size=self.world_size,
                tensor_parallel_size=tp_size,
                max_num_seqs=self.vllm_config.scheduler_config.max_num_seqs,
            )

        set_multiprocessing_worker_envs(self.local_world_size)

        if aiter_requires_tcp_store():
            distributed_init_method = get_distributed_init_method(
                get_loopback_ip(), get_open_port()
            )
        else:
            distributed_init_method = get_file_store_init_method()
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        if self.parallel_config.node_rank_within_dp == 0:
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            mq_connect_ip = get_ip()
            logger.info(
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "
                "master_addr=%s, mq_connect_ip=%s (local), "
                "world_size=%d, local_world_size=%d",
                self.parallel_config.node_rank,
                self.parallel_config.node_rank_within_dp,
                self.parallel_config.master_addr,
                mq_connect_ip,
                self.world_size,
                self.local_world_size,
            )
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
                name="rpc_broadcast",
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # Create workers
        context = get_mp_context()
        if self._stability_generation_coordinator() is not None:
            self._stability_heartbeat_capacity = _stability_positive_int_env(
                "VLLM_TP_LIFECYCLE_HEARTBEAT_CAPACITY", 256
            )
            self._stability_heartbeat_drain_limit = _stability_positive_int_env(
                "VLLM_TP_LIFECYCLE_HEARTBEAT_DRAIN_LIMIT",
                self._stability_heartbeat_capacity,
            )
            self._stability_heartbeat_queue = context.Queue(
                maxsize=self._stability_heartbeat_capacity
            )
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            # When using fork, keep track of socket file descriptors that are
            # inherited by the worker, so that we can close them in subsequent
            # workers
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )

            # For CPU backend only, to setup OpenMP threads affinity
            cpu_omp_manager = OMPProcessManager(self.vllm_config)
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                with cpu_omp_manager.configure_omp_envs(
                    rank=global_rank, local_rank=local_rank
                ):
                    unready_worker_handle = WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=local_rank,
                        rank=global_rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                        is_driver_worker=is_driver_worker,
                        stability_executor_identity=self._stability_executor_identity,
                        stability_heartbeat_queue=self._stability_heartbeat_queue,
                        inherited_fds=inherited_fds,
                    )
                unready_workers.append(unready_worker_handle)
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # Wait for all local workers to be ready.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # The workers have inherited their thread count (see
            # set_multiprocessing_worker_envs); this process only schedules, so
            # it gets no benefit from torch intra-op parallelism, just CPU
            # contention with them.
            set_torch_threads_for_runtime()

            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            self.futures_queue = deque[FutureWrapper]()

            self._post_init_executor()

            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()
        if self._stability_generation_coordinator() is not None:
            # This occurs after both worker response MQs are ready and before
            # EngineCore begins accepting HTTP requests.  It proves the clean
            # generation-zero baseline rather than merely trusting startup.
            try:
                self._stability_assert_rank_fence(0)
                self._stability_emit_lifecycle_status(
                    "tp_lifecycle_status", baseline_pass=True
                )
                self._stability_start_watchdog()
            except GenerationLifecycleError as exc:
                self._stability_poison(f"startup rank fence failed: {exc}")
                # Startup did create a complete TP group.  A failed initial
                # fence must therefore tear down the complete group now,
                # rather than relying on a later finalizer to leave it alive.
                self.shutdown()
                raise

    def get_response_mqs(self, unique_reply_rank: int = -1) -> list[MessageQueue]:
        assert unique_reply_rank >= -1 and unique_reply_rank < self.world_size, (
            f"unique_reply_rank must be -1 or < world_size,"
            f"unique_reply_rank = {unique_reply_rank}, "
            f"world_size={self.world_size}"
        )
        ranks = (
            [unique_reply_rank] if unique_reply_rank != -1 else range(self.world_size)
        )
        return [self.workers[rank].worker_response_mq for rank in ranks]

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        return rank % self.parallel_config.tensor_parallel_size == 0

    def start_worker_monitor(self, inline=False) -> None:
        workers = self.workers
        self_ref = weakref.ref(self)
        monitored_executor_identity = getattr(
            self, "_stability_executor_identity", None
        )

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if (
                not _self
                or getattr(_self, "shutting_down", False)
                or getattr(_self, "workers", None) is not workers
                or getattr(_self, "_stability_executor_identity", None)
                != monitored_executor_identity
            ):
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
                return
            _self.is_failed = True
            proc = next(h.proc for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly (exit code: %s), "
                "shutting down executor.",
                proc.name,
                proc.exitcode,
            )
            if _self._stability_generation_coordinator() is not None:
                _self._stability_poison(
                    f"TP worker died rank/process={proc.name} exit={proc.exitcode}"
                )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        if not inline:
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        monitor_workers()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        stability_context = self._take_stability_sample_context()
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
            stability_context=stability_context,
        )

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    # The methods in this block are intentionally a small coordinator layer,
    # not a second scheduler.  They are enabled only in the independent TP2
    # stability profile and retain metadata for resources whose ownership is
    # provable at an executor boundary.  They never clear a worker, KV cache,
    # GDN state, CUDA event, or SHM slot by force.
    def _stability_generation_coordinator(self) -> GenerationCoordinator | None:
        return getattr(self, "_stability_lifecycle", None)

    def stability_request_admission_state(self) -> str:
        """Return the generation-admission state without changing it.

        The EngineCore uses this only to keep later client ADD messages out of
        the scheduler while the one permitted generation is still completing
        its ordinary worker cleanup tick.  It is deliberately a read-only
        state query: it cannot clear, cancel, or recover a TP resource.
        """
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return "disabled"
        if (
            coordinator.phase is GenerationPhase.POISONED
            or not coordinator.recovery.can_accept_requests
        ):
            return "poisoned"
        return "ready" if coordinator.can_accept_requests else "busy"

    def stability_begin_generation(self, request_id: str) -> int | None:
        """Fence every TP rank before admitting the next request generation."""
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return None

        try:
            self._stability_assert_rank_fence(coordinator.last_completed_generation)
            token = coordinator.begin(request_id)
            coordinator.registry.register(
                generation=token.generation,
                request_id=token.request_id,
                rank=0,
                resource_type=ResourceType.REQUEST_STATE,
                resource_id=f"scheduler:{token.request_id}",
            )
        except GenerationLifecycleError as exc:
            self._stability_poison(f"generation admission failed: {exc}")
            raise TPGroupPoisoned(str(exc)) from exc

        self._stability_generation_by_request[token.request_id] = token.generation
        emit_stability_telemetry(
            "generation_begin",
            generation=token.generation,
            request_id=token.request_id,
            executor_identity=token.executor_identity,
            phase=coordinator.phase.value,
        )
        self._stability_emit_lifecycle_status(
            "tp_lifecycle_transition", baseline_pass=False
        )
        return token.generation

    def stability_mark_abort(self, request_id: str) -> None:
        """Mark a scheduler abort; normal finished-request cleanup still runs."""
        coordinator = self._stability_generation_coordinator()
        generation = self._stability_generation_by_request.get(request_id)
        if coordinator is None or generation is None:
            return
        try:
            if coordinator.phase is GenerationPhase.BEGIN:
                coordinator.mark_running(generation)
            if coordinator.phase is GenerationPhase.RUNNING:
                coordinator.mark_aborting(generation)
            self._stability_pending_finalization.add(request_id)
            emit_stability_telemetry(
                "generation_aborting",
                generation=generation,
                request_id=request_id,
                executor_identity=coordinator.executor_identity,
            )
        except GenerationLifecycleError as exc:
            self._stability_poison(f"abort lifecycle transition failed: {exc}")

    def stability_admission_failed(self, request_id: str) -> None:
        """Poison an unadmitted generation whose scheduler state is unknown.

        A scheduler ``add_request`` exception has no later ``finished_req_ids``
        boundary through which normal TP-wide cleanup could be proven.  Keep
        the ownership metadata for diagnostics and require a full replacement
        instead of leaving the generation in ``ABORTING`` indefinitely.
        """
        coordinator = self._stability_generation_coordinator()
        generation = self._stability_generation_by_request.get(request_id)
        if coordinator is None or generation is None:
            return
        self._stability_poison(
            "scheduler admission failed without a worker cleanup proof: "
            f"generation={generation} request_id={request_id}"
        )

    def stability_mark_completing(self, request_id: str) -> None:
        """Mark final output, then wait for normal worker cleanup scheduling."""
        coordinator = self._stability_generation_coordinator()
        generation = self._stability_generation_by_request.get(request_id)
        if coordinator is None or generation is None:
            return
        try:
            if coordinator.phase is GenerationPhase.BEGIN:
                coordinator.mark_running(generation)
            if coordinator.phase is GenerationPhase.RUNNING:
                coordinator.mark_completing(generation)
            self._stability_pending_finalization.add(request_id)
            emit_stability_telemetry(
                "generation_completing",
                generation=generation,
                request_id=request_id,
                executor_identity=coordinator.executor_identity,
            )
        except GenerationLifecycleError as exc:
            self._stability_poison(f"completion lifecycle transition failed: {exc}")

    def stability_finalize_generation(
        self,
        request_id: str,
        *,
        engine_active_request_count: int,
        engine_active_request_ids: Sequence[str],
    ) -> None:
        """Run all-rank prepare/commit only after normal worker cleanup.

        ``finished_req_ids`` is consumed by GPUModelRunner on a subsequent
        normal execution.  EngineCore calls this method only after that batch
        has completed, so observing empty request state is a proof rather than
        a local reset.
        """
        coordinator = self._stability_generation_coordinator()
        generation = self._stability_generation_by_request.get(request_id)
        if coordinator is None or generation is None:
            return
        if request_id not in self._stability_pending_finalization:
            return
        try:
            if engine_active_request_count or engine_active_request_ids:
                raise GenerationLifecycleError(
                    "scheduler has request state at post-request boundary"
                )
            coordinator.registry.mark_quiescent(
                generation=generation,
                rank=0,
                resource_type=ResourceType.REQUEST_STATE,
                resource_id=f"scheduler:{request_id}",
            )
            coordinator.begin_quiesce(generation)

            prepared_states = self._stability_control_rank_states(
                "prepare", generation, request_id
            )
            for rank in range(self.world_size):
                if not coordinator.prepare(generation, prepared_states[rank]):
                    raise TPGroupPoisoned("one TP rank did not prepare")

            committed_states = self._stability_control_rank_states(
                "commit", generation, request_id
            )
            # Each rank must report that its local lifecycle metadata is now
            # retired before coordinator metadata is committed.
            for rank, state in committed_states.items():
                if state.active_generation is not None or state.quiescence_failures(
                    generation
                ):
                    raise TPGroupPoisoned(
                        f"rank {rank} did not commit a clean generation"
                    )
            if not coordinator.commit(generation):
                raise TPGroupPoisoned("coordinator cleanup commit failed")

            baseline_states = self._stability_control_rank_states(
                "fence", generation, request_id
            )
            if not coordinator.baseline(generation, baseline_states):
                raise TPGroupPoisoned("post-request baseline was not clean")
        except (GenerationLifecycleError, TPGroupPoisoned) as exc:
            self._stability_poison(f"generation finalization failed: {exc}")
            raise TPGroupPoisoned(str(exc)) from exc

        self._stability_pending_finalization.discard(request_id)
        self._stability_generation_by_request.pop(request_id, None)
        # Some unit tests intentionally construct a lightweight executor via
        # ``object.__new__`` to exercise only the control path.  Lifecycle
        # finalization must remain safe in that diagnostic-only shape too.
        watchdog = getattr(self, "_stability_progress_watchdog", None)
        if watchdog is not None:
            watchdog.retire_completed_generations(generation)
        self._stability_last_watchdog_diagnostic = {
            observed_generation: timestamp
            for observed_generation, timestamp in getattr(
                self, "_stability_last_watchdog_diagnostic", {}
            ).items()
            if observed_generation > generation
        }
        emit_stability_telemetry(
            "post_request_baseline",
            generation=generation,
            request_id=request_id,
            executor_identity=coordinator.executor_identity,
            phase=coordinator.phase.value,
            engine_active_request_count=engine_active_request_count,
            engine_active_request_ids=list(engine_active_request_ids),
            lifecycle=self._stability_lifecycle_snapshot(),
        )
        self._stability_emit_lifecycle_status(
            "tp_lifecycle_transition", baseline_pass=True
        )

    def stability_note_scheduler_progress(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Record scheduler evidence without changing the hard deadline.

        The watchdog's rank samples come only from the two WorkerProc
        heartbeats.  Treating EngineCore scheduler activity as rank zero would
        hide a rank-zero worker stall and make a TP skew appear healthy.
        """
        coordinator = self._stability_generation_coordinator()
        if coordinator is None or coordinator.active is None:
            return
        summary = _stability_scheduler_summary((scheduler_output,))
        for request in summary.get("request_progress", []):
            if request.get("request_id") != coordinator.active.request_id:
                continue
            emit_stability_telemetry(
                "generation_progress",
                generation=coordinator.active.generation,
                request_id=coordinator.active.request_id,
                executor_identity=coordinator.active.executor_identity,
                phase=coordinator.phase.value,
                computed_tokens=request.get("num_computed_tokens"),
                scheduled_tokens=request.get("scheduled_tokens"),
            )

    def _stability_heartbeat_snapshot(
        self, *, timestamp: float | None = None, drained_last_cycle: int = 0
    ) -> dict[str, Any]:
        """Return bounded diagnostic state for the noncollective channel."""
        now = time.monotonic() if timestamp is None else timestamp
        last_by_rank = {
            str(rank): {
                **heartbeat,
                "age_seconds": max(0.0, now - float(heartbeat["timestamp"])),
            }
            for rank, heartbeat in sorted(
                getattr(self, "_stability_heartbeat_last_by_rank", {}).items()
            )
        }
        return {
            "kind": "bounded_multiprocessing_queue",
            "capacity": getattr(self, "_stability_heartbeat_capacity", 0),
            "drain_limit": getattr(self, "_stability_heartbeat_drain_limit", 0),
            "drained_last_cycle": drained_last_cycle,
            "received_total": getattr(self, "_stability_heartbeat_received", 0),
            "rejected_total": getattr(self, "_stability_heartbeat_rejected", 0),
            "stale_total": getattr(self, "_stability_heartbeat_stale", 0),
            "transport_error_total": getattr(
                self, "_stability_heartbeat_transport_errors", 0
            ),
            "last_by_rank": last_by_rank,
        }

    def _stability_drain_tp_heartbeats(
        self,
        active: GenerationToken,
        *,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Aggregate only current-generation worker heartbeats in the parent.

        The queue is strictly diagnostic.  Its contents cannot submit work,
        change a deadline, trigger a collective, clear a resource, or affect
        the generation state machine.  Invalid or delayed messages are
        retained only as bounded counters and never interpreted as current
        rank progress.
        """
        now = time.monotonic() if timestamp is None else timestamp
        channel = getattr(self, "_stability_heartbeat_queue", None)
        watchdog = getattr(self, "_stability_progress_watchdog", None)
        if channel is None or watchdog is None:
            return self._stability_heartbeat_snapshot(timestamp=now)

        drain_limit = getattr(self, "_stability_heartbeat_drain_limit", 0)
        if drain_limit <= 0:
            return self._stability_heartbeat_snapshot(timestamp=now)

        drained = 0
        for _ in range(drain_limit):
            try:
                heartbeat = channel.get_nowait()
            except queue.Empty:
                break
            except Exception as exc:
                # The diagnostic reader must not die just because the queue
                # implementation has become unavailable during teardown.
                self._stability_heartbeat_transport_errors += 1
                emit_stability_telemetry(
                    "worker_heartbeat_transport_error",
                    exception_type=type(exc).__name__,
                )
                break
            drained += 1
            if not isinstance(heartbeat, dict):
                self._stability_heartbeat_rejected += 1
                continue
            try:
                rank = int(heartbeat["rank"])
                generation = int(heartbeat["generation"])
                request_id = str(heartbeat["request_id"])
                executor_identity = str(heartbeat["executor_identity"])
                phase = GenerationPhase(str(heartbeat["phase"]))
                computed_tokens = int(heartbeat["computed_tokens"])
                heartbeat_timestamp = float(heartbeat["timestamp"])
                rpc_id = str(heartbeat.get("rpc_id", "")) or None
            except (KeyError, OverflowError, TypeError, ValueError):
                self._stability_heartbeat_rejected += 1
                continue
            if (
                rank not in range(self.world_size)
                or computed_tokens < 0
                or not math.isfinite(heartbeat_timestamp)
            ):
                self._stability_heartbeat_rejected += 1
                continue
            if (
                executor_identity != active.executor_identity
                or generation != active.generation
                or request_id != active.request_id
            ):
                self._stability_heartbeat_stale += 1
                emit_stability_telemetry(
                    "stale_worker_heartbeat",
                    expected_executor_identity=active.executor_identity,
                    expected_generation=active.generation,
                    expected_request_id=active.request_id,
                    observed=heartbeat,
                )
                continue
            normalized = {
                "executor_identity": executor_identity,
                "generation": generation,
                "request_id": request_id,
                "phase": phase.value,
                "computed_tokens": computed_tokens,
                "timestamp": heartbeat_timestamp,
                "rpc_id": rpc_id,
            }
            self._stability_heartbeat_last_by_rank[rank] = normalized
            self._stability_heartbeat_received += 1
            watchdog.observe(
                generation=generation,
                rank=rank,
                phase=phase,
                computed_tokens=computed_tokens,
                timestamp=heartbeat_timestamp,
                rpc_id=rpc_id,
            )
        return self._stability_heartbeat_snapshot(
            timestamp=now, drained_last_cycle=drained
        )

    def _stability_start_watchdog(self) -> None:
        """Start a read-only, rate-limited diagnostic watchdog.

        It never submits an RPC, changes the 300s deadline, or tries a local
        reset.  A stuck worker may be inside a collective, so a diagnostic
        thread must not inject a second collective behind it.
        """
        if self._stability_watchdog_thread is not None:
            return
        watchdog = self._stability_progress_watchdog
        if watchdog is None:
            return
        interval = float(
            _stability_positive_int_env("VLLM_TP_LIFECYCLE_WATCHDOG_INTERVAL_S", 15)
        )

        def loop() -> None:
            while not self._stability_watchdog_stop.wait(interval):
                coordinator = self._stability_generation_coordinator()
                active = coordinator.active if coordinator is not None else None
                if active is None:
                    continue
                heartbeat = self._stability_drain_tp_heartbeats(active)
                assessment = watchdog.assess(active.generation, time.monotonic())
                if assessment.classification not in {
                    WatchdogClassification.NO_PROGRESS,
                    WatchdogClassification.RANK_SKEW,
                }:
                    continue
                last_emitted = self._stability_last_watchdog_diagnostic.get(
                    active.generation, 0.0
                )
                now = time.monotonic()
                if now - last_emitted < interval:
                    continue
                self._stability_last_watchdog_diagnostic[active.generation] = now
                emit_stability_failure_window(
                    "tp_lifecycle_watchdog",
                    executor_identity=active.executor_identity,
                    generation=active.generation,
                    request_id=active.request_id,
                    classification=assessment.classification.value,
                    reason=assessment.reason,
                    lifecycle=self._stability_lifecycle_snapshot(),
                    rpc_lifecycle=self._stability_active_rpc_snapshot(),
                    queues=self._stability_queue_snapshots(include_slots=True),
                    heartbeat=heartbeat,
                    events=[
                        {
                            "event": event.event,
                            "rank": event.rank,
                            "phase": (
                                event.phase.value if event.phase is not None else None
                            ),
                            "computed_tokens": event.computed_tokens,
                        }
                        for event in assessment.events[-32:]
                    ],
                )

        self._stability_watchdog_thread = Thread(
            target=loop,
            daemon=True,
            name="TPGenerationLifecycleWatchdog",
        )
        self._stability_watchdog_thread.start()

    def _stability_assert_rank_fence(self, generation: int) -> None:
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return
        states = self._stability_control_rank_states("fence", generation, None)
        failures = coordinator.rank_fence_failures(states, generation)
        if failures:
            raise TPGroupPoisoned("rank fence failed: " + ",".join(failures))

    def _stability_control_rank_states(
        self, operation: str, generation: int, request_id: str | None
    ) -> dict[int, RankFenceState]:
        """Use an all-rank control RPC as the bounded TP lifecycle fence."""
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return {}
        try:
            responses = self.collective_rpc(
                _STABILITY_CONTROL_RPC,
                args=(operation, generation, request_id),
                unique_reply_rank=None,
                timeout=float(
                    _stability_positive_int_env(
                        "VLLM_TP_LIFECYCLE_CONTROL_TIMEOUT_S", 30
                    )
                ),
                stability_control=True,
            )
        except Exception as exc:
            # Lifecycle control is the proof needed before reuse.  A timeout
            # or transport error leaves that proof absent, so preserve the
            # old group for diagnostics and fail closed immediately.
            self._stability_poison(
                f"TP lifecycle control {operation} failed: {type(exc).__name__}: {exc}"
            )
            raise TPGroupPoisoned(f"TP lifecycle control {operation} failed") from exc
        if not isinstance(responses, list) or len(responses) != self.world_size:
            raise TPGroupPoisoned("TP lifecycle control response count mismatch")
        states: dict[int, RankFenceState] = {}
        for expected_rank, response in enumerate(responses):
            if not isinstance(response, dict):
                raise TPGroupPoisoned("TP lifecycle control response is malformed")
            try:
                persistent_violations = response.get(
                    "persistent_baseline_violations", ()
                )
                if not isinstance(persistent_violations, (list, tuple)):
                    raise ValueError("persistent baseline violations are malformed")
                rank_state = RankFenceState(
                    rank=int(response["rank"]),
                    executor_identity=str(response["executor_identity"]),
                    last_completed_generation=int(
                        response["last_completed_generation"]
                    ),
                    active_generation=response.get("active_generation"),
                    active_request_count=int(response.get("active_request_count", 0)),
                    active_generation_rpc_count=int(
                        response.get("active_generation_rpc_count", 0)
                    ),
                    unresolved_generation_future_count=int(
                        response.get("unresolved_generation_future_count", 0)
                    ),
                    owned_generation_shm_count=int(
                        response.get("owned_generation_shm_count", 0)
                    ),
                    known_request_owned_state_count=int(
                        response.get("known_request_owned_state_count", 0)
                    ),
                    worker_alive=bool(response.get("worker_alive", False)),
                    poisoned=bool(response.get("poisoned", False)),
                    collective_sequence=response.get("collective_sequence"),
                    candidate_swap_bytes=int(response.get("candidate_swap_bytes", 0)),
                    persistent_baseline_violations=tuple(
                        str(value) for value in persistent_violations
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise TPGroupPoisoned(
                    "TP lifecycle control response has invalid fields"
                ) from exc
            if rank_state.rank != expected_rank:
                raise TPGroupPoisoned("TP lifecycle control rank ordering mismatch")
            states[expected_rank] = rank_state
        self._stability_last_rank_states = states
        return states

    def _stability_lifecycle_snapshot(self) -> dict[str, Any]:
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return {}
        active = coordinator.active
        return {
            "executor_identity": coordinator.executor_identity,
            "phase": coordinator.phase.value,
            "active_generation": active.generation if active else None,
            "active_request_id": active.request_id if active else None,
            "last_completed_generation": coordinator.last_completed_generation,
            "registry_record_count": (
                coordinator.registry.count(
                    generation=active.generation,
                    include_released=True,
                )
                if active is not None
                else 0
            ),
            "pending_finalization": sorted(self._stability_pending_finalization),
            "recovery_state": coordinator.recovery.state.value,
            "recovery_reason": coordinator.recovery.reason,
            "recovery_attempts": coordinator.recovery.attempts,
            "event_window_dropped": coordinator.events.dropped,
        }

    def _stability_emit_lifecycle_status(
        self, event: str, *, baseline_pass: bool | None = None
    ) -> None:
        """Write the compact, machine-readable lifecycle status contract.

        This is diagnostic-only and uses the existing bounded async telemetry
        writer.  Campaign recovery treats a missing or inconsistent status as
        unavailable; it never infers health from a best-effort log line.
        """
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return
        active = coordinator.active
        rank_states = getattr(self, "_stability_last_rank_states", {})
        ranks = {
            str(rank): {
                "executor_id": state.executor_identity,
                "last_completed_generation": state.last_completed_generation,
                "active_generation": state.active_generation,
                "active_request_count": state.active_request_count,
                "active_generation_rpc_count": state.active_generation_rpc_count,
                "unresolved_generation_future_count": (
                    state.unresolved_generation_future_count
                ),
                "owned_generation_shm_count": state.owned_generation_shm_count,
                "known_request_owned_state_count": (
                    state.known_request_owned_state_count
                ),
                "worker_alive": state.worker_alive,
                "candidate_swap_bytes": state.candidate_swap_bytes,
                "persistent_violations": list(state.persistent_baseline_violations),
            }
            for rank, state in sorted(rank_states.items())
        }
        outstanding_rpc_count = self._stability_active_rpc_snapshot()[
            "active_rpc_count"
        ]
        active_request_count = sum(
            rank_state.active_request_count for rank_state in rank_states.values()
        )
        if baseline_pass is None:
            baseline_pass = (
                coordinator.phase
                in {GenerationPhase.IDLE, GenerationPhase.CLEAN_COMPLETE}
                and active is None
                and active_request_count == 0
                and outstanding_rpc_count == 0
            )
        state = (
            "TP_GROUP_POISONED"
            if coordinator.phase is GenerationPhase.POISONED
            else coordinator.phase.value.upper()
        )
        emit_stability_telemetry(
            event,
            tp_lifecycle={
                "state": state,
                "executor_id": coordinator.executor_identity,
                "ranks": ranks,
                "baseline": {
                    "pass": baseline_pass,
                    "state": state,
                    "last_completed_generation": coordinator.last_completed_generation,
                    "active_generation": active.generation if active else None,
                    "active_request_count": active_request_count,
                    "outstanding_rpc_count": outstanding_rpc_count,
                    # Keep the same per-rank proof under baseline so an
                    # external whole-group recovery supervisor can validate
                    # it without inferring counts from a log line.
                    "ranks": ranks,
                },
            },
            lifecycle=self._stability_lifecycle_snapshot(),
        )

    def _stability_poison(self, reason: str) -> None:
        """Mark the current TP group unreusable and preserve bounded evidence."""
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return
        if coordinator.phase is not GenerationPhase.POISONED:
            coordinator.poison(reason)
        self._stability_recovery_required = True
        emit_stability_failure_window(
            "tp_group_poisoned",
            executor_identity=coordinator.executor_identity,
            generation=(coordinator.active.generation if coordinator.active else None),
            request_id=(coordinator.active.request_id if coordinator.active else None),
            reason=reason,
            lifecycle=self._stability_lifecycle_snapshot(),
            rpc_lifecycle=self._stability_active_rpc_snapshot(),
            queues=self._stability_queue_snapshots(include_slots=True),
            heartbeat=self._stability_heartbeat_snapshot(),
        )
        self._stability_emit_lifecycle_status(
            "tp_lifecycle_status", baseline_pass=False
        )

    def _stability_queue_snapshots(self, include_slots: bool = False) -> dict[str, Any]:
        if not stability_telemetry_enabled():
            return {}
        snapshots: dict[str, Any] = {}
        try:
            if self.rpc_broadcast_mq is not None:
                snapshots["rpc_broadcast"] = self.rpc_broadcast_mq.diagnostic_snapshot(
                    include_slots
                )
            for rank, message_queue in enumerate(self.response_mqs):
                snapshots[f"response_rank_{rank}"] = message_queue.diagnostic_snapshot(
                    include_slots
                )
        except Exception as exc:
            snapshots["snapshot_error"] = f"{type(exc).__name__}: {exc}"
        return snapshots

    def _stability_make_rpc_envelope(
        self,
        *,
        method_name: str,
        rpc_id: int,
        scheduler: dict[str, Any],
        control: bool,
    ) -> dict[str, Any] | None:
        """Create correlation metadata for a request-scoped TP RPC.

        The transport payload remains unchanged when the feature gate is off.
        Lifecycle control calls use generation zero and never enter a request
        ownership registry; data-plane calls require the active generation.
        """
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return None
        if control:
            return {
                "kind": "control",
                "executor_identity": coordinator.executor_identity,
                "generation": 0,
                "request_id": "__tp_lifecycle_control__",
                "rpc_id": str(rpc_id),
                "method": method_name,
            }

        active = coordinator.active
        if active is None:
            # Startup and model-global RPCs have no request-owned lifetime.
            return None
        if coordinator.phase is GenerationPhase.POISONED:
            raise TPGroupPoisoned(coordinator.poison_reason or "TP group poisoned")
        if coordinator.phase is GenerationPhase.BEGIN:
            coordinator.mark_running(active.generation)
        if coordinator.phase not in {
            GenerationPhase.RUNNING,
            GenerationPhase.COMPLETING,
            GenerationPhase.ABORTING,
        }:
            reason = (
                f"data-plane RPC {method_name} is forbidden in "
                f"{coordinator.phase.value}"
            )
            # A request-scoped RPC in a phase that has already stopped
            # dispatching work has no safe completion proof.  Do not merely
            # reject the call and leave this TP group reusable.
            self._stability_poison(reason)
            raise TPGroupPoisoned(reason)

        summary = scheduler.get("scheduler", scheduler)
        progress = summary.get("request_progress", [])
        request_ids = {
            str(item["request_id"])
            for item in progress
            if isinstance(item, dict) and item.get("request_id") is not None
        }
        if request_ids and request_ids != {active.request_id}:
            reason = (
                "lifecycle max_num_seqs=1 request mismatch: "
                f"active={active.request_id}, scheduled={sorted(request_ids)}"
            )
            self._stability_poison(reason)
            raise TPGroupPoisoned(reason)
        computed_tokens = 0
        for progress_item in progress:
            if (
                isinstance(progress_item, dict)
                and progress_item.get("request_id") == active.request_id
            ):
                computed_tokens = int(progress_item.get("num_computed_tokens", 0))
                break
        return {
            "kind": "data",
            "executor_identity": active.executor_identity,
            "generation": active.generation,
            "request_id": active.request_id,
            "rpc_id": str(rpc_id),
            "method": method_name,
            "computed_tokens": computed_tokens,
        }

    def _stability_register_lifecycle_rpc_resources(
        self, envelope: dict[str, Any] | None
    ) -> None:
        if envelope is None or envelope.get("kind") != "data":
            return
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return
        generation = int(envelope["generation"])
        request_id = str(envelope["request_id"])
        rpc_id = str(envelope["rpc_id"])
        try:
            for resource_type, prefix in (
                (ResourceType.RPC, "rpc"),
                (ResourceType.FUTURE, "future"),
                (ResourceType.SHM_MESSAGE, "shm-message"),
            ):
                coordinator.registry.register(
                    generation=generation,
                    request_id=request_id,
                    rank=0,
                    resource_type=resource_type,
                    resource_id=f"{prefix}:{rpc_id}",
                )
        except GenerationLifecycleError as exc:
            self._stability_poison(f"RPC ownership registration failed: {exc}")
            raise TPGroupPoisoned(str(exc)) from exc

    def _stability_quiesce_lifecycle_rpc_resources(
        self, envelope: dict[str, Any] | None
    ) -> None:
        if envelope is None or envelope.get("kind") != "data":
            return
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            return
        generation = int(envelope["generation"])
        rpc_id = str(envelope["rpc_id"])
        for resource_type, prefix in (
            (ResourceType.RPC, "rpc"),
            (ResourceType.FUTURE, "future"),
            (ResourceType.SHM_MESSAGE, "shm-message"),
        ):
            try:
                coordinator.registry.mark_quiescent(
                    generation=generation,
                    rank=0,
                    resource_type=resource_type,
                    resource_id=f"{prefix}:{rpc_id}",
                )
            except GenerationLifecycleError as exc:
                self._stability_poison(f"RPC ownership completion failed: {exc}")
                raise TPGroupPoisoned(str(exc)) from exc
        # All underlying parent-side handles for this RPC have completed. Keep
        # only bounded aggregate proof plus a diagnostic tail until TP commit;
        # do not release any transport or future here.
        coordinator.registry.compact_quiescent(generation, rank=0)

    def _stability_accept_response_envelope(
        self,
        response_envelope: Any,
        expected_envelope: dict[str, Any] | None,
        expected_rank: int,
    ) -> bool:
        """Accept only a structurally valid, fully correlated response.

        A same-generation response for a different RPC/method is stale, not
        accepted.  A malformed response cannot prove lifecycle completion, so
        it poisons the TP group and lets the caller retain ownership metadata.
        """
        if expected_envelope is None:
            return response_envelope is None
        if not isinstance(response_envelope, dict):
            reason = "lifecycle response omitted its envelope"
            self._stability_poison(reason)
            raise TPGroupPoisoned(reason)
        required_fields = (
            "executor_identity",
            "generation",
            "request_id",
            "rpc_id",
            "method",
            "kind",
            "rank",
        )
        if expected_envelope["kind"] == "data":
            required_fields += ("computed_tokens",)
        missing_fields = [
            field for field in required_fields if field not in response_envelope
        ]
        if missing_fields:
            reason = (
                "lifecycle response is missing correlation fields: "
                f"{','.join(missing_fields)}"
            )
            self._stability_poison(reason)
            raise TPGroupPoisoned(reason)
        invalid_text_fields = [
            field
            for field in ("executor_identity", "request_id", "rpc_id", "method", "kind")
            if not isinstance(response_envelope[field], str)
        ]
        integer_fields = ("generation", "rank")
        if expected_envelope["kind"] == "data":
            integer_fields += ("computed_tokens",)
        invalid_int_fields = [
            field
            for field in integer_fields
            if type(response_envelope[field]) is not int
            or int(response_envelope[field]) < 0
        ]
        if invalid_text_fields or invalid_int_fields:
            fields = invalid_text_fields + invalid_int_fields
            reason = "lifecycle response has malformed correlation fields: " + ",".join(
                fields
            )
            self._stability_poison(reason)
            raise TPGroupPoisoned(reason)
        expected_fields = (
            "executor_identity",
            "generation",
            "request_id",
            "rpc_id",
            "method",
            "kind",
        )
        if expected_envelope["kind"] == "data":
            expected_fields += ("computed_tokens",)
        if any(
            response_envelope.get(field) != expected_envelope[field]
            for field in expected_fields
        ):
            self._stability_record_stale_response(
                reason="response correlation does not match the expected RPC",
                expected_envelope=expected_envelope,
                response_envelope=response_envelope,
                expected_rank=expected_rank,
            )
            return False
        if response_envelope["rank"] != expected_rank:
            self._stability_record_stale_response(
                reason="response rank does not match response queue",
                expected_envelope=expected_envelope,
                response_envelope=response_envelope,
                expected_rank=expected_rank,
            )
            return False
        if expected_envelope["kind"] == "control":
            return True
        coordinator = self._stability_generation_coordinator()
        if coordinator is None:
            reason = "lifecycle response arrived without an active coordinator"
            self._stability_poison(reason)
            raise TPGroupPoisoned(reason)
        accepted = coordinator.accept_response(
            ResponseEnvelope(
                generation=response_envelope["generation"],
                request_id=response_envelope["request_id"],
                executor_identity=response_envelope["executor_identity"],
                rpc_id=response_envelope["rpc_id"],
            )
        )
        if not accepted:
            emit_stability_telemetry(
                "stale_response",
                reason="response generation is no longer active",
                expected=expected_envelope,
                observed=response_envelope,
                expected_rank=expected_rank,
            )
        return accepted

    def _stability_record_stale_response(
        self,
        *,
        reason: str,
        expected_envelope: dict[str, Any],
        response_envelope: dict[str, Any],
        expected_rank: int,
    ) -> None:
        """Retain evidence for a rejected response without accepting it.

        This helper is called only after structural envelope validation.  It
        deliberately does not call ``accept_response``: that method cannot
        distinguish a same-generation but wrong-RPC response from a valid one.
        """
        emit_stability_telemetry(
            "stale_response",
            reason=reason,
            expected=expected_envelope,
            observed=response_envelope,
            expected_rank=expected_rank,
        )
        coordinator = self._stability_generation_coordinator()
        if coordinator is not None:
            coordinator.record_stale_response(
                generation=response_envelope["generation"],
                request_id=response_envelope["request_id"],
                rpc_id=response_envelope["rpc_id"],
                rank=response_envelope["rank"],
                executor_identity=response_envelope["executor_identity"],
                reason=reason,
            )

    def _stability_active_rpc_snapshot(self) -> dict[str, Any]:
        active_rpcs = getattr(self, "_stability_active_rpcs", {})
        active_ids = sorted(active_rpcs)
        return {
            "active_rpc_count": len(active_ids),
            "active_rpc_ids": active_ids,
            "active_rpc_methods": {
                str(rpc_id): active_rpcs[rpc_id]["method"] for rpc_id in active_ids
            },
            "future_queue_depth": len(self.futures_queue),
        }

    def _stability_register_rpc(
        self,
        rpc_id: int,
        method_name: str,
        output_rank: int | None,
        non_block: bool,
        scheduler: dict[str, Any],
        lifecycle_envelope: dict[str, Any] | None = None,
    ) -> None:
        active_rpcs = getattr(self, "_stability_active_rpcs", None)
        if active_rpcs is None:
            active_rpcs = {}
            self._stability_active_rpcs = active_rpcs
        active_rpcs[rpc_id] = {
            "method": method_name,
            "output_rank": output_rank,
            "non_block": non_block,
            "scheduler": scheduler,
            "lifecycle_envelope": lifecycle_envelope,
            "enqueued_monotonic_ns": time.monotonic_ns(),
        }

    def _stability_finish_rpc(
        self,
        rpc_id: int | None,
        lifecycle_envelope: dict[str, Any] | None = None,
        *,
        quiescent: bool,
    ) -> None:
        """Retire only a response whose lifecycle envelope proved completion."""
        if rpc_id is None:
            return
        if quiescent:
            self._stability_quiesce_lifecycle_rpc_resources(lifecycle_envelope)
            getattr(self, "_stability_active_rpcs", {}).pop(rpc_id, None)
            return
        active = getattr(self, "_stability_active_rpcs", {}).get(rpc_id)
        if active is not None:
            active["unresolved"] = True
            active["last_transition_monotonic_ns"] = time.monotonic_ns()

    def record_stability_sample_context(
        self,
        scheduler_output: SchedulerOutput,
        *,
        batch_queue_depth: int,
    ) -> None:
        """Associate the next sample RPC with its originating scheduled batch."""
        if (
            stability_telemetry_enabled()
            or self._stability_generation_coordinator() is not None
        ):
            self._stability_next_sample_context = {
                "scheduler": _stability_scheduler_summary((scheduler_output,)),
                "batch_queue_depth": batch_queue_depth,
            }

    def _take_stability_sample_context(self) -> dict[str, Any] | None:
        if (
            not stability_telemetry_enabled()
            and self._stability_generation_coordinator() is None
        ):
            return None
        context = getattr(self, "_stability_next_sample_context", None)
        self._stability_next_sample_context = None
        return context

    def stability_diagnostic_snapshot(self) -> dict[str, Any]:
        """Return request-boundary transport state for the isolated campaign."""
        if (
            not stability_telemetry_enabled()
            and self._stability_generation_coordinator() is None
        ):
            return {}
        return {
            "rpc_lifecycle": self._stability_active_rpc_snapshot(),
            "queues": self._stability_queue_snapshots(include_slots=True),
            "next_sample_context_pending": (
                getattr(self, "_stability_next_sample_context", None) is not None
            ),
            "tp_generation_lifecycle": self._stability_lifecycle_snapshot(),
        }

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
        stability_context: dict[str, Any] | None = None,
        stability_control: bool = False,
    ) -> Any:
        """Returns single result if unique_reply_rank and/or kv_output_aggregator
        is provided, otherwise list."""
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        coordinator = self._stability_generation_coordinator()
        if (
            coordinator is not None
            and coordinator.phase is GenerationPhase.POISONED
            and not stability_control
        ):
            raise TPGroupPoisoned(coordinator.poison_reason or "TP group is poisoned")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}
        telemetry_enabled = stability_telemetry_enabled()
        lifecycle_enabled = coordinator is not None
        rpc_id: int | None = None
        if telemetry_enabled or lifecycle_enabled:
            rpc_id = self._stability_next_rpc_id
            self._stability_next_rpc_id += 1

        if kv_output_aggregator is not None:
            output_rank = None
            aggregate: Callable[[Any], Any] = partial(
                kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0
            )
        else:
            output_rank = unique_reply_rank
            aggregate = lambda x: x

        if isinstance(method, str):
            send_method = method
            method_name = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
            method_name = getattr(method, "__name__", type(method).__name__)
        scheduler = (
            stability_context
            if stability_context is not None
            else {"scheduler": _stability_scheduler_summary(args)}
        )
        lifecycle_envelope: dict[str, Any] | None = None
        if lifecycle_enabled:
            assert rpc_id is not None
            lifecycle_envelope = self._stability_make_rpc_envelope(
                method_name=method_name,
                rpc_id=rpc_id,
                scheduler=scheduler,
                control=stability_control,
            )
        if telemetry_enabled or lifecycle_enabled:
            assert rpc_id is not None
            self._stability_register_rpc(
                rpc_id,
                method_name,
                output_rank,
                non_block,
                scheduler,
                lifecycle_envelope,
            )
        self._stability_register_lifecycle_rpc_resources(lifecycle_envelope)
        payload: tuple[Any, ...] = (send_method, args, kwargs, output_rank)
        if rpc_id is not None:
            payload += (rpc_id,)
        if lifecycle_envelope is not None:
            payload += (lifecycle_envelope,)

        with (
            stability_lifecycle_scope(
                (int(lifecycle_envelope["generation"]) if lifecycle_envelope else None),
                (str(lifecycle_envelope["request_id"]) if lifecycle_envelope else None),
            ),
            stability_rpc_scope(rpc_id),
        ):
            if telemetry_enabled:
                emit_stability_telemetry(
                    "rpc_enqueue_begin",
                    method=method_name,
                    output_rank=output_rank,
                    timeout_s=timeout,
                    rpc_lifecycle=self._stability_active_rpc_snapshot(),
                    scheduler=scheduler,
                    queues=self._stability_queue_snapshots(),
                    lifecycle_envelope=lifecycle_envelope,
                )
            try:
                self.rpc_broadcast_mq.enqueue(payload)
            except Exception:
                uncertain = bool(
                    lifecycle_envelope and lifecycle_envelope.get("kind") == "data"
                )
                self._stability_finish_rpc(
                    rpc_id, lifecycle_envelope, quiescent=not uncertain
                )
                if uncertain:
                    self._stability_poison("request RPC enqueue outcome is unknown")
                if telemetry_enabled:
                    emit_stability_telemetry(
                        "rpc_enqueue_failure",
                        method=method_name,
                        output_rank=output_rank,
                        rpc_lifecycle=self._stability_active_rpc_snapshot(),
                        queues=self._stability_queue_snapshots(include_slots=True),
                    )
                raise
            if telemetry_enabled:
                emit_stability_telemetry(
                    "rpc_enqueue_end",
                    method=method_name,
                    output_rank=output_rank,
                    rpc_lifecycle=self._stability_active_rpc_snapshot(),
                    queues=self._stability_queue_snapshots(),
                )

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        response_ranks: Sequence[int] = tuple(range(self.world_size))
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)
            response_ranks = (output_rank,)

        def get_response():
            with (
                stability_lifecycle_scope(
                    (
                        int(lifecycle_envelope["generation"])
                        if lifecycle_envelope
                        else None
                    ),
                    (
                        str(lifecycle_envelope["request_id"])
                        if lifecycle_envelope
                        else None
                    ),
                ),
                stability_rpc_scope(rpc_id),
            ):
                # A request-owned parent RPC is quiescent only after every
                # response it waits for has passed structural, correlation,
                # queue-rank, and status validation.  In particular, never
                # let an exception while unpacking a malformed response fall
                # through the finally block as a successful completion.
                lifecycle_data = bool(
                    lifecycle_envelope and lifecycle_envelope.get("kind") == "data"
                )
                correlation_proven = not lifecycle_data
                outcome = "unresolved"
                if telemetry_enabled:
                    emit_stability_telemetry(
                        "rpc_wait_begin",
                        method=method_name,
                        output_rank=output_rank,
                        rpc_lifecycle=self._stability_active_rpc_snapshot(),
                        scheduler=scheduler,
                        queues=self._stability_queue_snapshots(),
                    )
                try:
                    responses = []
                    for expected_rank, mq in zip(response_ranks, response_mqs):
                        while True:
                            dequeue_timeout = (
                                None
                                if deadline is None
                                else max(0.0, deadline - time.monotonic())
                            )
                            try:
                                raw_response = mq.dequeue(timeout=dequeue_timeout)
                            except TimeoutError as exc:
                                outcome = "timeout"
                                if telemetry_enabled:
                                    emit_stability_telemetry(
                                        "rpc_timeout",
                                        method=method_name,
                                        output_rank=output_rank,
                                        timeout_s=timeout,
                                        rpc_lifecycle=self._stability_active_rpc_snapshot(),
                                        scheduler=scheduler,
                                        queues=self._stability_queue_snapshots(
                                            include_slots=True
                                        ),
                                    )
                                if (
                                    lifecycle_envelope
                                    and lifecycle_envelope.get("kind") == "data"
                                ):
                                    self._stability_poison(
                                        "request RPC deadline expired without a "
                                        "quiescence proof"
                                    )
                                    raise TPGroupPoisoned(
                                        f"RPC call to {method} timed out."
                                    ) from exc
                                raise TimeoutError(
                                    f"RPC call to {method} timed out."
                                ) from exc

                            response_envelope = None
                            if lifecycle_envelope is not None:
                                if not (
                                    isinstance(raw_response, tuple)
                                    and len(raw_response) == 3
                                ):
                                    outcome = "malformed_response"
                                    self._stability_poison(
                                        "lifecycle response is missing correlation "
                                        "metadata"
                                    )
                                    raise TPGroupPoisoned(
                                        "lifecycle response is malformed"
                                    )
                                status, result, response_envelope = raw_response
                                if not isinstance(status, WorkerProc.ResponseStatus):
                                    outcome = "malformed_response"
                                    reason = "lifecycle response has an invalid status"
                                    self._stability_poison(reason)
                                    raise TPGroupPoisoned(reason)
                                try:
                                    accepted = self._stability_accept_response_envelope(
                                        response_envelope,
                                        lifecycle_envelope,
                                        expected_rank,
                                    )
                                except TPGroupPoisoned:
                                    outcome = "malformed_response"
                                    raise
                                if not accepted:
                                    outcome = "stale_response"
                                    current = self._stability_generation_coordinator()
                                    if (
                                        current is not None
                                        and current.phase is GenerationPhase.POISONED
                                    ):
                                        raise TPGroupPoisoned(
                                            current.poison_reason or "TP group poisoned"
                                        )
                                    # A delayed response is retired by reading it,
                                    # but never handed to this FutureWrapper.
                                    continue
                            else:
                                status, result = raw_response

                            if status != WorkerProc.ResponseStatus.SUCCESS:
                                outcome = "worker_failure"
                                if telemetry_enabled:
                                    emit_stability_telemetry(
                                        "rpc_worker_failure",
                                        method=method_name,
                                        output_rank=output_rank,
                                        rpc_lifecycle=self._stability_active_rpc_snapshot(),
                                        scheduler=scheduler,
                                        queues=self._stability_queue_snapshots(
                                            include_slots=True
                                        ),
                                    )
                                if (
                                    lifecycle_envelope
                                    and lifecycle_envelope.get("kind") == "data"
                                ):
                                    self._stability_poison(
                                        "worker raised during request generation"
                                    )
                                    raise TPGroupPoisoned(
                                        f"Worker failed with error '{result}'"
                                    )
                                raise RuntimeError(
                                    f"Worker failed with error '{result}', please "
                                    "check the stack trace above for the root cause"
                                )
                            outcome = "response"
                            responses.append(result)
                            break
                    # The current RPC owns exactly the response set above.
                    # Mark it quiescent only once every required response has
                    # passed the complete validation path.
                    correlation_proven = True
                    outcome = "response"
                    if telemetry_enabled:
                        emit_stability_telemetry(
                            "rpc_response_received",
                            method=method_name,
                            output_rank=output_rank,
                            rpc_lifecycle=self._stability_active_rpc_snapshot(),
                            queues=self._stability_queue_snapshots(),
                        )
                    return responses[0] if output_rank is not None else responses
                finally:
                    self._stability_finish_rpc(
                        rpc_id,
                        lifecycle_envelope,
                        quiescent=(not lifecycle_data) or correlation_proven,
                    )
                    if telemetry_enabled:
                        emit_stability_telemetry(
                            "rpc_finalized",
                            method=method_name,
                            output_rank=output_rank,
                            outcome=outcome,
                            rpc_lifecycle=self._stability_active_rpc_snapshot(),
                            queues=self._stability_queue_snapshots(),
                        )

        future = FutureWrapper(
            self.futures_queue, get_response=get_response, aggregate=aggregate
        )

        return future if non_block else future.result()

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]
        initial_count = len(active_procs())

        # Give processes time to clean themselves up properly first
        logger.info(
            "[shutdown] Executor: waiting for worker exit count=%d",
            initial_count,
        )
        if wait_for_termination(
            active_procs(), timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
        ):
            logger.info_once("[shutdown] Executor: all workers exited gracefully")
            return

        # Send SIGTERM if still running
        remaining = active_procs()
        logger.warning(
            "[shutdown] Executor: workers still running after grace period; "
            "sending SIGTERM count=%d",
            len(remaining),
        )
        for p in remaining:
            p.terminate()
        if not wait_for_termination(active_procs(), 4):
            # Send SIGKILL if still running
            remaining = active_procs()
            logger.warning(
                "[shutdown] Executor: workers still running after SIGTERM; "
                "sending SIGKILL count=%d",
                len(remaining),
            )
            for p in remaining:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        watchdog_stop = getattr(self, "_stability_watchdog_stop", None)
        if watchdog_stop is not None:
            watchdog_stop.set()
        if not getattr(self, "shutting_down", False):
            worker_count = len(getattr(self, "workers", None) or [])
            logger.debug(
                "[shutdown] Executor: start worker_count=%d",
                worker_count,
            )
            self.shutting_down = True

            # Make sure all the worker processes are terminated first.
            if workers := getattr(self, "workers", None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                self._ensure_worker_termination([w.proc for w in workers])

                for w in workers:
                    # Shutdown response queues
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []

        heartbeat_queue = getattr(self, "_stability_heartbeat_queue", None)
        if heartbeat_queue is not None:
            try:
                heartbeat_queue.close()
                heartbeat_queue.join_thread()
            except (AttributeError, OSError, ValueError):
                pass
            self._stability_heartbeat_queue = None

        logger.debug_once("[shutdown] Executor: complete")

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""

    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    # The worker process writes to this MQ in single-node mode
    worker_response_mq: MessageQueue | None
    # This is only non empty on driver node,
    # the peer worker process i writes to MQ
    # `peer_worker_response_mqs[i]`
    peer_worker_response_mqs: list[MessageQueue | None]
    death_writer: Connection | None = None

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # Initialize MessageQueue for receiving SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # Initializes a message queue for sending the model output
            self.worker_response_mq = MessageQueue(
                1, 1, name=f"worker_response_rank_{self.rank}"
            )
            self.peer_response_handles = []
        else:
            # Initialize remote MessageQueue for receiving SchedulerOutput across nodes
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                # Since there is external_writer_handle from executor proc,
                # where the ready signal from actual writer is sent out of the
                # create_mq_broadcaster method and after this setup, we make it
                # non blocking. The handshake will be triggered when
                # worker.rpc_broadcast_mq.wait_until_ready() is called
                blocking=False,
            )
            # Initializes remote message queue for sending the model output to the
            # driver worker, exposing peer_response_handles for driver worker
            # that include handles for all ranks
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )

    @instrument(span_name="Worker init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        stability_executor_identity: str | None = None,
        stability_heartbeat_queue: Any | None = None,
    ):
        self.rank = rank
        configure_stability_process(f"worker-rank-{rank}")
        self._stability_async_output_rpc_ids: dict[int, int | None] = {}
        self._stability_executor_identity = stability_executor_identity
        self._stability_participant: RankLifecycleParticipant | None = None
        self._stability_heartbeat_queue = stability_heartbeat_queue
        self._stability_heartbeat_dropped = 0
        if stability_heartbeat_queue is not None:
            with suppress(AttributeError, OSError, ValueError):
                # Diagnostics must never delay worker shutdown while a parent
                # has already declared the whole group unavailable.
                stability_heartbeat_queue.cancel_join_thread()
        if stability_executor_identity is not None and stability_lifecycle_enabled():
            self._stability_participant = RankLifecycleParticipant(
                rank=rank,
                executor_identity=stability_executor_identity,
                registry_capacity=_stability_positive_int_env(
                    "VLLM_TP_LIFECYCLE_REGISTRY_CAPACITY", 1024
                ),
            )
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        # Load model
        self.worker.init_device()
        # Update process title now that parallel groups are initialized
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.worker.elastic_ep_execute("load_model")
        else:
            self.worker.load_model()

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        # Set block size based on the attention backends
        current_platform.update_block_size_for_backend(vllm_config)

        # Initialize message queues after init_device() since multi-node setups
        # (nnodes_within_dp > 1) require distributed groups to be initialized
        self._init_message_queues(input_shm_handle, vllm_config)

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        stability_executor_identity: str | None = None,
        stability_heartbeat_queue: Any | None = None,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            "stability_executor_identity": stability_executor_identity,
            "stability_heartbeat_queue": stability_heartbeat_queue,
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        # Apply NUMA binding if configured
        with numa_utils.configure_subprocess(
            vllm_config, local_rank, process_kind="worker"
        ):
            proc.start()

        # Close child ends of pipes here in the parent
        ready_writer.close()
        death_reader.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        response_handle = handles["handle"]
        worker_response_mq: MessageQueue | None = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        peer_response_handles = handles["peer_response_handles"]
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
        )

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        e = Exception(
            "WorkerProc initialization failed due to an exception in a "
            "background process. See stack trace for root cause."
        )

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
        if death_pipe is None:
            return

        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            try:
                # This will block until parent process exits (pipe closes)
                death_pipe.recv()
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        # Pass queue references directly to avoid gc issues if passing self
        Thread(
            target=death_pipe_monitor,
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),
            daemon=True,
            name="DeathPipeMonitor",
        ).start()

    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker initialization and execution loops.
        This runs a background process"""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = threading.Event()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Publish the logical-to-physical mapping early so topology helpers
        # work before init_device (needed by set_worker_net_device below).
        assigned_physical_gpu_ids = kwargs[
            "vllm_config"
        ].parallel_config.assigned_physical_gpu_ids
        if assigned_physical_gpu_ids is not None:
            from vllm.platforms.interface import set_assigned_physical_gpu_ids

            set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)

        # Set net device env vars for the worker if VLLM_GPU_NIC_PCIE_MAPPING is set
        set_worker_net_device(kwargs.get("local_rank", 0), kwargs["vllm_config"])

        worker = None
        ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        # Close inherited pipes from parent (incl. other worker pipes)
        # Explicitly passing in existing pipes and closing them makes the pipe
        # behave when using fork. Otherwise, a hidden reference to the pipes
        # exist in the child process and prevents EOF closure.
        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)

        try:
            # Initialize tracer
            rank = kwargs.get("rank", 0)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"Worker_{rank}",
            )

            worker = WorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None
            if kwargs["vllm_config"].parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(f"Worker_{worker.rank}")

            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            # Send READY once we know everything is loaded
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: exiting after shutdown request"
                )
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested.set()

        except SystemExit as e:
            # SystemExit is raised on SIGTERM or SIGKILL, which usually indicates that
            # the graceful shutdown process did not succeed
            if shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: terminated by shutdown signal"
                )
            else:
                logger.warning("WorkerProc was terminated")
            # SystemExit must never be ignored
            raise e

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def _stability_async_queue_depth(self) -> int | None:
        if not self.use_async_scheduling:
            return None
        try:
            return self.async_output_queue.qsize()
        except (NotImplementedError, OSError):
            return None

    def _stability_async_rpc_id_map_count(self) -> int:
        return len(self._stability_async_output_rpc_ids)

    @staticmethod
    def _stability_message_queue_snapshot(
        message_queue: MessageQueue | None,
    ) -> dict[str, Any] | None:
        if message_queue is None:
            return None
        try:
            return message_queue.diagnostic_snapshot()
        except Exception as exc:
            return {"snapshot_error": type(exc).__name__}

    @staticmethod
    def _stability_response_token(envelope: dict[str, Any]) -> ResponseEnvelope:
        return ResponseEnvelope(
            generation=int(envelope["generation"]),
            request_id=str(envelope["request_id"]),
            executor_identity=str(envelope["executor_identity"]),
            rpc_id=str(envelope["rpc_id"]),
        )

    def _stability_begin_worker_rpc(
        self,
        envelope: dict[str, Any] | None,
        *,
        output_expected: bool,
    ) -> bool:
        """Adopt a data-plane envelope before it can invoke model code."""
        participant = self._stability_participant
        if envelope is None or envelope.get("kind") != "data":
            return True
        if participant is None:
            return False
        try:
            token = self._stability_response_token(envelope)
            if participant.active is None:
                participant.begin(
                    GenerationToken(
                        generation=token.generation,
                        request_id=token.request_id,
                        executor_identity=token.executor_identity,
                    )
                )
            elif participant.active != GenerationToken(
                generation=token.generation,
                request_id=token.request_id,
                executor_identity=token.executor_identity,
            ):
                return False
            return participant.begin_rpc(
                token,
                future_id=(f"response:{token.rpc_id}" if output_expected else None),
                shm_message_id=f"input:{token.rpc_id}",
            )
        except GenerationLifecycleError:
            return False

    def _stability_complete_worker_rpc(
        self, envelope: dict[str, Any] | None, *, output_expected: bool
    ) -> None:
        """Mark only the local transport ownership proved quiescent."""
        participant = self._stability_participant
        if participant is None or envelope is None or envelope.get("kind") != "data":
            return
        token = self._stability_response_token(envelope)
        try:
            participant.complete_rpc(
                token,
                future_id=(f"response:{token.rpc_id}" if output_expected else None),
                shm_message_id=f"input:{token.rpc_id}",
            )
        except GenerationLifecycleError:
            # A stale/unknown local message cannot be declared quiescent.  The
            # coordinator will see its missing proof and poison the group.
            return

    def _stability_runner_state_for_fence(
        self,
    ) -> tuple[int, int, int | None, tuple[str, ...]]:
        """Observe worker request state without modifying it.

        Unknown or pending state is a failed quiescence proof, not a reason to
        clear a runner field.  This is called only by lifecycle control RPCs,
        never from the model execution hot path.
        """
        try:
            wrapped_worker = getattr(self.worker, "worker", self.worker)
            model_runner = getattr(wrapped_worker, "model_runner", None)
            snapshotter = getattr(model_runner, "_stability_runner_snapshot", None)
            if not callable(snapshotter):
                return 1, 1, None, ("runner_snapshot_unavailable",)
            snapshot = snapshotter()
            active_request_count = max(
                int(snapshot.get("active_request_count", 0)),
                int(snapshot.get("input_batch_request_count", 0)),
            )
            known_request_owned_state_count = int(
                snapshot.get("mamba_state_index_count", 0)
            )
            if snapshot.get("execute_model_state_present"):
                known_request_owned_state_count += 1
            if snapshot.get("kv_connector_output_present"):
                known_request_owned_state_count += 1
            pending = []
            for name, ready in snapshot.get("event_ready", {}).items():
                if ready is False:
                    known_request_owned_state_count += 1
                    pending.append(f"cuda_event_pending:{name}")
            for name, ready in snapshot.get("stream_ready", {}).items():
                if ready is False:
                    known_request_owned_state_count += 1
                    pending.append(f"cuda_stream_pending:{name}")
            collective = snapshot.get("tp_collective", {})
            sequence = collective.get("sequence_number")
            return (
                active_request_count,
                known_request_owned_state_count,
                (int(sequence) if sequence is not None else None),
                tuple(pending),
            )
        except Exception as exc:
            return 1, 1, None, (f"runner_snapshot_error:{type(exc).__name__}",)

    @staticmethod
    def _stability_rank_state_as_dict(state: RankFenceState) -> dict[str, Any]:
        return {
            field: getattr(state, field)
            for field in RankFenceState.__dataclass_fields__
        }

    def _stability_lifecycle_control(
        self,
        operation: str,
        generation: int,
        request_id: str | None,
        envelope: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Implement all-rank prepare/commit/fence without force cleanup."""
        del request_id
        participant = self._stability_participant
        if participant is None or envelope is None:
            raise TPGroupPoisoned("worker lacks lifecycle participant")
        if (
            envelope.get("kind") != "control"
            or envelope.get("executor_identity") != participant.executor_identity
            or int(envelope.get("generation", -1)) != 0
        ):
            raise TPGroupPoisoned("invalid TP lifecycle control envelope")

        (
            active_request_count,
            known_state_count,
            collective_sequence,
            persistent_violations,
        ) = self._stability_runner_state_for_fence()
        candidate_swap_bytes, swap_violations = _stability_candidate_swap_snapshot()
        persistent_violations = persistent_violations + swap_violations
        if operation == "prepare":
            state = participant.prepare_summary(
                generation,
                worker_alive=True,
                active_request_count=active_request_count,
                known_request_owned_state_count=known_state_count,
                collective_sequence=collective_sequence,
                candidate_swap_bytes=candidate_swap_bytes,
                persistent_baseline_violations=persistent_violations,
            )
        elif operation == "commit":
            participant.commit(generation)
            state = participant.baseline_summary(
                worker_alive=True,
                active_request_count=active_request_count,
                known_request_owned_state_count=known_state_count,
                collective_sequence=collective_sequence,
                candidate_swap_bytes=candidate_swap_bytes,
                persistent_baseline_violations=persistent_violations,
            )
        elif operation == "fence":
            if participant.active is not None:
                state = RankFenceState(
                    rank=participant.rank,
                    executor_identity=participant.executor_identity,
                    last_completed_generation=participant.last_completed_generation,
                    active_generation=participant.active.generation,
                    active_request_count=active_request_count,
                    active_generation_rpc_count=participant.registry.count(
                        generation=participant.active.generation,
                        rank=participant.rank,
                        state=ResourceState.ACTIVE,
                    ),
                    unresolved_generation_future_count=participant.registry.count(
                        generation=participant.active.generation,
                        rank=participant.rank,
                        state=ResourceState.ACTIVE,
                    ),
                    known_request_owned_state_count=known_state_count,
                    worker_alive=True,
                    collective_sequence=collective_sequence,
                    candidate_swap_bytes=candidate_swap_bytes,
                    persistent_baseline_violations=persistent_violations,
                )
            else:
                state = participant.baseline_summary(
                    worker_alive=True,
                    active_request_count=active_request_count,
                    known_request_owned_state_count=known_state_count,
                    collective_sequence=collective_sequence,
                    candidate_swap_bytes=candidate_swap_bytes,
                    persistent_baseline_violations=persistent_violations,
                )
        else:
            raise TPGroupPoisoned(f"unknown TP lifecycle control operation {operation}")
        return self._stability_rank_state_as_dict(state)

    def _stability_note_worker_progress(self, envelope: dict[str, Any] | None) -> None:
        """Publish one rank heartbeat without running a TP collective."""
        participant = self._stability_participant
        if (
            participant is None
            or participant.active is None
            or envelope is None
            or envelope.get("kind") != "data"
        ):
            return
        channel = getattr(self, "_stability_heartbeat_queue", None)
        if channel is None:
            return
        try:
            computed_tokens = int(envelope.get("computed_tokens", 0))
        except (TypeError, ValueError):
            self._stability_heartbeat_dropped += 1
            return
        if computed_tokens < 0:
            self._stability_heartbeat_dropped += 1
            return
        heartbeat = {
            "executor_identity": participant.executor_identity,
            "generation": participant.active.generation,
            "request_id": participant.active.request_id,
            "rank": self.rank,
            "phase": participant.phase.value,
            "computed_tokens": computed_tokens,
            "timestamp": time.monotonic(),
            "rpc_id": str(envelope.get("rpc_id", "")) or None,
        }
        try:
            channel.put_nowait(heartbeat)
        except Exception as exc:
            # The heartbeat channel is intentionally lossy under pressure.  A
            # full or unavailable diagnostic queue cannot block model work or
            # influence the RPC deadline.
            self._stability_heartbeat_dropped += 1
            emit_stability_telemetry(
                "worker_heartbeat_dropped",
                rank=self.rank,
                executor_identity=participant.executor_identity,
                generation=participant.active.generation,
                request_id=participant.active.request_id,
                dropped_total=self._stability_heartbeat_dropped,
                exception_type=type(exc).__name__,
            )
            return
        emit_stability_telemetry(
            "worker_generation_heartbeat",
            rank=self.rank,
            executor_identity=participant.executor_identity,
            generation=participant.active.generation,
            request_id=participant.active.request_id,
            phase=participant.phase.value,
            computed_tokens=computed_tokens,
            dropped_total=self._stability_heartbeat_dropped,
        )

    def enqueue_output(
        self,
        output: Any,
        rpc_id: int | None = None,
        lifecycle_envelope: dict[str, Any] | None = None,
    ):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        with (
            stability_lifecycle_scope(
                (int(lifecycle_envelope["generation"]) if lifecycle_envelope else None),
                (str(lifecycle_envelope["request_id"]) if lifecycle_envelope else None),
            ),
            stability_rpc_scope(rpc_id),
        ):
            if stability_telemetry_enabled():
                emit_stability_telemetry(
                    "worker_response_enqueue_begin",
                    rank=self.rank,
                    async_output=isinstance(output, AsyncModelRunnerOutput),
                    async_queue_depth=self._stability_async_queue_depth(),
                    async_rpc_id_map_count=self._stability_async_rpc_id_map_count(),
                )
            if isinstance(output, AsyncModelRunnerOutput):
                try:
                    output = output.get_output()
                except Exception as exc:
                    logger.exception("Error getting async model runner output")
                    output = exc

            if isinstance(output, Exception):
                result: tuple[Any, ...] = (
                    WorkerProc.ResponseStatus.FAILURE,
                    str(output),
                )
            else:
                result = (WorkerProc.ResponseStatus.SUCCESS, output)
            if lifecycle_envelope is not None:
                result += ({**lifecycle_envelope, "rank": self.rank},)
            if (response_mq := self.worker_response_mq) is not None:
                response_mq.enqueue(result)
                self._stability_complete_worker_rpc(
                    lifecycle_envelope, output_expected=True
                )
            if stability_telemetry_enabled():
                emit_stability_telemetry(
                    "worker_response_enqueue_end",
                    rank=self.rank,
                    async_queue_depth=self._stability_async_queue_depth(),
                    async_rpc_id_map_count=self._stability_async_rpc_id_map_count(),
                    response_queue=self._stability_message_queue_snapshot(response_mq),
                )

    def handle_output(
        self,
        output: Any,
        rpc_id: int | None = None,
        lifecycle_envelope: dict[str, Any] | None = None,
    ):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            if lifecycle_envelope is not None:
                # Do not recover correlation from id(output): a delayed async
                # object must carry its generation with it until response MQ
                # enqueue.  The legacy map remains only for the feature-off
                # runtime path.
                self.async_output_queue.put((output, rpc_id, lifecycle_envelope))
                return
            if stability_telemetry_enabled():
                self._stability_async_output_rpc_ids[id(output)] = rpc_id
                emit_stability_telemetry(
                    "worker_async_output_queued",
                    rank=self.rank,
                    async_queue_depth=self._stability_async_queue_depth(),
                    async_rpc_id_map_count=self._stability_async_rpc_id_map_count(),
                )
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output, rpc_id, lifecycle_envelope)

    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""

        # set device to the worker device for the thread.
        # a thread will not inherit the context of the main thread.
        # when calling any cuda runtime functions, it will implicitly
        # create a new cuda context on device 0, consuming extra memory.
        # here we set the device to the worker device for the thread,
        # enforcing the context to be the same as the main thread.
        from vllm.platforms import current_platform

        if hasattr(self.worker, "device"):
            current_platform.set_device(self.worker.device)

        while True:
            output = self.async_output_queue.get()
            self._handle_async_output(output)

    def _handle_async_output(self, output: Any) -> None:
        if (
            isinstance(output, tuple)
            and len(output) == 3
            and isinstance(output[2], dict)
        ):
            async_output, rpc_id, lifecycle_envelope = output
            with (
                stability_lifecycle_scope(
                    int(lifecycle_envelope["generation"]),
                    str(lifecycle_envelope["request_id"]),
                ),
                stability_rpc_scope(rpc_id),
            ):
                self.enqueue_output(async_output, rpc_id, lifecycle_envelope)
            return
        rpc_id = self._stability_async_output_rpc_ids.pop(id(output), None)
        with stability_rpc_scope(rpc_id):
            if stability_telemetry_enabled():
                emit_stability_telemetry(
                    "worker_async_output_dequeued",
                    rank=self.rank,
                    async_queue_depth=self._stability_async_queue_depth(),
                    async_rpc_id_map_count=self._stability_async_rpc_id_map_count(),
                )
            self.enqueue_output(output, rpc_id)

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None
        while True:
            payload = self.rpc_broadcast_mq.dequeue(indefinite=True)
            lifecycle_envelope: dict[str, Any] | None = None
            if len(payload) == 6:
                method, args, kwargs, output_rank, rpc_id, lifecycle_envelope = payload
            elif len(payload) == 5:
                method, args, kwargs, output_rank, rpc_id = payload
            else:
                method, args, kwargs, output_rank = payload
                rpc_id = None
            method_name = (
                method if isinstance(method, str) else "serialized_worker_callable"
            )
            with (
                stability_lifecycle_scope(
                    (
                        int(lifecycle_envelope["generation"])
                        if lifecycle_envelope
                        else None
                    ),
                    (
                        str(lifecycle_envelope["request_id"])
                        if lifecycle_envelope
                        else None
                    ),
                ),
                stability_rpc_scope(rpc_id),
            ):
                if stability_telemetry_enabled():
                    emit_stability_telemetry(
                        "worker_rpc_dequeued",
                        rank=self.rank,
                        method=method_name,
                        output_rank=output_rank,
                        use_async_scheduling=self.use_async_scheduling,
                        async_queue_depth=self._stability_async_queue_depth(),
                        input_queue=self._stability_message_queue_snapshot(
                            self.rpc_broadcast_mq
                        ),
                    )
                try:
                    output_expected = output_rank is None or self.rank == output_rank
                    if method == _STABILITY_CONTROL_RPC:
                        output = self._stability_lifecycle_control(
                            *args, lifecycle_envelope
                        )
                    else:
                        if not self._stability_begin_worker_rpc(
                            lifecycle_envelope, output_expected=output_expected
                        ):
                            raise TPGroupPoisoned(
                                "worker rejected stale or inconsistent "
                                "generation envelope"
                            )
                        self._stability_note_worker_progress(lifecycle_envelope)
                        if isinstance(method, str):
                            func = getattr(self.worker, method)
                        elif isinstance(method, bytes):
                            func = partial(cloudpickle.loads(method), self.worker)

                        if stability_telemetry_enabled():
                            emit_stability_telemetry(
                                "worker_rpc_begin",
                                rank=self.rank,
                                method=method_name,
                                output_rank=output_rank,
                                async_queue_depth=self._stability_async_queue_depth(),
                            )
                        output = func(*args, **kwargs)
                        self._stability_note_worker_progress(lifecycle_envelope)

                    if stability_telemetry_enabled():
                        emit_stability_telemetry(
                            "worker_rpc_function_return",
                            rank=self.rank,
                            method=method_name,
                            output_rank=output_rank,
                            async_output=isinstance(output, AsyncModelRunnerOutput),
                            async_queue_depth=self._stability_async_queue_depth(),
                        )

                    if output_expected:
                        if lifecycle_envelope is None:
                            self.handle_output(output, rpc_id)
                        else:
                            self.handle_output(output, rpc_id, lifecycle_envelope)
                    else:
                        self._stability_complete_worker_rpc(
                            lifecycle_envelope, output_expected=False
                        )
                    continue

                except Exception as exc:
                    # Notes have been introduced in python 3.11
                    if hasattr(exc, "add_note"):
                        exc.add_note(traceback.format_exc())
                    logger.exception("WorkerProc hit an exception.")
                    if stability_telemetry_enabled():
                        emit_stability_telemetry(
                            "worker_rpc_exception",
                            rank=self.rank,
                            method=method_name,
                            output_rank=output_rank,
                            exception_type=type(exc).__name__,
                        )
                    # exception might not be serializable, so we convert it to
                    # string, only for logging purpose.
                    if output_rank is None or self.rank == output_rank:
                        if lifecycle_envelope is None:
                            self.handle_output(exc, rpc_id)
                        else:
                            self.handle_output(exc, rpc_id, lifecycle_envelope)

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        # Check if parallel groups are initialized first
        if not model_parallel_is_initialized():
            # Parallel groups not yet initialized, use default process name
            set_process_title(name="Worker")
            decorate_logs("Worker")
            return

        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if pcp_size > 1:
            process_name += f"_PCP{pcp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if dcp_size > 1:
            process_name += f"_DCP{dcp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        set_process_title(name=process_name)
        decorate_logs(process_name)


def set_multiprocessing_worker_envs(local_world_size: int = 1):
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    _maybe_force_spawn()

    if current_platform.is_cpu() or "OMP_NUM_THREADS" in os.environ:
        return

    # Choose the workers' thread count here, before they start, since a worker
    # must not set its own: `torch.set_num_threads()` spawns the thread pool
    # eagerly, and doing that part way through a worker's startup either races
    # the dlopen of shared objects or, in a forked worker, deadlocks (libgomp
    # is not fork-safe).
    num_threads = startup_omp_num_threads(local_world_size)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ[OMP_NUM_THREADS_SET_BY_VLLM] = "1"

    # A spawned worker picks the count up from the environment when it imports
    # torch. A forked worker instead inherits it from this process, so set it
    # here too. This is safe as long as we don't *use* the pool before forking:
    # a forked child whose parent had run a parallel region deadlocks, whereas
    # one whose parent merely sized the pool does not.
    torch.set_num_threads(num_threads)
    logger.debug(
        "Set OMP_NUM_THREADS=%d for %d worker process(es).",
        num_threads,
        local_world_size,
    )
