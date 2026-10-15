"""Core data structures and resource models for multi-accelerator scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class PackingStrategy(str, Enum):
    """Strategy for distributing workloads across compute instances."""
    BALANCE = "balance"          # Water-filling: equalize remaining quota / usage across instances (DEFAULT)
    CONCENTRATE = "concentrate"  # Minimal instances: pack onto fewest nodes to keep instances untouched


class ObjectiveMode(str, Enum):
    """Optimization objective for scheduling."""
    COMPLETE_ALL = "complete_all"              # Mandatory all jobs; minimize makespan
    MAXIMIZE_WORK = "maximize_work"            # Maximize priority-weighted work before quota/deadline
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"  # Maximize work done per billed quota hour


@dataclass
class Job:
    """A computational unit of work to be scheduled on an accelerator."""
    job_id: str
    duration_h: float
    vram_gb: float = 0.5
    ram_gb: float = 1.5
    cpu_cores: float = 1.0
    device_count: int = 1
    priority: float = 1.0
    resumable: bool = False
    segments: Optional[List[float]] = None     # Optional checkpoint segments (hours per segment)
    hardware_class: Optional[str] = None      # Target hardware class requirement (e.g. "nvidia_t4")
    depends_on: Optional[str] = None          # Prerequisite job_id that must complete before this job
    parent_job_id: Optional[str] = None       # ID of original unsegmented job
    segment_index: Optional[int] = None       # 0-indexed sequence position for checkpoint segments
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.duration_h, (int, float)) or not math.isfinite(self.duration_h) or self.duration_h <= 0:
            raise ValueError(f"Job duration must be a finite positive number, got {self.duration_h}")
        for name, val in [("vram_gb", self.vram_gb), ("ram_gb", self.ram_gb), ("cpu_cores", self.cpu_cores)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be a finite non-negative number, got {val}")
        if not isinstance(self.priority, (int, float)) or not math.isfinite(self.priority) or self.priority <= 0:
            raise ValueError(f"Job priority must be a finite positive number, got {self.priority}")
        if self.device_count != 1:
            raise NotImplementedError("Multi-accelerator single-job gang scheduling is not yet supported.")
        if self.segments is not None:
            if not self.segments:
                raise ValueError("Job segments list cannot be empty if specified.")
            if any(not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 for s in self.segments):
                raise ValueError(f"All segments must be finite numbers > 0 hours, got {self.segments}")
            seg_sum = sum(self.segments)
            if not math.isclose(seg_sum, self.duration_h, rel_tol=1e-2, abs_tol=1e-2):
                raise ValueError(
                    f"Sum of segment durations ({seg_sum:.3f}h) must match duration_h ({self.duration_h:.3f}h)"
                )


@dataclass
class Instance:
    """A compute node host providing multi-accelerator devices with session & quota boundaries.
    
    Attributes:
        instance_id: Unique instance identifier.
        account_id: Account or quota pool identifier. Multiple instances can share the same
                    account_id to share a single quota balance. Defaults to instance_id.
        device_count: Number of physical accelerator devices.
        remaining_quota_h: Remaining quota in hours.
        max_session_h: Maximum duration of a single execution session.
        vram_per_device_gb: VRAM capacity per physical device.
        host_ram_gb: Host physical memory capacity.
        host_cpu_cores: Available host vCPUs / CPU cores.
        slots_per_device: Maximum concurrent worker execution streams per physical device.
        debit_rate: Quota debit multiplier per container wall-clock hour.
        reset_at: Optional ISO 8601 timestamp string for quota refresh.
        must_finish_before_reset: If True, jobs must complete before reset_at deadline.
        hardware_class: Hardware accelerator class (e.g. "nvidia_t4", "nvidia_p100").
        concurrency_slowdown_factors: Mapping from active workers per device to duration slowdown multiplier.
        max_cpu_oversubscription_ratio: Optional hard cutoff for CPU oversubscription. None allows soft penalty.
    """
    instance_id: str
    account_id: Optional[str] = None
    device_count: int = 2
    remaining_quota_h: float = 30.0
    max_session_h: float = 11.0
    vram_per_device_gb: float = 16.0
    host_ram_gb: float = 30.0
    host_cpu_cores: float = 4.0
    slots_per_device: int = 1
    debit_rate: float = 1.0
    reset_at: Optional[str] = None
    must_finish_before_reset: bool = False
    hardware_class: str = "nvidia_t4"
    concurrency_slowdown_factors: Dict[int, float] = field(
        default_factory=lambda: {1: 1.0, 2: 1.30, 3: 1.65, 4: 2.0}
    )
    max_cpu_oversubscription_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.account_id:
            self.account_id = self.instance_id
        for name, val in [
            ("remaining_quota_h", self.remaining_quota_h),
            ("max_session_h", self.max_session_h),
            ("vram_per_device_gb", self.vram_per_device_gb),
            ("host_ram_gb", self.host_ram_gb),
            ("host_cpu_cores", self.host_cpu_cores),
            ("debit_rate", self.debit_rate),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val}")
        if self.device_count <= 0:
            raise ValueError(f"device_count must be > 0, got {self.device_count}")
        if self.max_session_h <= 0:
            raise ValueError(f"max_session_h must be > 0, got {self.max_session_h}")
        if self.slots_per_device <= 0:
            raise ValueError(f"slots_per_device must be > 0, got {self.slots_per_device}")
        if self.debit_rate <= 0:
            raise ValueError(f"debit_rate must be > 0, got {self.debit_rate}")

        # Normalize and validate concurrency_slowdown_factors (coerce string keys from JSON to int)
        normalized_factors: Dict[int, float] = {}
        for k, v in self.concurrency_slowdown_factors.items():
            try:
                ik = int(k)
                fv = float(v)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid concurrency slowdown factor key/value ({k}: {v}): {e}")
            if ik <= 0:
                raise ValueError(f"Worker count in slowdown factors must be > 0, got {ik}")
            if not math.isfinite(fv) or fv <= 0:
                raise ValueError(f"Slowdown factor for {ik} workers must be finite and > 0, got {fv}")
            normalized_factors[ik] = fv
        self.concurrency_slowdown_factors = normalized_factors

    @property
    def max_concurrent_workers(self) -> int:
        return self.device_count * self.slots_per_device

    def get_concurrency_slowdown(self, concurrent_workers: int) -> float:
        """Returns the duration slowdown factor for concurrent workers on the same physical device."""
        if concurrent_workers <= 1:
            return 1.0
        if concurrent_workers in self.concurrency_slowdown_factors:
            return self.concurrency_slowdown_factors[concurrent_workers]
        # Conservative linear interpolation fallback
        max_known = max(self.concurrency_slowdown_factors.keys(), default=1)
        base = self.concurrency_slowdown_factors.get(max_known, 1.0 + 0.3 * (max_known - 1))
        return base + 0.35 * (concurrent_workers - max_known)


@dataclass
class DeviceSlot:
    """An execution slot on an accelerator device, supporting multi-job concurrent streams."""
    instance_id: str
    device_index: int
    slot_index: int = 0
    load_h: float = 0.0  # Base worker-hours assigned to this slot
    jobs: List[Job] = field(default_factory=list)

    def can_fit_session(self, job: Job, max_session_h: float, slowdown: float = 1.0) -> bool:
        return (self.load_h + job.duration_h) * slowdown <= max_session_h

    def assign(self, job: Job) -> None:
        self.jobs.append(job)
        self.load_h += job.duration_h

    def effective_load_h(self, slowdown: float = 1.0) -> float:
        """Returns effective wall-clock duration accounting for concurrency slowdown."""
        return self.load_h * slowdown

    @property
    def peak_vram_gb(self) -> float:
        return max((j.vram_gb for j in self.jobs), default=0.0)

    @property
    def peak_ram_gb(self) -> float:
        return max((j.ram_gb for j in self.jobs), default=0.0)

    @property
    def peak_cpu_cores(self) -> float:
        return max((j.cpu_cores for j in self.jobs), default=0.0)


@dataclass
class InstancePlan:
    """Scheduled workload and metrics for a single compute instance."""
    instance: Instance
    slots: List[DeviceSlot]
    setup_overhead_h: float = 0.10

    @property
    def is_active(self) -> bool:
        return any(len(s.jobs) > 0 for s in self.slots)

    @property
    def total_work_h(self) -> float:
        """Total unscaled worker-hours of computational work across all slots."""
        return sum(s.load_h for s in self.slots)

    def active_workers_on_device(self, device_index: int) -> int:
        """Number of active concurrent worker slots populated on this physical device."""
        return sum(1 for s in self.slots if s.device_index == device_index and len(s.jobs) > 0)

    @property
    def device_slowdowns(self) -> Dict[int, float]:
        """Slowdown factor applied to each physical device based on its concurrent worker count."""
        slowdowns: Dict[int, float] = {}
        for d in range(self.instance.device_count):
            workers = self.active_workers_on_device(d)
            slowdowns[d] = self.instance.get_concurrency_slowdown(workers)
        return slowdowns

    @property
    def device_loads_h(self) -> Dict[int, float]:
        """Calculates effective wall-clock load per physical device across its concurrent execution slots,
        accounting for concurrency slowdown when multiple worker slots run on the same physical device.
        """
        loads: Dict[int, float] = {d: 0.0 for d in range(self.instance.device_count)}
        slowdowns = self.device_slowdowns
        for s in self.slots:
            s_slowdown = slowdowns.get(s.device_index, 1.0)
            effective_slot_load = s.effective_load_h(s_slowdown)
            loads[s.device_index] = max(loads.get(s.device_index, 0.0), effective_slot_load)
        return loads

    @property
    def max_device_load_h(self) -> float:
        if not self.slots:
            return 0.0
        return max(self.device_loads_h.values(), default=0.0)

    @property
    def makespan_h(self) -> float:
        if not self.is_active:
            return 0.0
        return self.max_device_load_h + self.setup_overhead_h

    @property
    def billed_resource_h(self) -> float:
        if not self.is_active:
            return 0.0
        # Configurable quota debit semantics:
        # Defaults to debit_rate * makespan_h (e.g. 1.0 * wall-clock session hours on Kaggle,
        # or 2.0 * makespan_h if debiting per physical device-hour).
        return self.instance.debit_rate * self.makespan_h

    @property
    def load_imbalance_h(self) -> float:
        dev_loads = list(self.device_loads_h.values())
        if len(dev_loads) <= 1:
            return 0.0
        return max(dev_loads) - min(dev_loads)

    @property
    def quota_remaining_after_h(self) -> float:
        return max(0.0, self.instance.remaining_quota_h - self.billed_resource_h)

    @property
    def peak_concurrent_workers(self) -> int:
        return sum(1 for s in self.slots if len(s.jobs) > 0)

    @property
    def peak_vram_per_device_gb(self) -> Dict[int, float]:
        vram_map: Dict[int, float] = {d: 0.0 for d in range(self.instance.device_count)}
        for s in self.slots:
            vram_map[s.device_index] += s.peak_vram_gb
        return vram_map

    @property
    def peak_host_ram_gb(self) -> float:
        return sum(s.peak_ram_gb for s in self.slots)

    @property
    def peak_host_cpu_cores(self) -> float:
        """Estimated concurrent CPU demand across all active worker slots on this instance."""
        return sum(s.peak_cpu_cores for s in self.slots if len(s.jobs) > 0)

    @property
    def cpu_oversubscription_ratio(self) -> float:
        """Ratio of peak aggregate CPU demand to available physical host vCPUs."""
        if self.instance.host_cpu_cores <= 0:
            return 1.0
        return self.peak_host_cpu_cores / self.instance.host_cpu_cores

    @property
    def is_cpu_oversubscribed(self) -> bool:
        """True if estimated concurrent CPU demand exceeds nominal host vCPU capacity."""
        return self.peak_host_cpu_cores > self.instance.host_cpu_cores

    @property
    def cpu_contention_penalty(self) -> float:
        """Soft penalty cost for CPU oversubscription: max(0, cpu_ratio - 1) ** 2."""
        return max(0.0, self.cpu_oversubscription_ratio - 1.0) ** 2


@dataclass
class ExecutionPlan:
    """Global schedule outcome across all instances and jobs."""
    instance_plans: List[InstancePlan]
    unassigned_jobs: List[Job] = field(default_factory=list)

    @property
    def active_instances(self) -> List[InstancePlan]:
        return [ip for ip in self.instance_plans if ip.is_active]

    @property
    def untouched_instances(self) -> List[InstancePlan]:
        return [ip for ip in self.instance_plans if not ip.is_active]

    @property
    def total_work_h(self) -> float:
        """Total useful worker-hours of computational work planned."""
        return sum(ip.total_work_h for ip in self.instance_plans)

    @property
    def total_billed_resource_h(self) -> float:
        """Total billed quota-hours debited across all active instances."""
        return sum(ip.billed_resource_h for ip in self.instance_plans)

    def instance_schedule_timings(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculates dependency-aware start and finish wall-clock offsets for all active instances.
        
        If instance B contains a job that depends on a job on instance A, instance B cannot
        commence execution until instance A finishes its session and persists checkpoints.
        
        Returns:
            (start_times_map, finish_times_map) in hours relative to t=0.
        """
        if not self.active_instances:
            return {}, {}

        # Map each job to its host instance_id
        job_to_inst: Dict[str, str] = {}
        for ip in self.active_instances:
            for s in ip.slots:
                for j in s.jobs:
                    job_to_inst[j.job_id] = ip.instance.instance_id

        # Directed graph of instance dependencies: inst_b -> set of inst_a that must finish first
        inst_prereqs: Dict[str, Set[str]] = {ip.instance.instance_id: set() for ip in self.active_instances}
        for ip in self.active_instances:
            b_id = ip.instance.instance_id
            for s in ip.slots:
                for j in s.jobs:
                    if j.depends_on and j.depends_on in job_to_inst:
                        a_id = job_to_inst[j.depends_on]
                        if a_id != b_id:
                            inst_prereqs[b_id].add(a_id)
            prereq_inst_id = ip.instance.metadata.get("prereq_instance_id")
            if prereq_inst_id and prereq_inst_id in inst_prereqs:
                inst_prereqs[b_id].add(prereq_inst_id)

        start_times: Dict[str, float] = {}
        finish_times: Dict[str, float] = {}
        makespans = {ip.instance.instance_id: ip.makespan_h for ip in self.active_instances}

        remaining = set(inst_prereqs.keys())
        while remaining:
            ready = [i for i in remaining if inst_prereqs[i].issubset(start_times.keys())]
            if not ready:
                cycle_instances = sorted(list(remaining))
                raise ValueError(
                    f"Instance dependency cycle detected among instances: {cycle_instances}. "
                    f"Cross-instance job dependencies induce an unresolvable multi-session deadlock."
                )

            for i in ready:
                t_start = max([finish_times[p] for p in inst_prereqs[i]], default=0.0)
                start_times[i] = t_start
                finish_times[i] = t_start + makespans[i]
                remaining.remove(i)

        return start_times, finish_times

    @property
    def global_makespan_h(self) -> float:
        """Global elapsed wall-clock critical path makespan across all active instances."""
        if not self.active_instances:
            return 0.0
        _, finish_times = self.instance_schedule_timings()
        if finish_times:
            return max(finish_times.values(), default=0.0)
        return max(ip.makespan_h for ip in self.active_instances)

    @property
    def quota_productivity_ratio(self) -> float:
        """Useful worker-hours completed per billed quota-hour (unclipped, can exceed 1.0 with concurrency)."""
        if self.total_billed_resource_h <= 0:
            return 0.0
        return self.total_work_h / self.total_billed_resource_h

    @property
    def efficiency_ratio(self) -> float:
        """Unclipped productivity ratio (useful worker-hours / billed quota-hours)."""
        return self.quota_productivity_ratio

    def account_billed_quota(self) -> Dict[str, float]:
        """Aggregate billed quota hours grouped by account_id."""
        accts: Dict[str, float] = {}
        for ip in self.active_instances:
            acc_id = ip.instance.account_id or ip.instance.instance_id
            accts[acc_id] = accts.get(acc_id, 0.0) + ip.billed_resource_h
        return accts

    def account_declared_quota(self) -> Dict[str, float]:
        """Aggregate authoritative declared quota per account_id across all configured instances.
        Uses conservative minimum declared quota across all instances in that account,
        strictly eliminating list-order dependence for conflicting declarations.
        """
        pools: Dict[str, float] = {}
        for ip in self.instance_plans:
            acc_id = ip.instance.account_id or ip.instance.instance_id
            quota = ip.instance.remaining_quota_h
            if acc_id not in pools:
                pools[acc_id] = quota
            else:
                pools[acc_id] = min(pools[acc_id], quota)
        return pools

    def account_remaining_quota(self) -> Dict[str, float]:
        """Aggregate remaining quota hours after all planned active instance debits, grouped by account_id.
        Authoritative pool quota uses conservative minimum declared quota across all instances in that account,
        strictly eliminating list-order dependence for conflicting declarations.
        """
        acct_billed = self.account_billed_quota()
        declared_pools = self.account_declared_quota()
        return {
            acc_id: max(0.0, pool_quota - acct_billed.get(acc_id, 0.0))
            for acc_id, pool_quota in declared_pools.items()
        }

    def account_overdraft_quota(self) -> Dict[str, float]:
        """Aggregate overdraft quota hours (billed beyond declared quota), grouped by account_id."""
        acct_billed = self.account_billed_quota()
        declared_pools = self.account_declared_quota()
        return {
            acc_id: max(0.0, billed - declared_pools.get(acc_id, 0.0))
            for acc_id, billed in acct_billed.items()
        }

    def summary(self) -> str:
        start_times, finish_times = self.instance_schedule_timings()
        acct_rem_map = self.account_remaining_quota()
        overdraft_map = self.account_overdraft_quota()
        lines = [
            "=" * 70,
            "ACCELPACK WORKLOAD SCHEDULE SUMMARY",
            f"Total Jobs: {sum(len(s.jobs) for ip in self.instance_plans for s in ip.slots)}",
            f"Active Instances: {len(self.active_instances)} / {len(self.instance_plans)} "
            f"({len(self.untouched_instances)} preserved untouched)",
            f"Total Work: {self.total_work_h:.2f} worker-hours",
            f"Total Billed Resource Time: {self.total_billed_resource_h:.2f} quota-hours",
            f"Quota Productivity: {self.quota_productivity_ratio * 100:.1f}% "
            f"({self.quota_productivity_ratio:.2f} worker-hours / billed quota-hour)",
            f"Global Makespan: {self.global_makespan_h:.2f} hours",
            "-" * 70,
        ]

        # Account-level quota summary
        acct_billed = self.account_billed_quota()
        declared_pools = self.account_declared_quota()
        if len(acct_billed) > 1 or any(ip.instance.account_id != ip.instance.instance_id for ip in self.instance_plans):
            lines.append("ACCOUNT QUOTA POOLS:")
            for acc_id, billed in acct_billed.items():
                pool_quota = declared_pools.get(acc_id, 0.0)
                rem_after = max(0.0, pool_quota - billed)
                od_note = f" (+{overdraft_map[acc_id]:.2f}h Tail Overdraft)" if overdraft_map.get(acc_id, 0.0) > 0 else ""
                lines.append(f"  * Account [{acc_id}]: Billed {billed:.2f}h | Remaining After {rem_after:.2f}h / {pool_quota:.2f}h{od_note}")
            lines.append("-" * 70)

        for ip in self.active_instances:
            acc = ip.instance
            inst_id = acc.instance_id
            acc_id = acc.account_id or inst_id
            t_start = start_times.get(inst_id, 0.0)
            t_finish = finish_times.get(inst_id, ip.makespan_h)
            timing_info = f"Makespan: {ip.makespan_h:.2f}h"
            if t_start > 0:
                timing_info = f"Makespan: {ip.makespan_h:.2f}h (Start: +{t_start:.2f}h, Finish: {t_finish:.2f}h)"

            cpu_status = f"{ip.peak_host_cpu_cores:.1f}/{acc.host_cpu_cores:.1f} cores ({ip.cpu_oversubscription_ratio * 100:.0f}%)"
            if ip.is_cpu_oversubscribed:
                cpu_status += " [OVERSUBSCRIBED - Soft Penalty Applied]"

            shared_quota_rem = acct_rem_map.get(acc_id, ip.quota_remaining_after_h)
            od_note = f" (+{overdraft_map[acc_id]:.2f}h Overdraft)" if overdraft_map.get(acc_id, 0.0) > 0 else ""
            lines.append(
                f"[{acc.instance_id} (Account: {acc.account_id})] {timing_info} | "
                f"Billed: {ip.billed_resource_h:.2f}h | "
                f"Account Quota Remaining: {shared_quota_rem:.2f}h{od_note} | "
                f"Workers: {ip.peak_concurrent_workers}/{acc.max_concurrent_workers}"
            )
            lines.append(f"  CPU Demand: {cpu_status}")

            for d in range(acc.device_count):
                d_slots = [s for s in ip.slots if s.device_index == d]
                d_load = ip.device_loads_h.get(d, 0.0)
                d_vram = ip.peak_vram_per_device_gb.get(d, 0.0)
                d_slowdown = ip.device_slowdowns.get(d, 1.0)
                d_workers = ip.active_workers_on_device(d)
                slowdown_label = f" (Slowdown: {d_slowdown:.2f}x for {d_workers} workers)" if d_workers > 1 else ""
                lines.append(
                    f"  ├── Device {d} (Wall-load: {d_load:.2f}h{slowdown_label}, "
                    f"Peak VRAM: {d_vram:.2f}/{acc.vram_per_device_gb} GB):"
                )
                for s in d_slots:
                    slot_label = f"Slot {s.slot_index}" if acc.slots_per_device > 1 else "Queue"
                    if not s.jobs:
                        lines.append(f"  │     [{slot_label}] (idle)")
                    else:
                        j_titles = ", ".join(f"{j.job_id} ({j.duration_h:.2f}h)" for j in s.jobs)
                        lines.append(f"  │     [{slot_label}] {j_titles}")

        if self.untouched_instances:
            lines.append("-" * 70)
            preserved_strs = [
                f"{ip.instance.instance_id} ({ip.instance.remaining_quota_h:.1f}h)"
                for ip in self.untouched_instances
            ]
            lines.append(f"Preserved Instances (Untouched): {', '.join(preserved_strs)}")

        if self.unassigned_jobs:
            lines.append("-" * 70)
            lines.append(f"WARNING: {len(self.unassigned_jobs)} UNASSIGNED JOBS (Workload could not fit!)")
            for j in self.unassigned_jobs:
                reason = j.metadata.get("unassigned_reason", "Quota or session limits exceeded")
                lines.append(f"  * {j.job_id} ({j.duration_h:.2f}h) - Reason: {reason}")

        lines.append("=" * 70)
        return "\n".join(lines)
