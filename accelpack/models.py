"""Core data structures and resource models for multi-accelerator scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.duration_h <= 0:
            raise ValueError(f"Job duration must be > 0, got {self.duration_h}")
        if self.device_count != 1:
            raise NotImplementedError("Multi-accelerator single-job gang scheduling is not yet supported.")


@dataclass
class Instance:
    """A compute node host providing multi-accelerator devices with session & quota boundaries."""
    instance_id: str
    device_count: int = 2
    remaining_quota_h: float = 30.0
    max_session_h: float = 11.0
    vram_per_device_gb: float = 16.0
    host_ram_gb: float = 30.0
    host_cpu_cores: float = 4.0
    slots_per_device: int = 1       # Number of concurrent worker execution streams per physical device
    debit_rate: float = 1.0         # Quota debit multiplier per container wall-clock hour
    reset_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.device_count <= 0:
            raise ValueError(f"device_count must be > 0, got {self.device_count}")
        if self.max_session_h <= 0:
            raise ValueError(f"max_session_h must be > 0, got {self.max_session_h}")
        if self.slots_per_device <= 0:
            raise ValueError(f"slots_per_device must be > 0, got {self.slots_per_device}")
        if self.debit_rate <= 0:
            raise ValueError(f"debit_rate must be > 0, got {self.debit_rate}")

    @property
    def max_concurrent_workers(self) -> int:
        return self.device_count * self.slots_per_device


@dataclass
class DeviceSlot:
    """An execution slot on an accelerator device, supporting multi-job concurrent streams."""
    instance_id: str
    device_index: int
    slot_index: int = 0
    load_h: float = 0.0
    jobs: List[Job] = field(default_factory=list)

    def can_fit_session(self, job: Job, max_session_h: float) -> bool:
        return (self.load_h + job.duration_h) <= max_session_h

    def assign(self, job: Job) -> None:
        self.jobs.append(job)
        self.load_h += job.duration_h

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
        return sum(s.load_h for s in self.slots)

    @property
    def device_loads_h(self) -> Dict[int, float]:
        """Calculates effective wall-clock load per physical device across its concurrent execution slots."""
        loads: Dict[int, float] = {d: 0.0 for d in range(self.instance.device_count)}
        for s in self.slots:
            loads[s.device_index] = max(loads.get(s.device_index, 0.0), s.load_h)
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
        # Configurable, calibrated quota accounting:
        # Defaults to 1.0 * makespan for wall-clock account session quotas,
        # or device_count * makespan if debit_rate is configured for device-hours.
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
        return sum(s.peak_cpu_cores for s in self.slots)


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
        return sum(ip.total_work_h for ip in self.instance_plans)

    @property
    def total_billed_resource_h(self) -> float:
        return sum(ip.billed_resource_h for ip in self.instance_plans)

    @property
    def global_makespan_h(self) -> float:
        if not self.active_instances:
            return 0.0
        return max(ip.makespan_h for ip in self.active_instances)

    @property
    def efficiency_ratio(self) -> float:
        if self.total_billed_resource_h <= 0:
            return 1.0
        return min(1.0, self.total_work_h / self.total_billed_resource_h)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "ACCELPACK WORKLOAD SCHEDULE SUMMARY",
            f"Total Jobs: {sum(len(s.jobs) for ip in self.instance_plans for s in ip.slots)}",
            f"Active Instances: {len(self.active_instances)} / {len(self.instance_plans)} "
            f"({len(self.untouched_instances)} preserved untouched)",
            f"Total Work: {self.total_work_h:.2f} worker-hours",
            f"Total Billed Resource Time: {self.total_billed_resource_h:.2f} quota-hours",
            f"Packing Efficiency: {self.efficiency_ratio * 100:.1f}%",
            f"Global Makespan: {self.global_makespan_h:.2f} hours",
            "-" * 70,
        ]

        for ip in self.active_instances:
            acc = ip.instance
            lines.append(
                f"[{acc.instance_id}] Makespan: {ip.makespan_h:.2f}h | "
                f"Billed: {ip.billed_resource_h:.2f}h | "
                f"Quota Remaining After: {ip.quota_remaining_after_h:.2f}h | "
                f"Workers: {ip.peak_concurrent_workers}/{acc.max_concurrent_workers}"
            )
            for d in range(acc.device_count):
                d_slots = [s for s in ip.slots if s.device_index == d]
                d_load = ip.device_loads_h.get(d, 0.0)
                d_vram = ip.peak_vram_per_device_gb.get(d, 0.0)
                lines.append(f"  ├── Device {d} (Wall-load: {d_load:.2f}h, Peak VRAM: {d_vram:.2f}/{acc.vram_per_device_gb} GB):")
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
