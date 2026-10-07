"""Core data structures and resource models for multi-accelerator scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PackingStrategy(str, Enum):
    """Strategy for distributing workloads across compute instances."""
    BALANCE = "balance"          # Water-filling: equalize remaining quota / usage across instances (DEFAULT)
    CONCENTRATE = "concentrate"  # Minimal instances: pack onto fewest nodes to keep instances untouched
@dataclass
class Job:
    """A computational unit of work to be scheduled on an accelerator."""
    job_id: str
    duration_h: float
    vram_gb: float = 0.5
    ram_gb: float = 1.5
    cpu_cores: float = 1.0
    device_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.duration_h <= 0:
            raise ValueError(f"Job duration must be > 0, got {self.duration_h}")
        if self.device_count != 1:
            raise NotImplementedError("Multi-accelerator distributed single-job gang scheduling is not yet supported.")


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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.device_count <= 0:
            raise ValueError(f"device_count must be > 0, got {self.device_count}")
        if self.max_session_h <= 0:
            raise ValueError(f"max_session_h must be > 0, got {self.max_session_h}")


@dataclass
class DeviceSlot:
    """An individual accelerator device slot inside a compute instance."""
    instance_id: str
    device_index: int
    load_h: float = 0.0
    jobs: List[Job] = field(default_factory=list)

    def can_fit_session(self, job: Job, max_session_h: float) -> bool:
        return (self.load_h + job.duration_h) <= max_session_h

    def assign(self, job: Job) -> None:
        self.jobs.append(job)
        self.load_h += job.duration_h


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
    def max_device_load_h(self) -> float:
        if not self.slots:
            return 0.0
        return max(s.load_h for s in self.slots)

    @property
    def makespan_h(self) -> float:
        if not self.is_active:
            return 0.0
        return self.max_device_load_h + self.setup_overhead_h

    @property
    def billed_resource_h(self) -> float:
        if not self.is_active:
            return 0.0
        return self.instance.device_count * self.makespan_h

    @property
    def load_imbalance_h(self) -> float:
        if len(self.slots) <= 1:
            return 0.0
        loads = [s.load_h for s in self.slots]
        return max(loads) - min(loads)

    @property
    def quota_remaining_after_h(self) -> float:
        return max(0.0, self.instance.remaining_quota_h - self.billed_resource_h)


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
            f"Total Work: {self.total_work_h:.2f} device-hours",
            f"Total Billed Resource Time: {self.total_billed_resource_h:.2f} device-hours",
            f"Packing Efficiency: {self.efficiency_ratio * 100:.1f}%",
            f"Global Makespan: {self.global_makespan_h:.2f} hours",
            "-" * 70,
        ]

        for ip in self.active_instances:
            acc = ip.instance
            lines.append(
                f"[{acc.instance_id}] Makespan: {ip.makespan_h:.2f}h | "
                f"Billed: {ip.billed_resource_h:.2f}h | "
                f"Quota Remaining After: {ip.quota_remaining_after_h:.2f}h"
            )
            for s in ip.slots:
                lines.append(f"  ├── Device {s.device_index} (Load: {s.load_h:.2f}h):")
                if not s.jobs:
                    lines.append("  │     (idle)")
                for idx, j in enumerate(s.jobs, start=1):
                    lines.append(f"  │     {idx}. {j.job_id} ({j.duration_h:.2f}h)")

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
                lines.append(f"  * {j.job_id} ({j.duration_h:.2f}h)")

        lines.append("=" * 70)
        return "\n".join(lines)
