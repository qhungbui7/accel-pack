"""Algorithmic multi-accelerator bin-packing and quota-preserving scheduler."""

from __future__ import annotations

import copy
import math
from typing import Callable, Dict, List, Optional, Tuple

from accelpack.models import DeviceSlot, ExecutionPlan, Instance, InstancePlan, Job


class QuotaExceededException(Exception):
    """Raised when workload cannot fit within available instance session limits or quotas."""
    pass


class AccelPackScheduler:
    """Multi-device workload packing and quota-preserving scheduler."""

    def __init__(
        self,
        setup_overhead_h: float = 0.10,
        quota_buffer_h: float = 2.0,
        max_session_h: Optional[float] = None,
        max_concurrent_workers_per_instance: Optional[int] = None,
    ):
        """
        Args:
            setup_overhead_h: Fixed container cold-start overhead in hours (~6 min).
            quota_buffer_h: Minimum safety quota reserve that must remain untouched on each account.
            max_session_h: Optional global override for instance session timeout.
            max_concurrent_workers_per_instance: Max concurrent worker processes (defaults to device_count).
        """
        self.setup_overhead_h = setup_overhead_h
        self.quota_buffer_h = quota_buffer_h
        self.max_session_h = max_session_h
        self.max_concurrent_workers = max_concurrent_workers_per_instance

    def schedule(
        self,
        jobs: List[Job],
        instances: List[Instance],
        allow_partial: bool = False,
    ) -> ExecutionPlan:
        """
        Schedules jobs across instances minimizing opened instances and load imbalance.

        Args:
            jobs: List of jobs to schedule.
            instances: Available compute instances (sorted by preference or remaining quota).
            allow_partial: If True, schedules as many jobs as possible without raising exception.

        Returns:
            ExecutionPlan detailing per-instance slot assignments and efficiency metrics.
        """
        if not jobs:
            return ExecutionPlan(
                instance_plans=[
                    InstancePlan(
                        instance=inst,
                        slots=[
                            DeviceSlot(instance_id=inst.instance_id, device_index=d)
                            for d in range(inst.device_count)
                        ],
                        setup_overhead_h=self.setup_overhead_h,
                    )
                    for inst in instances
                ],
                unassigned_jobs=[],
            )

        if not instances:
            if allow_partial:
                return ExecutionPlan(instance_plans=[], unassigned_jobs=list(jobs))
            raise QuotaExceededException("No compute instances provided to schedule workload.")

        # 1. Sort jobs descending by duration (Longest Processing Time First)
        sorted_jobs = sorted(jobs, key=lambda j: j.duration_h, reverse=True)
        total_work_h = sum(j.duration_h for j in sorted_jobs)

        # 2. Prioritize instances with highest remaining quota
        candidate_instances = sorted(instances, key=lambda inst: inst.remaining_quota_h, reverse=True)

        # Compute theoretical minimum instances needed
        # Each instance can handle at most (device_count * max_session_h) of sequential work
        effective_max_session = min(
            inst.max_session_h if self.max_session_h is None else self.max_session_h
            for inst in candidate_instances
        )
        max_work_per_inst = candidate_instances[0].device_count * effective_max_session
        k_min = max(1, min(len(candidate_instances), int(math.ceil(total_work_h / max(max_work_per_inst, 1e-6)))))

        # 3. Iteratively search for minimal instance count k that satisfies all constraints
        for k in range(k_min, len(candidate_instances) + 1):
            sub_instances = candidate_instances[:k]
            plan = self._try_pack_workload(sorted_jobs, sub_instances)
            if plan is not None and not plan.unassigned_jobs:
                # Add back untouched instances into the final plan
                untouched = candidate_instances[k:]
                for inst in untouched:
                    slots = [
                        DeviceSlot(instance_id=inst.instance_id, device_index=d)
                        for d in range(inst.device_count)
                    ]
                    plan.instance_plans.append(
                        InstancePlan(
                            instance=inst,
                            slots=slots,
                            setup_overhead_h=self.setup_overhead_h,
                        )
                    )
                return plan

        # If full packing failed across all instances
        if allow_partial:
            # Greedily pack as many as possible into all instances
            partial_plan = self._greedy_partial_pack(sorted_jobs, candidate_instances)
            return partial_plan

        raise QuotaExceededException(
            f"Workload of {len(jobs)} jobs ({total_work_h:.2f} device-hours) cannot be scheduled "
            f"within session timeout and safety quota limits across the {len(instances)} available instances."
        )

    def _try_pack_workload(
        self,
        sorted_jobs: List[Job],
        instances: List[Instance],
    ) -> Optional[ExecutionPlan]:
        """Attempts to pack all jobs into the selected subset of instances."""
        # Initialize slots across all candidate instances
        instance_plans: List[InstancePlan] = []
        all_slots: List[Tuple[Instance, DeviceSlot]] = []

        for inst in instances:
            session_limit = self.max_session_h if self.max_session_h is not None else inst.max_session_h
            slots = [
                DeviceSlot(instance_id=inst.instance_id, device_index=d)
                for d in range(inst.device_count)
            ]
            ip = InstancePlan(instance=inst, slots=slots, setup_overhead_h=self.setup_overhead_h)
            instance_plans.append(ip)
            for s in slots:
                all_slots.append((inst, s))

        # Greedy LPT placement with load leveling
        for job in sorted_jobs:
            # Sort slots by current load (least loaded first)
            # Secondary key: balance across devices of the same instance
            all_slots.sort(key=lambda item: (item[1].load_h, len(item[1].jobs)))

            placed = False
            for inst, slot in all_slots:
                session_limit = self.max_session_h if self.max_session_h is not None else inst.max_session_h
                # Check session limit
                if (slot.load_h + job.duration_h) > session_limit:
                    continue

                # Check hardware limits
                if not self._check_hardware_fit(inst, slot, job):
                    continue

                # Check quota limit for this instance
                ip = next(p for p in instance_plans if p.instance.instance_id == inst.instance_id)
                # Tentatively calculate makespan if added to this slot
                hypothetical_loads = [
                    (s.load_h + job.duration_h if s is slot else s.load_h)
                    for s in ip.slots
                ]
                hypothetical_makespan = max(hypothetical_loads) + self.setup_overhead_h
                hypothetical_billed = inst.device_count * hypothetical_makespan

                if hypothetical_billed > (inst.remaining_quota_h - self.quota_buffer_h):
                    continue

                # Valid slot found
                slot.assign(job)
                placed = True
                break

            if not placed:
                return None  # Cannot pack this job in the current instance subset

        return ExecutionPlan(instance_plans=instance_plans, unassigned_jobs=[])

    def _greedy_partial_pack(
        self,
        sorted_jobs: List[Job],
        instances: List[Instance],
    ) -> ExecutionPlan:
        """Packs as many jobs as possible into all available instances."""
        instance_plans: List[InstancePlan] = []
        all_slots: List[Tuple[Instance, DeviceSlot]] = []

        for inst in instances:
            slots = [
                DeviceSlot(instance_id=inst.instance_id, device_index=d)
                for d in range(inst.device_count)
            ]
            ip = InstancePlan(instance=inst, slots=slots, setup_overhead_h=self.setup_overhead_h)
            instance_plans.append(ip)
            for s in slots:
                all_slots.append((inst, s))

        unassigned: List[Job] = []

        for job in sorted_jobs:
            all_slots.sort(key=lambda item: (item[1].load_h, len(item[1].jobs)))
            placed = False
            for inst, slot in all_slots:
                session_limit = self.max_session_h if self.max_session_h is not None else inst.max_session_h
                if (slot.load_h + job.duration_h) > session_limit:
                    continue
                if not self._check_hardware_fit(inst, slot, job):
                    continue

                ip = next(p for p in instance_plans if p.instance.instance_id == inst.instance_id)
                hypothetical_loads = [
                    (s.load_h + job.duration_h if s is slot else s.load_h)
                    for s in ip.slots
                ]
                hypothetical_makespan = max(hypothetical_loads) + self.setup_overhead_h
                hypothetical_billed = inst.device_count * hypothetical_makespan

                if hypothetical_billed > (inst.remaining_quota_h - self.quota_buffer_h):
                    continue

                slot.assign(job)
                placed = True
                break

            if not placed:
                unassigned.append(job)

        return ExecutionPlan(instance_plans=instance_plans, unassigned_jobs=unassigned)

    def _check_hardware_fit(self, inst: Instance, slot: DeviceSlot, job: Job) -> bool:
        """Verifies VRAM, host RAM, and CPU constraints."""
        # 1. Device VRAM
        if job.vram_gb > inst.vram_per_device_gb:
            return False

        # 2. Host RAM and CPU per instance during execution
        # Each device executes at most 1 job concurrently.
        # Max simultaneous concurrent jobs on an instance is inst.device_count.
        # Verify single-worker limits do not exceed instance capacity.
        if job.ram_gb * inst.device_count > inst.host_ram_gb:
            return False
        if job.cpu_cores * inst.device_count > inst.host_cpu_cores:
            return False

        return True
