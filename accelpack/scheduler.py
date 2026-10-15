"""Algorithmic multi-accelerator bin-packing and quota-preserving scheduler."""

from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from accelpack.models import (
    DeviceSlot,
    ExecutionPlan,
    Instance,
    InstancePlan,
    Job,
    ObjectiveMode,
    PackingStrategy,
)


class QuotaExceededException(Exception):
    """Raised when workload cannot fit within available instance session limits or quotas."""
    pass


class AccelPackScheduler:
    """Multi-device workload packing and quota-preserving scheduler supporting concurrent device streams."""

    def __init__(
        self,
        strategy: PackingStrategy = PackingStrategy.BALANCE,
        objective: ObjectiveMode = ObjectiveMode.COMPLETE_ALL,
        setup_overhead_h: float = 0.10,
        quota_buffer_h: float = 2.0,
        max_session_h: Optional[float] = None,
        slots_per_device: Optional[int] = None,
        safety_vram_margin_gb: float = 0.5,
        safety_ram_margin_gb: float = 1.0,
        cpu_penalty_weight: float = 0.5,
        max_cpu_oversubscription_ratio: Optional[float] = None,
        enforce_reset_deadline: bool = False,
    ):
        """
        Args:
            strategy: PackingStrategy.BALANCE (water-filling across instances, default)
                      or PackingStrategy.CONCENTRATE (pack onto minimal instances).
            objective: ObjectiveMode.COMPLETE_ALL, MAXIMIZE_WORK, or MAXIMIZE_EFFICIENCY.
            setup_overhead_h: Fixed container cold-start overhead in hours (~6 min).
            quota_buffer_h: Minimum safety quota reserve that must remain untouched on each account.
            max_session_h: Optional global override for instance session timeout.
            slots_per_device: Optional override for concurrent execution streams per device (e.g. 2 workers per GPU).
            safety_vram_margin_gb: VRAM headroom buffer per physical device.
            safety_ram_margin_gb: Host RAM headroom buffer.
            cpu_penalty_weight: Weight applied to CPU oversubscription penalty in scheduling choices.
            max_cpu_oversubscription_ratio: Optional hard ceiling on CPU overcommit. None allows soft preference.
            enforce_reset_deadline: If True, reject placing jobs if makespan crosses instance reset_at timestamp.
        """
        self.strategy = strategy
        self.objective = objective
        self.setup_overhead_h = setup_overhead_h
        self.quota_buffer_h = quota_buffer_h
        self.max_session_h = max_session_h
        self.slots_per_device_override = slots_per_device
        self.safety_vram_margin_gb = safety_vram_margin_gb
        self.safety_ram_margin_gb = safety_ram_margin_gb
        self.cpu_penalty_weight = cpu_penalty_weight
        self.max_cpu_oversubscription_ratio = max_cpu_oversubscription_ratio
        self.enforce_reset_deadline = enforce_reset_deadline

    def schedule(
        self,
        jobs: List[Job],
        instances: List[Instance],
        allow_partial: bool = False,
    ) -> ExecutionPlan:
        """
        Schedules jobs across instances minimizing opened instances and load imbalance while respecting hardware limits.
        """
        if not jobs:
            return ExecutionPlan(
                instance_plans=[self._create_instance_plan(inst) for inst in instances],
                unassigned_jobs=[],
            )

        if not instances:
            if allow_partial:
                return ExecutionPlan(instance_plans=[], unassigned_jobs=list(jobs))
            raise QuotaExceededException("No compute instances provided to schedule workload.")

        # 1. Sort jobs based on objective with deterministic tie-breaking
        if self.objective == ObjectiveMode.MAXIMIZE_WORK:
            # Sort by priority-weighted duration descending, then priority descending, then duration ascending
            sorted_jobs = sorted(
                jobs,
                key=lambda j: (j.priority * j.duration_h, j.priority, -j.duration_h, j.job_id),
                reverse=True,
            )
        elif self.objective == ObjectiveMode.MAXIMIZE_EFFICIENCY:
            # Sort to pack densest workloads first to maximize useful work per billed hour
            sorted_jobs = sorted(
                jobs,
                key=lambda j: (j.duration_h, j.priority, j.job_id),
                reverse=True,
            )
        else:
            # COMPLETE_ALL: Longest Processing Time First (LPT)
            sorted_jobs = sorted(
                jobs,
                key=lambda j: (j.duration_h, j.priority, j.job_id),
                reverse=True,
            )

        total_work_h = sum(j.duration_h for j in sorted_jobs)

        # 2. Sort candidate instances by remaining quota descending, preserving stable order on ties
        candidate_instances = sorted(
            instances,
            key=lambda inst: inst.remaining_quota_h,
            reverse=True,
        )

        if self.strategy == PackingStrategy.BALANCE:
            plan = self._pack_balance_waterfill(sorted_jobs, candidate_instances, allow_partial=allow_partial)
            if plan is not None and (allow_partial or not plan.unassigned_jobs):
                return plan
            if not allow_partial:
                raise QuotaExceededException(
                    f"Workload of {len(jobs)} jobs ({total_work_h:.2f} worker-hours) cannot be balanced "
                    f"within session timeout and safety quota limits across the {len(instances)} available instances."
                )

        # PackingStrategy.CONCENTRATE:
        # Search minimal active instance count k
        for k in range(1, len(candidate_instances) + 1):
            subset = candidate_instances[:k]
            plan = self._pack_greedy(sorted_jobs, subset, candidate_instances, allow_partial=False)
            if plan is not None and not plan.unassigned_jobs:
                return plan

        if allow_partial:
            plan = self._pack_greedy(sorted_jobs, candidate_instances, candidate_instances, allow_partial=True)
            if plan is not None:
                return plan

        raise QuotaExceededException(
            f"Workload of {len(jobs)} jobs ({total_work_h:.2f} worker-hours) cannot fit within "
            f"available session limits and quota safety buffers."
        )

    def _create_instance_plan(self, inst: Instance) -> InstancePlan:
        num_slots = self.slots_per_device_override or inst.slots_per_device
        slots = [
            DeviceSlot(instance_id=inst.instance_id, device_index=d, slot_index=s)
            for d in range(inst.device_count)
            for s in range(num_slots)
        ]
        return InstancePlan(instance=inst, slots=slots, setup_overhead_h=self.setup_overhead_h)

    def _get_account_id(self, inst: Instance) -> str:
        return inst.account_id or inst.instance_id

    def _compute_hypothetical_state(
        self,
        ip: InstancePlan,
        candidate_slot: DeviceSlot,
        job: Job,
    ) -> Tuple[Dict[int, float], float, float, float, float]:
        """Calculates hypothetical device loads, makespan, billed quota, CPU ratio, and CPU penalty
        accounting for concurrency slowdown when adding job to candidate_slot.
        """
        inst = ip.instance
        dev_loads: Dict[int, float] = {}

        for d in range(inst.device_count):
            # Calculate active workers on device d after adding job
            workers = 0
            for s in ip.slots:
                if s.device_index == d:
                    has_jobs = len(s.jobs) > 0 or (s is candidate_slot)
                    if has_jobs:
                        workers += 1

            slowdown = inst.get_concurrency_slowdown(workers)

            d_load = 0.0
            for s in ip.slots:
                if s.device_index == d:
                    s_load = (s.load_h + job.duration_h) if s is candidate_slot else s.load_h
                    d_load = max(d_load, s_load * slowdown)
            dev_loads[d] = d_load

        makespan = max(dev_loads.values(), default=0.0) + self.setup_overhead_h
        billed = inst.debit_rate * makespan

        # CPU oversubscription estimate
        peak_cpu = 0.0
        for s in ip.slots:
            slot_cpu = max(s.peak_cpu_cores, job.cpu_cores) if s is candidate_slot else s.peak_cpu_cores
            if len(s.jobs) > 0 or (s is candidate_slot):
                peak_cpu += slot_cpu

        cpu_ratio = peak_cpu / inst.host_cpu_cores if inst.host_cpu_cores > 0 else 1.0
        cpu_penalty = max(0.0, cpu_ratio - 1.0) ** 2

        return dev_loads, makespan, billed, cpu_ratio, cpu_penalty

    def _check_reset_deadline(self, inst: Instance, hypothetical_makespan: float) -> bool:
        """Returns True if schedule respects quota reset deadline."""
        if not (self.enforce_reset_deadline or inst.must_finish_before_reset):
            return True
        if not inst.reset_at:
            return True
        try:
            reset_dt = datetime.fromisoformat(inst.reset_at.replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            hours_to_reset = (reset_dt - now_dt).total_seconds() / 3600.0
            if hours_to_reset > 0 and hypothetical_makespan > hours_to_reset:
                return False
        except Exception:
            pass
        return True

    def _pack_balance_waterfill(
        self,
        sorted_jobs: List[Job],
        instances: List[Instance],
        allow_partial: bool,
    ) -> Optional[ExecutionPlan]:
        instance_plans = [self._create_instance_plan(inst) for inst in instances]
        unassigned_jobs: List[Job] = []

        for job in sorted_jobs:
            candidate_choices: List[Tuple[int, float, float, InstancePlan, DeviceSlot]] = []

            for ip in instance_plans:
                inst = ip.instance
                session_limit = self.max_session_h if self.max_session_h is not None else inst.max_session_h

                if not self._check_hardware_fit(ip, None, job):
                    continue

                for slot in ip.slots:
                    if not self._check_hardware_fit(ip, slot, job):
                        continue

                    (
                        hyp_dev_loads,
                        hyp_makespan,
                        hyp_billed,
                        cpu_ratio,
                        cpu_penalty,
                    ) = self._compute_hypothetical_state(ip, slot, job)

                    # Check session cap
                    if max(hyp_dev_loads.values(), default=0.0) > session_limit:
                        continue

                    # Check reset deadline if applicable
                    if not self._check_reset_deadline(inst, hyp_makespan):
                        continue

                    # Check account-level shared quota buffer
                    acc_id = self._get_account_id(inst)
                    other_inst_billed = sum(
                        other_ip.billed_resource_h
                        for other_ip in instance_plans
                        if other_ip is not ip and self._get_account_id(other_ip.instance) == acc_id
                    )
                    total_acc_billed = other_inst_billed + hyp_billed
                    post_quota_rem = inst.remaining_quota_h - total_acc_billed

                    if post_quota_rem < self.quota_buffer_h:
                        continue

                    # In BALANCE mode:
                    # Priority 1: Fill an idle partner device or empty concurrent slot on an already-active instance
                    currently_active = ip.is_active
                    is_filling_idle = currently_active and slot.load_h == 0 and any(s.load_h > 0 for s in ip.slots)
                    partner_rank = 0 if is_filling_idle else 1

                    # Priority 2: Direct work to instance/account with highest remaining quota after billing
                    quota_bin = -round(post_quota_rem, 2)

                    # Priority 3: Minimize device load imbalance + CPU contention penalty
                    dev_loads = list(hyp_dev_loads.values())
                    hyp_imbalance = max(dev_loads) - min(dev_loads)
                    score = hyp_imbalance + self.cpu_penalty_weight * cpu_penalty

                    candidate_choices.append((partner_rank, quota_bin, score, ip, slot))

            if candidate_choices:
                candidate_choices.sort(key=lambda item: (item[0], item[1], item[2]))
                _, _, _, chosen_ip, chosen_slot = candidate_choices[0]
                chosen_slot.assign(job)
            else:
                if not allow_partial:
                    return None
                job.metadata["unassigned_reason"] = (
                    "Exceeds session max, reset window, or quota safety buffer across all candidate instances"
                )
                unassigned_jobs.append(job)

        return ExecutionPlan(instance_plans=instance_plans, unassigned_jobs=unassigned_jobs)

    def _pack_greedy(
        self,
        sorted_jobs: List[Job],
        active_instances: List[Instance],
        all_instances: List[Instance],
        allow_partial: bool,
    ) -> Optional[ExecutionPlan]:
        active_plans = [self._create_instance_plan(inst) for inst in active_instances]
        unassigned_jobs: List[Job] = []

        for job in sorted_jobs:
            best_slot: Optional[Tuple[float, InstancePlan, DeviceSlot]] = None

            for ip in active_plans:
                inst = ip.instance
                session_limit = self.max_session_h if self.max_session_h is not None else inst.max_session_h

                for slot in ip.slots:
                    if not self._check_hardware_fit(ip, slot, job):
                        continue

                    (
                        hyp_dev_loads,
                        hyp_makespan,
                        hyp_billed,
                        cpu_ratio,
                        cpu_penalty,
                    ) = self._compute_hypothetical_state(ip, slot, job)

                    if max(hyp_dev_loads.values(), default=0.0) > session_limit:
                        continue

                    if not self._check_reset_deadline(inst, hyp_makespan):
                        continue

                    acc_id = self._get_account_id(inst)
                    other_inst_billed = sum(
                        other_ip.billed_resource_h
                        for other_ip in active_plans
                        if other_ip is not ip and self._get_account_id(other_ip.instance) == acc_id
                    )
                    total_acc_billed = other_inst_billed + hyp_billed
                    post_quota_rem = inst.remaining_quota_h - total_acc_billed

                    if post_quota_rem < self.quota_buffer_h:
                        continue

                    slot_score = slot.load_h + self.cpu_penalty_weight * cpu_penalty
                    if best_slot is None or slot_score < best_slot[0]:
                        best_slot = (slot_score, ip, slot)

            if best_slot is not None:
                _, chosen_ip, chosen_slot = best_slot
                chosen_slot.assign(job)
            else:
                if not allow_partial:
                    return None
                job.metadata["unassigned_reason"] = "Cannot fit into active instance budget"
                unassigned_jobs.append(job)

        all_plans = list(active_plans)
        active_ids = {inst.instance_id for inst in active_instances}
        for inst in all_instances:
            if inst.instance_id not in active_ids:
                all_plans.append(self._create_instance_plan(inst))

        return ExecutionPlan(instance_plans=all_plans, unassigned_jobs=unassigned_jobs)

    def _check_hardware_fit(self, ip: InstancePlan, candidate_slot: Optional[DeviceSlot], job: Job) -> bool:
        """
        Verifies hardware feasibility:
        1. VRAM: Per-device peak VRAM + safety margin <= physical accelerator capacity (HARD constraint).
        2. Host RAM: Sum of peak RAM across all active slots + safety margin <= host RAM (HARD constraint).
        3. Host CPU: Sum of peak CPU demand across active slots. CPU oversubscription is allowed as a soft
           preference, but blocked if an explicit emergency cap (max_cpu_oversubscription_ratio) is configured.
        """
        inst = ip.instance

        # 1. Device-specific VRAM constraint
        if candidate_slot is not None:
            current_dev_vram = 0.0
            for s in ip.slots:
                if s.device_index == candidate_slot.device_index:
                    slot_vram = s.peak_vram_gb if s is not candidate_slot else max(s.peak_vram_gb, job.vram_gb)
                    current_dev_vram += slot_vram

            if (current_dev_vram + self.safety_vram_margin_gb) > inst.vram_per_device_gb:
                return False
        else:
            if (job.vram_gb + self.safety_vram_margin_gb) > inst.vram_per_device_gb:
                return False

        # 2. Host-level RAM constraint
        current_host_ram = 0.0
        for s in ip.slots:
            slot_ram = s.peak_ram_gb if s is not candidate_slot else max(s.peak_ram_gb, job.ram_gb)
            current_host_ram += slot_ram
        if candidate_slot is None:
            current_host_ram += job.ram_gb

        if (current_host_ram + self.safety_ram_margin_gb) > inst.host_ram_gb:
            return False

        # 3. Host-level CPU constraint (Soft preference with optional hard ceiling)
        current_host_cpu = 0.0
        for s in ip.slots:
            slot_cpu = s.peak_cpu_cores if s is not candidate_slot else max(s.peak_cpu_cores, job.cpu_cores)
            if len(s.jobs) > 0 or (s is candidate_slot):
                current_host_cpu += slot_cpu
        if candidate_slot is None:
            current_host_cpu += job.cpu_cores

        emergency_ratio = inst.max_cpu_oversubscription_ratio or self.max_cpu_oversubscription_ratio
        if emergency_ratio is not None and emergency_ratio > 0:
            if current_host_cpu > (inst.host_cpu_cores * emergency_ratio):
                return False

        return True
