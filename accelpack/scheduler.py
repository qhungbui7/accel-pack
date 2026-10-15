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
        reject_conflicting_account_quotas: bool = False,
        expand_segments: bool = True,
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
            reject_conflicting_account_quotas: If True, raise exception when instances sharing an account_id
                                              declare conflicting remaining quotas. If False, normalize to min().
            expand_segments: If True, split jobs declaring valid checkpoint segments into schedulable sub-jobs.
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
        self.reject_conflicting_account_quotas = reject_conflicting_account_quotas
        self.expand_segments = expand_segments

    def _normalize_account_quotas(self, instances: List[Instance]) -> List[Instance]:
        """Ensures every account pool has exactly one authoritative quota balance."""
        account_groups: Dict[str, List[Instance]] = {}
        for inst in instances:
            acc_id = inst.account_id or inst.instance_id
            account_groups.setdefault(acc_id, []).append(inst)

        normalized: List[Instance] = []
        for acc_id, inst_list in account_groups.items():
            quotas = [i.remaining_quota_h for i in inst_list]
            if max(quotas) - min(quotas) > 1e-4:
                if self.reject_conflicting_account_quotas:
                    raise QuotaExceededException(
                        f"Conflicting quota declarations for account '{acc_id}': {quotas}. "
                        f"Cannot safely schedule without authoritative quota balance."
                    )
                auth_quota = min(quotas)
            else:
                auth_quota = quotas[0]

            for inst in inst_list:
                inst_copy = copy.copy(inst)
                inst_copy.remaining_quota_h = auth_quota
                normalized.append(inst_copy)

        return normalized

    def _expand_segmented_jobs(self, jobs: List[Job]) -> List[Job]:
        """Expands jobs with checkpoint segments into schedulable discrete segment units with explicit dependencies."""
        if not self.expand_segments:
            return jobs

        expanded: List[Job] = []
        for j in jobs:
            if j.segments and len(j.segments) > 0:
                for idx, seg_dur in enumerate(j.segments):
                    prev_id = f"{j.job_id}_seg{idx - 1}" if idx > 0 else None
                    seg_job = Job(
                        job_id=f"{j.job_id}_seg{idx}",
                        duration_h=seg_dur,
                        vram_gb=j.vram_gb,
                        ram_gb=j.ram_gb,
                        cpu_cores=j.cpu_cores,
                        priority=j.priority,
                        hardware_class=j.hardware_class,
                        resumable=j.resumable,
                        depends_on=prev_id,
                        parent_job_id=j.job_id,
                        segment_index=idx,
                        metadata={
                            **j.metadata,
                            "parent_job_id": j.job_id,
                            "segment_index": idx,
                            "total_segments": len(j.segments),
                            "depends_on": prev_id,
                        },
                    )
                    expanded.append(seg_job)
            else:
                expanded.append(j)
        return expanded

    def _order_jobs(self, jobs: List[Job]) -> List[Job]:
        """Orders jobs preserving prerequisite dependencies (depends_on) while applying
        objective-specific sorting among ready jobs."""
        if self.objective == ObjectiveMode.MAXIMIZE_WORK:
            key_fn = lambda j: (j.priority * j.duration_h, j.priority, -j.duration_h, j.job_id)
        else:
            key_fn = lambda j: (j.duration_h, j.priority, j.job_id)

        job_map = {j.job_id: j for j in jobs}
        in_degree: Dict[str, int] = {j.job_id: 0 for j in jobs}
        dependents: Dict[str, List[Job]] = {j.job_id: [] for j in jobs}

        for j in jobs:
            if j.depends_on and j.depends_on in job_map:
                in_degree[j.job_id] += 1
                dependents[j.depends_on].append(j)

        ready = [j for j in jobs if in_degree[j.job_id] == 0]
        ordered: List[Job] = []

        while ready:
            ready.sort(key=key_fn, reverse=True)
            chosen = ready.pop(0)
            ordered.append(chosen)

            for dep in dependents[chosen.job_id]:
                in_degree[dep.job_id] -= 1
                if in_degree[dep.job_id] == 0:
                    ready.append(dep)

        if len(ordered) < len(jobs):
            for j in jobs:
                if j not in ordered:
                    ordered.append(j)

        return ordered

    def _check_dependency_satisfaction(
        self,
        job: Job,
        candidate_ip: InstancePlan,
        candidate_slot: DeviceSlot,
        all_plans: List[InstancePlan],
    ) -> bool:
        """Ensures checkpoint dependencies are satisfied:
        1. If job has depends_on, that prerequisite job must already be assigned in the plan.
        2. If prerequisite is on the same instance, it MUST be in the same slot (running sequentially before this job).
           (Different slots on the same node execute concurrently from container boot, violating checkpoint causality).
        3. If prerequisite is on a different instance, the cross-instance/session dependency is preserved.
        """
        if not job.depends_on:
            return True

        prereq_ip: Optional[InstancePlan] = None
        prereq_slot: Optional[DeviceSlot] = None
        for ip in all_plans:
            for s in ip.slots:
                if any(j.job_id == job.depends_on for j in s.jobs):
                    prereq_ip = ip
                    prereq_slot = s
                    break
            if prereq_ip is not None:
                break

        if prereq_ip is None or prereq_slot is None:
            return False

        if prereq_ip.instance.instance_id == candidate_ip.instance.instance_id:
            return prereq_slot is candidate_slot

        return True

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

        # 1. Authoritative account quota normalization
        normalized_instances = self._normalize_account_quotas(instances)

        # 2. Expand segmented jobs if enabled
        schedulable_jobs = self._expand_segmented_jobs(jobs)

        # 3. Sort jobs topologically respecting dependencies and objective mode
        sorted_jobs = self._order_jobs(schedulable_jobs)
        total_work_h = sum(j.duration_h for j in sorted_jobs)

        # 4. Sort candidate instances by remaining quota descending, preserving stable order on ties
        candidate_instances = sorted(
            normalized_instances,
            key=lambda inst: inst.remaining_quota_h,
            reverse=True,
        )

        if self.strategy == PackingStrategy.BALANCE:
            plan = self._pack_balance_waterfill(sorted_jobs, candidate_instances, allow_partial=allow_partial)
            if plan is not None and (allow_partial or not plan.unassigned_jobs):
                return plan
            if not allow_partial:
                raise QuotaExceededException(
                    f"Heuristic scheduler could not find a placement balancing {len(jobs)} jobs "
                    f"({total_work_h:.2f} worker-hours) within session timeouts and quota safety buffers "
                    f"across {len(instances)} available instances."
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
            f"Heuristic scheduler could not find a placement packing {len(jobs)} jobs "
            f"({total_work_h:.2f} worker-hours) within session timeouts and quota safety buffers."
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

    def _check_standalone_fit(self, inst: Instance, job: Job) -> bool:
        """Verifies if job can fit on this instance standalone (hardware class, standalone VRAM/RAM)."""
        # Hardware class compatibility
        if job.hardware_class is not None:
            if job.hardware_class.strip().lower() != inst.hardware_class.strip().lower():
                return False

        # Standalone VRAM ceiling
        if (job.vram_gb + self.safety_vram_margin_gb) > inst.vram_per_device_gb:
            return False

        # Standalone RAM ceiling
        if (job.ram_gb + self.safety_ram_margin_gb) > inst.host_ram_gb:
            return False

        return True

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

        max_load = max(dev_loads.values(), default=0.0)
        makespan = max_load + (self.setup_overhead_h if max_load > 0 else 0.0)
        billed = inst.debit_rate * makespan

        # Aggregate CPU oversubscription estimate
        peak_cpu = 0.0
        for s in ip.slots:
            slot_cpu = max(s.peak_cpu_cores, job.cpu_cores) if s is candidate_slot else s.peak_cpu_cores
            if len(s.jobs) > 0 or (s is candidate_slot):
                peak_cpu += slot_cpu

        cpu_ratio = peak_cpu / inst.host_cpu_cores if inst.host_cpu_cores > 0 else 1.0
        cpu_penalty = max(0.0, cpu_ratio - 1.0) ** 2

        return dev_loads, makespan, billed, cpu_ratio, cpu_penalty

    def _check_reset_deadline(self, inst: Instance, hypothetical_makespan: float) -> bool:
        """Returns True if schedule respects quota reset deadline. Fails safe (False) on invalid/stale timestamps."""
        if not (self.enforce_reset_deadline or inst.must_finish_before_reset):
            return True

        if not inst.reset_at:
            # Enforcement requested but reset_at is missing -> Fail safe
            return False

        try:
            reset_dt = datetime.fromisoformat(inst.reset_at.replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            hours_to_reset = (reset_dt - now_dt).total_seconds() / 3600.0

            # If reset timestamp has already passed, quota is stale -> Fail safe
            if hours_to_reset <= 0:
                return False

            # Planned makespan must not cross the deadline
            if hypothetical_makespan > hours_to_reset:
                return False

            return True
        except Exception:
            # Malformed timestamp with deadline enforcement requested -> Fail safe
            return False

    def _pack_balance_waterfill(
        self,
        sorted_jobs: List[Job],
        instances: List[Instance],
        allow_partial: bool,
    ) -> Optional[ExecutionPlan]:
        instance_plans = [self._create_instance_plan(inst) for inst in instances]
        unassigned_jobs: List[Job] = []

        for job in sorted_jobs:
            candidate_choices: List[Tuple[Any, ...]] = []

            for ip in instance_plans:
                inst = ip.instance
                session_limit = self.max_session_h if self.max_session_h is not None else inst.max_session_h

                # 1. Standalone feasibility check (Hardware class, standalone VRAM/RAM)
                # Does NOT reject jobs that fit sequentially in existing slots!
                if not self._check_standalone_fit(inst, job):
                    continue

                for slot in ip.slots:
                    # 2. Per-slot actual aggregate placement feasibility check
                    if not self._check_hardware_fit(ip, slot, job):
                        continue

                    # Check checkpoint dependency satisfaction
                    if not self._check_dependency_satisfaction(job, ip, slot, instance_plans):
                        continue

                    (
                        hyp_dev_loads,
                        hyp_makespan,
                        hyp_billed,
                        cpu_ratio,
                        cpu_penalty,
                    ) = self._compute_hypothetical_state(ip, slot, job)

                    # Check session cap including setup overhead
                    if hyp_makespan > session_limit:
                        continue

                    # Check reset deadline safely
                    if not self._check_reset_deadline(inst, hyp_makespan):
                        continue

                    # Check account-level shared quota buffer against authoritative pool
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

                    # Scoring based on ObjectiveMode
                    if self.objective == ObjectiveMode.MAXIMIZE_EFFICIENCY:
                        # Prioritize packing into already-active instances & devices to maximize useful work per billed hour
                        current_work = ip.total_work_h
                        new_work = current_work + job.duration_h
                        new_efficiency = (new_work / hyp_billed) if hyp_billed > 0 else 0.0

                        partner_rank = 0 if ip.is_active else 1
                        is_active_device = any(s.load_h > 0 for s in ip.slots if s.device_index == slot.device_index)
                        device_pref = 0 if is_active_device else 1

                        # Sort by: partner_rank (prefer active nodes), -new_efficiency, device_pref, weighted cpu_penalty
                        candidate_choices.append((
                            partner_rank,
                            -round(new_efficiency, 4),
                            device_pref,
                            round(self.cpu_penalty_weight * cpu_penalty, 3),
                            ip,
                            slot,
                        ))
                    else:
                        # COMPLETE_ALL & MAXIMIZE_WORK:
                        # Priority 1: Fill an idle partner device or empty concurrent slot on an already-active instance
                        currently_active = ip.is_active
                        is_filling_idle = currently_active and slot.load_h == 0 and any(s.load_h > 0 for s in ip.slots)
                        partner_rank = 0 if is_filling_idle else 1

                        # Priority 2: Direct work to instance/account with highest remaining quota after billing
                        quota_bin = -round(post_quota_rem, 2)

                        # Priority 3: Normalized device imbalance + dimensionless CPU contention penalty
                        dev_loads = list(hyp_dev_loads.values())
                        hyp_imbalance = max(dev_loads) - min(dev_loads)
                        max_l = max(dev_loads) if dev_loads else 1.0
                        norm_imbalance = hyp_imbalance / max(1.0, max_l)
                        score = norm_imbalance + self.cpu_penalty_weight * cpu_penalty

                        candidate_choices.append((partner_rank, quota_bin, round(score, 4), ip, slot))

            if candidate_choices:
                candidate_choices.sort(key=lambda item: item[:-2])
                *_, chosen_ip, chosen_slot = candidate_choices[0]
                chosen_slot.assign(job)
            else:
                if not allow_partial:
                    return None
                job.metadata["unassigned_reason"] = (
                    "Exceeds session max, reset window, hardware compatibility, or quota safety buffer"
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

                if not self._check_standalone_fit(inst, job):
                    continue

                for slot in ip.slots:
                    if not self._check_hardware_fit(ip, slot, job):
                        continue

                    # Check checkpoint dependency satisfaction
                    if not self._check_dependency_satisfaction(job, ip, slot, active_plans):
                        continue

                    (
                        hyp_dev_loads,
                        hyp_makespan,
                        hyp_billed,
                        cpu_ratio,
                        cpu_penalty,
                    ) = self._compute_hypothetical_state(ip, slot, job)

                    # Check session cap including setup overhead
                    if hyp_makespan > session_limit:
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

                    norm_slot_load = slot.load_h / max(1.0, session_limit)
                    slot_score = norm_slot_load + self.cpu_penalty_weight * cpu_penalty
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

    def _check_hardware_fit(self, ip: InstancePlan, candidate_slot: DeviceSlot, job: Job) -> bool:
        """
        Verifies actual hardware feasibility for placing job into candidate_slot:
        1. Hardware class: Must match instance hardware_class if job specifies one.
        2. VRAM: Per-device peak VRAM across concurrent slots on candidate device + safety margin <= capacity.
        3. Host RAM: Sum of peak RAM across slots (accounting for sequential execution within candidate slot).
        4. Host CPU: Sum of peak CPU demand across active slots, with optional emergency ratio check.
        """
        inst = ip.instance

        # 1. Hardware class compatibility
        if job.hardware_class is not None:
            if job.hardware_class.strip().lower() != inst.hardware_class.strip().lower():
                return False

        # 2. Device-specific VRAM constraint
        current_dev_vram = 0.0
        for s in ip.slots:
            if s.device_index == candidate_slot.device_index:
                slot_vram = max(s.peak_vram_gb, job.vram_gb) if s is candidate_slot else s.peak_vram_gb
                current_dev_vram += slot_vram

        if (current_dev_vram + self.safety_vram_margin_gb) > inst.vram_per_device_gb:
            return False

        # 3. Host-level RAM constraint (Within candidate_slot, jobs run sequentially: max(peak, job))
        current_host_ram = 0.0
        for s in ip.slots:
            slot_ram = max(s.peak_ram_gb, job.ram_gb) if s is candidate_slot else s.peak_ram_gb
            current_host_ram += slot_ram

        if (current_host_ram + self.safety_ram_margin_gb) > inst.host_ram_gb:
            return False

        # 4. Host-level CPU constraint
        current_host_cpu = 0.0
        for s in ip.slots:
            slot_cpu = max(s.peak_cpu_cores, job.cpu_cores) if s is candidate_slot else s.peak_cpu_cores
            if len(s.jobs) > 0 or (s is candidate_slot):
                current_host_cpu += slot_cpu

        emergency_ratio = inst.max_cpu_oversubscription_ratio or self.max_cpu_oversubscription_ratio
        if emergency_ratio is not None and emergency_ratio > 0:
            if current_host_cpu > (inst.host_cpu_cores * emergency_ratio):
                return False

        return True
