"""Preflight validation pipeline for multi-accelerator fleet scheduling.

Validates compute nodes, account quota pools, reset boundaries, and job feasibility
prior to scheduling and launch without running expensive on-device diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from accelpack.models import Instance, Job


@dataclass
class PreflightIssue:
    severity: str  # "BLOCKER", "WARNING", "INFO"
    code: str
    message: str
    target_id: Optional[str] = None
    remediation: str = ""


@dataclass
class PreflightReport:
    """Outcome of preflight validation."""
    passed: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    issues: List[PreflightIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def blockers(self) -> List[PreflightIssue]:
        return [i for i in self.issues if i.severity == "BLOCKER"]

    @property
    def warnings(self) -> List[PreflightIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    def format_text(self) -> str:
        status_str = "PASSED" if self.passed else "FAILED"
        lines = [
            "=" * 70,
            f"ACCELPACK PREFLIGHT VALIDATION: {status_str}",
            f"Timestamp: {self.timestamp}",
            f"Blockers: {len(self.blockers)} | Warnings: {len(self.warnings)}",
            "-" * 70,
        ]
        for issue in self.issues:
            prefix = f"[{issue.severity}]"
            target = f" ({issue.target_id})" if issue.target_id else ""
            lines.append(f"{prefix:<10} {issue.code}{target}: {issue.message}")
            if issue.remediation:
                lines.append(f"           Fix: {issue.remediation}")
        lines.append("=" * 70)
        return "\n".join(lines)


class PreflightValidator:
    """Runs staged zero-cost validation checks before invoking the scheduler or launching sessions."""

    def __init__(
        self,
        min_quota_buffer_h: float = 2.0,
        max_session_overhead_ratio: float = 0.95,
        stale_quota_threshold_hours: float = 24.0,
    ):
        self.min_quota_buffer_h = min_quota_buffer_h
        self.max_session_overhead_ratio = max_session_overhead_ratio
        self.stale_quota_threshold_hours = stale_quota_threshold_hours

    def validate(
        self,
        jobs: List[Job],
        instances: List[Instance],
    ) -> PreflightReport:
        issues: List[PreflightIssue] = []

        # 1. Instance and Account-Level Quota Checks
        if not instances:
            issues.append(
                PreflightIssue(
                    severity="BLOCKER",
                    code="NO_INSTANCES_AVAILABLE",
                    message="No compute instances or accounts provided in fleet specification.",
                    remediation="Add at least one active compute instance with available accelerator quota.",
                )
            )

        # Group instances by account_id to prevent double-counting shared account quotas
        accounts: Dict[str, List[Instance]] = {}
        for inst in instances:
            acc_id = inst.account_id or inst.instance_id
            accounts.setdefault(acc_id, []).append(inst)

        total_usable_quota = 0.0
        for acc_id, inst_list in accounts.items():
            # Check for contradictory quota declarations across instances for the same account
            quotas = [i.remaining_quota_h for i in inst_list]
            if max(quotas) - min(quotas) > 1e-3:
                issues.append(
                    PreflightIssue(
                        severity="WARNING",
                        code="INCONSISTENT_ACCOUNT_QUOTA",
                        message=f"Instances sharing account '{acc_id}' declare differing remaining quotas: {quotas}.",
                        target_id=acc_id,
                        remediation="Synchronize quota balance across all node configurations belonging to this account.",
                    )
                )

            account_quota = min(quotas)
            usable_q = max(0.0, account_quota - self.min_quota_buffer_h)
            total_usable_quota += usable_q

            if account_quota <= self.min_quota_buffer_h:
                issues.append(
                    PreflightIssue(
                        severity="WARNING",
                        code="QUOTA_BUFFER_EXHAUSTED",
                        message=f"Account '{acc_id}' remaining quota ({account_quota:.2f}h) is below safety buffer ({self.min_quota_buffer_h:.2f}h).",
                        target_id=acc_id,
                        remediation="Reserve this account for emergency diagnostics or await quota reset window.",
                    )
                )

        # Reset timestamp checks
        for inst in instances:
            if inst.reset_at:
                try:
                    reset_dt = datetime.fromisoformat(inst.reset_at.replace("Z", "+00:00"))
                    now_dt = datetime.now(timezone.utc)
                    hours_to_reset = (reset_dt - now_dt).total_seconds() / 3600.0
                    if hours_to_reset < 0:
                        issues.append(
                            PreflightIssue(
                                severity="INFO",
                                code="RESET_TIME_PASSED",
                                message="Reset timestamp has passed; quota balance may have refreshed.",
                                target_id=inst.instance_id,
                                remediation="Refresh instance quota observation via API.",
                            )
                        )
                    elif hours_to_reset < inst.max_session_h:
                        issues.append(
                            PreflightIssue(
                                severity="INFO",
                                code="SESSION_BOUNDED_BY_RESET",
                                message=f"Time until quota reset ({hours_to_reset:.2f}h) is shorter than max session cap ({inst.max_session_h:.2f}h).",
                                target_id=inst.instance_id,
                                remediation="Ensure scheduled sessions fit within remaining hours before reset.",
                            )
                        )
                except Exception:
                    issues.append(
                        PreflightIssue(
                            severity="WARNING",
                            code="MALFORMED_RESET_TIMESTAMP",
                            message=f"Instance declares unparseable reset_at timestamp: '{inst.reset_at}'.",
                            target_id=inst.instance_id,
                            remediation="Provide an ISO 8601 formatted timestamp string (e.g. '2026-10-17T00:00:00Z').",
                        )
                    )

        # 2. Job Validation and Hardware Envelopes
        total_work_h = sum(j.duration_h for j in jobs) if jobs else 0.0

        if not jobs:
            issues.append(
                PreflightIssue(
                    severity="INFO",
                    code="EMPTY_WORKLOAD",
                    message="Workload job list is empty.",
                )
            )

        # Plan-compatible quota estimation:
        # Instead of naively comparing sum of raw worker-hours to quota hours,
        # calculate estimated billed quota accounting for concurrency across devices.
        if instances and jobs:
            total_slots = sum(inst.max_concurrent_workers for inst in instances)
            # Optimistic concurrent debit: total_work_h / slots * debit_rate
            avg_debit = sum(inst.debit_rate for inst in instances) / len(instances)
            est_billed_quota_h = (total_work_h / max(1, total_slots // len(instances))) * avg_debit

            if est_billed_quota_h > total_usable_quota:
                issues.append(
                    PreflightIssue(
                        severity="WARNING",
                        code="FLEET_QUOTA_TIGHT",
                        message=(
                            f"Estimated billed quota ({est_billed_quota_h:.2f}h across {len(jobs)} jobs totaling "
                            f"{total_work_h:.2f} worker-hours) exceeds usable fleet quota ({total_usable_quota:.2f}h)."
                        ),
                        remediation="Enable allow_partial=True or partition the workload across multiple reset cycles.",
                    )
                )

        max_instance_vram = max((inst.vram_per_device_gb for inst in instances), default=16.0)
        max_instance_session = max((inst.max_session_h for inst in instances), default=11.0)

        for job in jobs:
            # Check individual job length against session limits:
            # Crucial rule: resumable=True does NOT bypass the blocker unless valid checkpoint segments are modeled!
            session_threshold = max_instance_session * self.max_session_overhead_ratio
            if job.duration_h > session_threshold:
                has_valid_segments = (
                    job.segments is not None
                    and len(job.segments) > 0
                    and all(s <= session_threshold for s in job.segments)
                )
                if not has_valid_segments:
                    issues.append(
                        PreflightIssue(
                            severity="BLOCKER",
                            code="JOB_EXCEEDS_SESSION_LIMIT",
                            message=(
                                f"Job duration ({job.duration_h:.2f}h) exceeds maximum instance session cap "
                                f"({max_instance_session:.2f}h) "
                                f"{'even though resumable=True (no valid segments fitting session cap defined)' if job.resumable else 'without checkpointing'}."
                            ),
                            target_id=job.job_id,
                            remediation="Partition job into segments (job.segments) with durations <= session cap or reduce step count.",
                        )
                    )

            # Check VRAM feasibility
            if job.vram_gb > max_instance_vram:
                issues.append(
                    PreflightIssue(
                        severity="BLOCKER",
                        code="JOB_EXCEEDS_VRAM",
                        message=f"Job requires {job.vram_gb:.2f} GB VRAM, but maximum instance accelerator has {max_instance_vram:.2f} GB.",
                        target_id=job.job_id,
                        remediation="Reduce batch size, sequence rollout length, or precision.",
                    )
                )

        passed = not any(i.severity == "BLOCKER" for i in issues)

        return PreflightReport(
            passed=passed,
            issues=issues,
            summary={
                "instances_count": len(instances),
                "accounts_count": len(accounts),
                "jobs_count": len(jobs),
                "total_workload_hours": round(total_work_h, 2),
                "usable_fleet_quota_hours": round(total_usable_quota, 2),
            },
        )
