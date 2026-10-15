"""Post-execution reconciliation and lightweight telemetry learner for AccelPack.

Reconciles predicted runtimes, throughputs, and quota debits against actual
measurements from completed jobs, continuously updating profile records without
running separate diagnostic benchmark sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from accelpack.estimator import ProfileDurationEstimator


@dataclass
class JobExecutionRecord:
    job_id: str
    env_id: str
    model: str
    steps: int
    predicted_duration_h: float
    actual_duration_h: float
    success: bool
    instance_id: str
    device_index: int = 0
    concurrent_workers_on_device: int = 1
    peak_vram_gb: Optional[float] = None
    peak_ram_gb: Optional[float] = None
    measured_fps: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SessionReconciliationReport:
    instance_id: str
    planned_makespan_h: float
    actual_makespan_h: float
    planned_quota_debit_h: float
    actual_quota_debit_h: float
    jobs_succeeded: int
    jobs_failed: int
    runtime_ratio: float      # actual / planned
    debit_ratio: float        # actual / planned
    job_records: List[JobExecutionRecord] = field(default_factory=list)


class ExecutionReconciler:
    """Collects real job measurements and continuously calibrates runtime profiles."""

    def __init__(self, history_file: Optional[Path | str] = None, learning_rate: float = 0.25):
        self.history_file = Path(history_file) if history_file else None
        self.learning_rate = learning_rate
        self.history: List[JobExecutionRecord] = []
        self._load_history()

    def _load_history(self):
        if self.history_file and self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                for r in data:
                    self.history.append(JobExecutionRecord(**r))
            except Exception:
                self.history = []

    def _persist_history(self):
        if self.history_file:
            try:
                self.history_file.parent.mkdir(parents=True, exist_ok=True)
                data = [r.__dict__ for r in self.history[-500:]]  # Retain rolling 500 records
                self.history_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass

    def record_job(
        self,
        job_id: str,
        env_id: str,
        model: str,
        steps: int,
        predicted_duration_h: float,
        actual_duration_h: float,
        success: bool,
        instance_id: str,
        device_index: int = 0,
        concurrent_workers_on_device: int = 1,
        peak_vram_gb: Optional[float] = None,
        peak_ram_gb: Optional[float] = None,
        estimator: Optional[ProfileDurationEstimator] = None,
    ) -> JobExecutionRecord:
        """Records a completed job execution and updates the duration estimator profile."""
        actual_fps = (steps / (actual_duration_h * 3600.0)) if actual_duration_h > 0 else 0.0

        rec = JobExecutionRecord(
            job_id=job_id,
            env_id=env_id,
            model=model,
            steps=steps,
            predicted_duration_h=predicted_duration_h,
            actual_duration_h=actual_duration_h,
            success=success,
            instance_id=instance_id,
            device_index=device_index,
            concurrent_workers_on_device=concurrent_workers_on_device,
            peak_vram_gb=peak_vram_gb,
            peak_ram_gb=peak_ram_gb,
            measured_fps=round(actual_fps, 1),
        )
        self.history.append(rec)
        self._persist_history()

        # Update estimator if provided and job succeeded
        if estimator and success and actual_fps > 0:
            key = (env_id.strip().lower(), model.strip().lower())
            existing = estimator._profiles.get(key)
            prior_fps = existing.throughput if existing else estimator.default_fps
            new_fps = (1.0 - self.learning_rate) * prior_fps + self.learning_rate * actual_fps
            new_samples = (existing.samples + 1) if existing else 1
            new_conf = min(0.95, (existing.confidence + 0.05) if existing else 0.70)
            new_vram = max(existing.vram_gb, peak_vram_gb or 0.5) if existing else (peak_vram_gb or 0.5)
            new_ram = max(existing.ram_gb, peak_ram_gb or 1.5) if existing else (peak_ram_gb or 1.5)
            new_cpu = existing.cpu_cores if existing else 1.0
            estimator.register_profile(
                env_id=env_id,
                model=model,
                fps=new_fps,
                vram_gb=new_vram,
                ram_gb=new_ram,
                cpu_cores=new_cpu,
                samples=new_samples,
                confidence=new_conf,
            )

        return rec

    def reconcile_session(
        self,
        instance_id: str,
        planned_makespan_h: float,
        actual_makespan_h: float,
        planned_debit_h: float,
        actual_debit_h: float,
        job_records: List[JobExecutionRecord],
    ) -> SessionReconciliationReport:
        succeeded = sum(1 for r in job_records if r.success)
        failed = len(job_records) - succeeded
        r_ratio = (actual_makespan_h / planned_makespan_h) if planned_makespan_h > 0 else 1.0
        d_ratio = (actual_debit_h / planned_debit_h) if planned_debit_h > 0 else 1.0

        return SessionReconciliationReport(
            instance_id=instance_id,
            planned_makespan_h=round(planned_makespan_h, 3),
            actual_makespan_h=round(actual_makespan_h, 3),
            planned_quota_debit_h=round(planned_debit_h, 3),
            actual_quota_debit_h=round(actual_debit_h, 3),
            jobs_succeeded=succeeded,
            jobs_failed=failed,
            runtime_ratio=round(r_ratio, 2),
            debit_ratio=round(d_ratio, 2),
            job_records=job_records,
        )
