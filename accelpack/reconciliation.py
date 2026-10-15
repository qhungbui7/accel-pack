"""Post-execution reconciliation and lightweight telemetry learner for AccelPack.

Reconciles predicted runtimes, throughputs, and quota debits against actual
measurements from completed jobs, continuously updating profile records without
running separate diagnostic benchmark sessions.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    hardware_class: str = "nvidia_t4"
    peak_vram_gb: Optional[float] = None
    peak_ram_gb: Optional[float] = None
    cpu_busy_pct: Optional[float] = None
    measured_fps: Optional[float] = None
    missing_telemetry: List[str] = field(default_factory=list)
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
        self.load_error: Optional[str] = None
        self.persist_error: Optional[str] = None
        self._load_history()

    def _load_history(self):
        if self.history_file and self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                for r in data:
                    self.history.append(JobExecutionRecord(**r))
            except Exception as e:
                self.load_error = f"Corrupted or invalid history file {self.history_file}: {e}"
                # Preserve empty in-memory list but warn visibly
                sys.stderr.write(f"[WARN AccelPack] {self.load_error}\n")

    def _persist_history(self):
        if self.history_file:
            try:
                self.history_file.parent.mkdir(parents=True, exist_ok=True)
                data = [r.__dict__ for r in self.history[-500:]]  # Retain rolling 500 records
                self.history_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                self.persist_error = None
            except Exception as e:
                self.persist_error = f"Failed to write history file {self.history_file}: {e}"
                sys.stderr.write(f"[ERROR AccelPack] {self.persist_error}\n")

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
        hardware_class: str = "nvidia_t4",
        peak_vram_gb: Optional[float] = None,
        peak_ram_gb: Optional[float] = None,
        cpu_busy_pct: Optional[float] = None,
        estimator: Optional[ProfileDurationEstimator] = None,
    ) -> JobExecutionRecord:
        """Records a completed job execution and updates the duration estimator profile."""
        actual_fps = (steps / (actual_duration_h * 3600.0)) if actual_duration_h > 0 else 0.0

        missing = []
        if peak_vram_gb is None:
            missing.append("peak_vram_gb")
        if peak_ram_gb is None:
            missing.append("peak_ram_gb")
        if cpu_busy_pct is None:
            missing.append("cpu_busy_pct")

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
            hardware_class=hardware_class,
            peak_vram_gb=peak_vram_gb,
            peak_ram_gb=peak_ram_gb,
            cpu_busy_pct=cpu_busy_pct,
            measured_fps=round(actual_fps, 1),
            missing_telemetry=missing,
        )
        self.history.append(rec)
        self._persist_history()

        # Update estimator if provided and job succeeded
        if estimator and success and actual_fps > 0:
            existing = estimator.get_profile(
                env_id=env_id,
                model=model,
                hardware_class=hardware_class,
                workers_per_device=concurrent_workers_on_device,
            )
            prior_fps = existing.throughput if existing else (estimator.default_fps / (1.0 + 0.3 * (concurrent_workers_on_device - 1)))
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
                hardware_class=hardware_class,
                workers_per_device=concurrent_workers_on_device,
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

    def get_prediction_error_report(self) -> Dict[str, Any]:
        """Generates statistical prediction accuracy and error metrics across completed jobs."""
        if not self.history:
            return {
                "total_records": 0,
                "overall": {"mae_hours": 0.0, "rmse_hours": 0.0, "mape_pct": 0.0, "success_rate": 0.0},
                "by_stratum": {},
            }

        errors = []
        sq_errors = []
        pct_errors = []
        succeeded = 0

        # Stratified by (env_id, model, hardware_class, workers_on_device)
        strata: Dict[Tuple[str, str, str, int], List[JobExecutionRecord]] = {}

        for r in self.history:
            diff = abs(r.actual_duration_h - r.predicted_duration_h)
            errors.append(diff)
            sq_errors.append(diff ** 2)
            if r.actual_duration_h > 0:
                pct_errors.append(diff / r.actual_duration_h)
            if r.success:
                succeeded += 1

            key = (r.env_id, r.model, r.hardware_class, r.concurrent_workers_on_device)
            strata.setdefault(key, []).append(r)

        overall_mae = sum(errors) / len(errors)
        overall_rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
        overall_mape = (sum(pct_errors) / len(pct_errors) * 100.0) if pct_errors else 0.0
        success_rate = succeeded / len(self.history)

        strata_report = {}
        for (env, model, hw, workers), recs in strata.items():
            st_errors = [abs(r.actual_duration_h - r.predicted_duration_h) for r in recs]
            st_mae = sum(st_errors) / len(st_errors)
            strata_report[f"{env}/{model} [{hw}, {workers}w/dev]"] = {
                "count": len(recs),
                "mae_hours": round(st_mae, 4),
                "success_rate": round(sum(1 for r in recs if r.success) / len(recs), 3),
            }

        return {
            "total_records": len(self.history),
            "overall": {
                "mae_hours": round(overall_mae, 4),
                "rmse_hours": round(overall_rmse, 4),
                "mape_pct": round(overall_mape, 2),
                "success_rate": round(success_rate, 3),
            },
            "by_stratum": strata_report,
        }

    def replay_into_estimator(self, estimator: ProfileDurationEstimator) -> int:
        """Replays all successful historical job executions to calibrate estimator profiles."""
        count = 0
        for rec in self.history:
            if rec.success and rec.measured_fps and rec.measured_fps > 0:
                existing = estimator.get_profile(
                    env_id=rec.env_id,
                    model=rec.model,
                    hardware_class=rec.hardware_class,
                    workers_per_device=rec.concurrent_workers_on_device,
                )
                prior_fps = existing.throughput if existing else (estimator.default_fps / (1.0 + 0.3 * (rec.concurrent_workers_on_device - 1)))
                new_fps = (1.0 - self.learning_rate) * prior_fps + self.learning_rate * rec.measured_fps
                new_samples = (existing.samples + 1) if existing else 1
                new_conf = min(0.95, (existing.confidence + 0.05) if existing else 0.70)
                new_vram = max(existing.vram_gb, rec.peak_vram_gb or 0.5) if existing else (rec.peak_vram_gb or 0.5)
                new_ram = max(existing.ram_gb, rec.peak_ram_gb or 1.5) if existing else (rec.peak_ram_gb or 1.5)
                new_cpu = existing.cpu_cores if existing else 1.0

                estimator.register_profile(
                    env_id=rec.env_id,
                    model=rec.model,
                    fps=new_fps,
                    vram_gb=new_vram,
                    ram_gb=new_ram,
                    cpu_cores=new_cpu,
                    hardware_class=rec.hardware_class,
                    workers_per_device=rec.concurrent_workers_on_device,
                    samples=new_samples,
                    confidence=new_conf,
                )
                count += 1
        return count

