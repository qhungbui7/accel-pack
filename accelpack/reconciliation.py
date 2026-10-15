"""Post-execution reconciliation and lightweight telemetry learner for AccelPack.

Reconciles predicted runtimes, throughputs, and quota debits against actual
measurements from completed jobs, continuously updating profile records without
running separate diagnostic benchmark sessions.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from accelpack.estimator import ProfileDurationEstimator


class HistoryPersistenceException(Exception):
    """Raised when writing execution history to persistent storage fails."""
    pass


def atomic_save_json(target_path: Path | str, data: Any) -> None:
    """Durably and atomically writes JSON data to target_path using a temporary file and atomic rename."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex[:8]}")
    payload = json.dumps(data, indent=2)
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


class ReplayResult(int):
    """Result of replay_into_estimator representing newly replayed executions count
    while tracking whether profile state was modified by legacy migrations.
    Subclasses int for full backward compatibility."""

    replayed_count: int
    migrations_count: int
    modified: bool

    def __new__(cls, replayed: int, migrations: int = 0):
        obj = super().__new__(cls, replayed)
        obj.replayed_count = replayed
        obj.migrations_count = migrations
        obj.modified = (replayed > 0) or (migrations > 0)
        return obj


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
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
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
                for idx, r in enumerate(data):
                    if "run_id" not in r or not r["run_id"]:
                        jid = r.get("job_id", "job")
                        ts = r.get("timestamp", str(idx))
                        steps = r.get("steps", 0)
                        dur = r.get("actual_duration_h", 0.0)
                        content_sig = f"{jid}:{ts}:{steps}:{dur}:{idx}"
                        r["run_id"] = f"legacy_{jid}_{idx}_{hashlib.sha256(content_sig.encode('utf-8')).hexdigest()[:8]}"
                    self.history.append(JobExecutionRecord(**r))
            except Exception as e:
                self.load_error = f"Corrupted or invalid history file {self.history_file}: {e}"
                # Preserve empty in-memory list but warn visibly
                sys.stderr.write(f"[WARN AccelPack] {self.load_error}\n")

    def _file_lock(self):
        """Cross-process advisory lock to serialize read, duplicate check, append, and persistence."""
        if not self.history_file:
            @contextlib.contextmanager
            def _noop():
                yield
            return _noop()

        lock_path = self.history_file.with_name(f"{self.history_file.name}.lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        @contextlib.contextmanager
        def _flock():
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
        return _flock()

    def _reload_history_unlocked(self):
        """Reloads latest history from disk under active lock to avoid stale snapshots,
        preserving object identity for existing in-memory records."""
        if self.history_file and self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                existing_by_run_id = {r.run_id: r for r in self.history}
                recs = []
                for idx, r in enumerate(data):
                    rid = r.get("run_id")
                    if not rid:
                        jid = r.get("job_id", "job")
                        ts = r.get("timestamp", str(idx))
                        steps = r.get("steps", 0)
                        dur = r.get("actual_duration_h", 0.0)
                        content_sig = f"{jid}:{ts}:{steps}:{dur}:{idx}"
                        rid = f"legacy_{jid}_{idx}_{hashlib.sha256(content_sig.encode('utf-8')).hexdigest()[:8]}"
                        r["run_id"] = rid
                    if rid in existing_by_run_id:
                        recs.append(existing_by_run_id[rid])
                    else:
                        recs.append(JobExecutionRecord(**r))
                self.history = recs
                self.load_error = None
            except Exception as e:
                self.load_error = f"Corrupted or invalid history file {self.history_file}: {e}"
                sys.stderr.write(f"[WARN AccelPack] {self.load_error}\n")

    def _persist_history(self):
        if self.history_file:
            tmp_file = None
            try:
                self.history_file.parent.mkdir(parents=True, exist_ok=True)
                data = [r.__dict__ for r in self.history[-500:]]  # Retain rolling 500 records
                payload = json.dumps(data, indent=2)

                # Atomic file replacement: write to unique temporary file in same directory, then atomic rename
                tmp_file = self.history_file.with_name(f"{self.history_file.name}.tmp.{uuid.uuid4().hex[:8]}")
                tmp_file.write_text(payload, encoding="utf-8")
                os.replace(tmp_file, self.history_file)
                self.persist_error = None
            except Exception as e:
                if tmp_file and tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except Exception:
                        pass
                self.persist_error = f"Failed to write history file {self.history_file}: {e}"
                sys.stderr.write(f"[ERROR AccelPack] {self.persist_error}\n")
                raise HistoryPersistenceException(self.persist_error) from e

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
        run_id: Optional[str] = None,
        device_index: int = 0,
        concurrent_workers_on_device: int = 1,
        hardware_class: str = "nvidia_t4",
        peak_vram_gb: Optional[float] = None,
        peak_ram_gb: Optional[float] = None,
        cpu_busy_pct: Optional[float] = None,
        estimator: Optional[ProfileDurationEstimator] = None,
    ) -> JobExecutionRecord:
        """Records a completed job execution and updates the duration estimator profile idempotently.
        Serializes read, duplicate check, append, and persistence under cross-process file lock.
        Separates history deduplication from estimator calibration so retrying with an estimator
        repairs missing calibration state without duplicating history records.
        """
        effective_run_id = run_id or f"run_{job_id}_{uuid.uuid4().hex[:12]}"

        with self._file_lock():
            # 1. Reload latest state under lock to prevent lost updates from concurrent writers
            self._reload_history_unlocked()

            # 2. Check if already recorded in self.history (idempotent submission by run_id)
            existing_rec: Optional[JobExecutionRecord] = None
            for existing_h in self.history:
                if existing_h.run_id == effective_run_id:
                    existing_rec = existing_h
                    break

            if existing_rec is not None:
                rec = existing_rec
            else:
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
                    run_id=effective_run_id,
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
                try:
                    self._persist_history()
                except HistoryPersistenceException:
                    self.history.pop()
                    raise

        # 3. Update estimator if provided and job succeeded (handles both new records and retried historical records)
        if estimator and rec.success and rec.measured_fps and rec.measured_fps > 0:
            existing = estimator.get_profile(
                env_id=rec.env_id,
                model=rec.model,
                hardware_class=rec.hardware_class,
                workers_per_device=rec.concurrent_workers_on_device,
            )
            applied = list(existing.applied_run_ids) if existing else []
            if rec.run_id not in applied:
                applied.append(rec.run_id)
                migrated = list(existing.migrated_legacy_job_ids) if existing else []
                legacy = list(existing.legacy_job_ids) if existing else []

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
                    applied_run_ids=applied,
                    applied_job_ids=legacy,
                    migrated_legacy_job_ids=migrated,
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

    def replay_into_estimator(self, estimator: ProfileDurationEstimator) -> ReplayResult:
        """Replays all successful historical job executions to calibrate estimator profiles,
        skipping records that have already been incorporated into existing profiles.

        Performs explicit one-time legacy migration for profiles with legacy_job_ids:
        associates the legacy marker with the earliest matching historical run, emits a migration
        warning if multiple runs exist, and records the migrated run_id into applied_run_ids.
        Thereafter, deduplication is performed exclusively by execution run_id so subsequent
        runs of the same logical job are never silently suppressed.
        """
        migrations_count = 0
        # Step 1: Explicit one-time migration of legacy profile markers
        for key, profile in estimator._profiles.items():
            if profile.legacy_job_ids:
                env_k, model_k, hw_k, workers_k = key
                unmigrated = list(profile.legacy_job_ids)
                for legacy_jid in unmigrated:
                    matching_runs = [
                        r for r in self.history
                        if r.job_id == legacy_jid
                        and r.env_id.strip().lower() == env_k
                        and r.model.strip().lower() == model_k
                        and r.hardware_class.strip().lower() == hw_k
                        and r.concurrent_workers_on_device == workers_k
                        and r.success
                        and r.measured_fps and r.measured_fps > 0
                    ]
                    if matching_runs:
                        earliest = matching_runs[0]
                        if earliest.run_id not in profile.applied_run_ids:
                            profile.applied_run_ids.append(earliest.run_id)
                        if legacy_jid not in profile.migrated_legacy_job_ids:
                            profile.migrated_legacy_job_ids.append(legacy_jid)
                        migrations_count += 1
                        if len(matching_runs) > 1:
                            sys.stderr.write(
                                f"[WARN AccelPack] Legacy profile for {env_k}/{model_k} (workers={workers_k}) has "
                                f"unmigrated marker '{legacy_jid}'. Associated legacy calibration with run '{earliest.run_id}'. "
                                f"Subsequent {len(matching_runs) - 1} execution(s) will be replayed.\n"
                            )
                    else:
                        if legacy_jid not in profile.migrated_legacy_job_ids:
                            profile.migrated_legacy_job_ids.append(legacy_jid)
                        migrations_count += 1
                        sys.stderr.write(
                            f"[WARN AccelPack] Legacy profile marker for job '{legacy_jid}' has no matching historical runs.\n"
                        )
                # Clear legacy markers after migration so deduplication is strictly by run_id
                profile.legacy_job_ids = []

        # Step 2: Replay unapplied historical executions
        count = 0
        for rec in self.history:
            if rec.success and rec.measured_fps and rec.measured_fps > 0:
                existing = estimator.get_profile(
                    env_id=rec.env_id,
                    model=rec.model,
                    hardware_class=rec.hardware_class,
                    workers_per_device=rec.concurrent_workers_on_device,
                )
                if existing and rec.run_id in existing.applied_run_ids:
                    # Skip duplicate application
                    continue

                applied = list(existing.applied_run_ids) if existing else []
                applied.append(rec.run_id)
                migrated = list(existing.migrated_legacy_job_ids) if existing else []
                legacy = list(existing.legacy_job_ids) if existing else []

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
                    applied_run_ids=applied,
                    applied_job_ids=legacy,
                    migrated_legacy_job_ids=migrated,
                )
                count += 1
        return ReplayResult(count, migrations_count)

