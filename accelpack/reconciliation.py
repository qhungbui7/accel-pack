"""Post-execution reconciliation and lightweight telemetry learner for AccelPack.

Reconciles predicted runtimes, throughputs, and quota debits against actual
measurements from completed jobs, continuously updating profile records without
running separate diagnostic benchmark sessions.
"""

from __future__ import annotations

import contextlib
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore
    HAS_FCNTL = False

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    msvcrt = None  # type: ignore
    HAS_MSVCRT = False
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


class HistoryLoadException(Exception):
    """Raised when reading or parsing execution history from persistent storage fails."""
    pass


class ProfileLoadException(Exception):
    """Raised when reading, parsing, or validating profile database from persistent storage fails."""
    pass


class LockAcquisitionException(Exception):
    """Raised when an advisory cross-process lock cannot be acquired or locking is unsupported."""
    pass


@contextlib.contextmanager
def file_lock(lock_target_path: Optional[Path | str]):
    """Cross-process advisory lock using fcntl.flock (POSIX) or msvcrt.locking (Windows) on a sibling .lock file.
    Provides robust exclusion across concurrent CLI processes.
    Fails explicitly with LockAcquisitionException if lock file cannot be opened or locked,
    or if no platform locking mechanism is available, closing any opened file descriptor.
    """
    if not lock_target_path:
        yield
        return

    path = Path(lock_target_path)
    lock_path = path.with_name(f"{path.name}.lock")
    fd = None

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise LockAcquisitionException(f"Cannot create lock directory for {lock_path}: {e}") from e

    if HAS_FCNTL and fcntl is not None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        except (PermissionError, OSError) as e:
            raise LockAcquisitionException(f"Cannot open lock file {lock_path}: {e}") from e

        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except (PermissionError, OSError) as e:
            try:
                os.close(fd)
            except Exception:
                pass
            fd = None
            raise LockAcquisitionException(f"Failed to acquire advisory lock on {lock_path}: {e}") from e
    elif HAS_MSVCRT and msvcrt is not None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        except (PermissionError, OSError) as e:
            raise LockAcquisitionException(f"Cannot open lock file {lock_path}: {e}") from e

        try:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        except (PermissionError, OSError) as e:
            try:
                os.close(fd)
            except Exception:
                pass
            fd = None
            raise LockAcquisitionException(f"Failed to acquire Windows file lock on {lock_path}: {e}") from e
    else:
        raise LockAcquisitionException(
            f"Advisory locking is unavailable on platform '{sys.platform}' without fcntl or msvcrt. "
            f"Cross-process concurrency protection cannot be guaranteed."
        )

    try:
        yield
    finally:
        if fd is not None:
            try:
                if HAS_FCNTL and fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                elif HAS_MSVCRT and msvcrt is not None:
                    try:
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass
            finally:
                os.close(fd)


def atomic_save_json(target_path: Path | str, data: Any) -> None:
    """Durably and atomically writes JSON data to target_path using a temporary file,
    fsync on file and parent directory, and atomic rename."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex[:8]}")
    payload = json.dumps(data, indent=2)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass
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
                content = self.history_file.read_text(encoding="utf-8")
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError(f"History file {self.history_file} must contain a JSON array, got {type(data).__name__}")
                for idx, r in enumerate(data):
                    if not isinstance(r, dict):
                        raise ValueError(f"Record at index {idx} in {self.history_file} is not a valid JSON object")
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
        return file_lock(self.history_file)

    def _reload_history_unlocked(self):
        """Reloads latest history from disk under active lock to avoid stale snapshots,
        preserving object identity for existing in-memory records."""
        if self.history_file and self.history_file.exists():
            try:
                content = self.history_file.read_text(encoding="utf-8")
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError(f"History file {self.history_file} must contain a JSON array, got {type(data).__name__}")
                existing_by_run_id = {r.run_id: r for r in self.history}
                recs = []
                for idx, r in enumerate(data):
                    if not isinstance(r, dict):
                        raise ValueError(f"Record at index {idx} in {self.history_file} is not a valid JSON object")
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
            try:
                self.history_file.parent.mkdir(parents=True, exist_ok=True)
                data = [r.__dict__ for r in self.history[-500:]]  # Retain rolling 500 records
                atomic_save_json(self.history_file, data)
                self.persist_error = None
            except Exception as e:
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
            if self.load_error is not None:
                raise HistoryLoadException(f"Cannot record job: failed to load existing history from {self.history_file}: {self.load_error}")

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
        if self.load_error is not None:
            raise HistoryLoadException(f"Cannot replay into estimator: failed to load history from {self.history_file}: {self.load_error}")

        migrations_count = 0
        # Step 1: Explicit one-time migration of legacy profile markers
        for key, profile in estimator._profiles.items():
            if profile.legacy_job_ids:
                env_k, model_k, hw_k, workers_k = key
                unmigrated = list(profile.legacy_job_ids)
                for legacy_marker in unmigrated:
                    # Check 1: Does legacy_marker match a historical run_id directly?
                    matching_by_run_id = [
                        r for r in self.history
                        if r.run_id == legacy_marker
                        and r.env_id.strip().lower() == env_k
                        and r.model.strip().lower() == model_k
                        and r.hardware_class.strip().lower() == hw_k
                        and r.concurrent_workers_on_device == workers_k
                        and r.success
                        and r.measured_fps and r.measured_fps > 0
                    ]
                    # Check 2: Does legacy_marker match a historical job_id?
                    matching_by_job_id = [
                        r for r in self.history
                        if r.job_id == legacy_marker
                        and r.env_id.strip().lower() == env_k
                        and r.model.strip().lower() == model_k
                        and r.hardware_class.strip().lower() == hw_k
                        and r.concurrent_workers_on_device == workers_k
                        and r.success
                        and r.measured_fps and r.measured_fps > 0
                    ]

                    if matching_by_run_id:
                        run_rec = matching_by_run_id[0]
                        if run_rec.run_id not in profile.applied_run_ids:
                            profile.applied_run_ids.append(run_rec.run_id)
                        migrations_count += 1
                    elif matching_by_job_id:
                        earliest = matching_by_job_id[0]
                        if earliest.run_id not in profile.applied_run_ids:
                            profile.applied_run_ids.append(earliest.run_id)
                        if legacy_marker not in profile.migrated_legacy_job_ids:
                            profile.migrated_legacy_job_ids.append(legacy_marker)
                        migrations_count += 1
                        if len(matching_by_job_id) > 1:
                            sys.stderr.write(
                                f"[WARN AccelPack] Legacy profile for {env_k}/{model_k} (workers={workers_k}) has "
                                f"unmigrated marker '{legacy_marker}'. Associated legacy calibration with run '{earliest.run_id}'. "
                                f"Subsequent {len(matching_by_job_id) - 1} execution(s) will be replayed.\n"
                            )
                    else:
                        if legacy_marker not in profile.migrated_legacy_job_ids:
                            profile.migrated_legacy_job_ids.append(legacy_marker)
                        migrations_count += 1
                        sys.stderr.write(
                            f"[WARN AccelPack] Legacy profile marker for job '{legacy_marker}' has no matching historical runs.\n"
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


def reconcile_and_persist_profiles(
    profiles_path: Path | str,
    reconciler: ExecutionReconciler,
    estimator: ProfileDurationEstimator,
    save_fn: Any = atomic_save_json,
) -> ReplayResult:
    """Full read-modify-write transactional coordinator for estimator profiles.
    Serializes profile reload, fresh history replay, and atomic persistence under a profile lock,
    guaranteeing that concurrent CLI processes cannot overwrite or lose telemetry updates.
    Aborts cleanly with ProfileLoadException if an existing profile database cannot be parsed
    or validated, leaving the original file byte-for-byte unchanged.
    """
    path = Path(profiles_path)
    with file_lock(path):
        # 1. Reload latest persisted profiles from disk into a fresh temporary estimator
        temp_estimator = ProfileDurationEstimator(default_fps=estimator.default_fps)
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                prof_data = json.loads(content)
                if not isinstance(prof_data, list):
                    raise ValueError(f"Profiles database {path} must contain a JSON array, got {type(prof_data).__name__}")
                for idx, p in enumerate(prof_data):
                    if not isinstance(p, dict):
                        raise ValueError(f"Profile record at index {idx} in {path} is not a valid JSON object")
                    if "env_id" not in p or "model" not in p or "fps" not in p:
                        raise ValueError(f"Profile record at index {idx} in {path} missing required fields ('env_id', 'model', 'fps')")
                    fps_val = float(p["fps"])
                    if fps_val <= 0:
                        raise ValueError(f"Profile record at index {idx} in {path} has invalid fps {fps_val}")
                    temp_estimator.register_profile(
                        env_id=p["env_id"],
                        model=p["model"],
                        fps=fps_val,
                        vram_gb=p.get("vram_gb", 0.5),
                        ram_gb=p.get("ram_gb", 1.5),
                        cpu_cores=p.get("cpu_cores", 1.0),
                        hardware_class=p.get("hardware_class", "nvidia_t4"),
                        workers_per_device=p.get("workers_per_device", 1),
                        samples=p.get("samples", 1),
                        confidence=p.get("confidence", 1.0),
                        applied_run_ids=p.get("applied_run_ids"),
                        applied_job_ids=p.get("applied_job_ids"),
                        migrated_legacy_job_ids=p.get("migrated_legacy_job_ids"),
                        legacy_job_ids=p.get("legacy_job_ids"),
                        schema_version=p.get("schema_version"),
                    )
            except Exception as e:
                err_msg = f"Failed to load and validate profiles database {path}: {e}"
                sys.stderr.write(f"[ERROR AccelPack] {err_msg}\n")
                raise ProfileLoadException(err_msg) from e
        else:
            # If path does not exist yet on disk, inherit current in-memory estimator state as initial baseline
            temp_estimator._profiles = dict(estimator._profiles)

        # 2. Obtain fresh snapshot of execution history under history lock (lock released immediately)
        with reconciler._file_lock():
            reconciler._reload_history_unlocked()
            if reconciler.load_error is not None:
                raise HistoryLoadException(reconciler.load_error)

        # 3. Replay all unapplied historical executions into temp_estimator in deterministic order
        replay_res = reconciler.replay_into_estimator(temp_estimator)

        # 4. Atomically persist updated profiles
        save_fn(path, temp_estimator.to_dict_list())

        # 5. Only upon successful persistence, replace active estimator's profile state
        estimator._profiles = temp_estimator._profiles

        return replay_res

