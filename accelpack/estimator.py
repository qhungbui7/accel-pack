"""Duration estimator based on empirical stepping throughput, hardware profiles, and confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from accelpack.models import Job


def is_execution_run_id(item: str) -> bool:
    """Determines whether an identifier represents an execution run ID or a logical job ID.
    Run IDs start with 'run_' (UUID-based execution IDs) or 'legacy_' with hash format 'legacy_{job}_{idx}_{hash}'."""
    if item.startswith("run_"):
        return True
    if item.startswith("legacy_"):
        parts = item.split("_")
        if len(parts) >= 4 and len(parts[-1]) == 8 and all(c in "0123456789abcdef" for c in parts[-1]):
            return True
    return False


@dataclass
class ProfileRecord:
    throughput: float             # Steps/sec or FPS
    samples: int = 1
    confidence: float = 1.0       # 0.0 to 1.0 confidence score
    vram_gb: float = 0.5
    ram_gb: float = 1.5
    cpu_cores: float = 1.0
    hardware_class: str = "nvidia_t4"
    workers_per_device: int = 1
    applied_run_ids: List[str] = field(default_factory=list)
    legacy_job_ids: List[str] = field(default_factory=list)
    migrated_legacy_job_ids: List[str] = field(default_factory=list)

    @property
    def applied_job_ids(self) -> List[str]:
        """Backward-compatible alias for applied_run_ids and legacy job markers."""
        return list(dict.fromkeys(self.applied_run_ids + self.legacy_job_ids + self.migrated_legacy_job_ids))


class ProfileDurationEstimator:
    """Estimates job duration and resource footprint from environment steps and benchmark profiles,
    stratified by task, model architecture, hardware class, and concurrent workers per device.
    """

    def __init__(self, default_fps: float = 100.0):
        self.default_fps = default_fps
        # Key: (env_id, model_architecture, hardware_class, workers_per_device) -> ProfileRecord
        self._profiles: Dict[Tuple[str, str, str, int], ProfileRecord] = {}

    def _normalize_key(
        self,
        env_id: str,
        model: str,
        hardware_class: str = "nvidia_t4",
        workers_per_device: int = 1,
    ) -> Tuple[str, str, str, int]:
        return (
            env_id.strip().lower(),
            model.strip().lower(),
            hardware_class.strip().lower(),
            max(1, workers_per_device),
        )

    def register_profile(
        self,
        env_id: str,
        model: str,
        fps: float,
        vram_gb: float = 0.5,
        ram_gb: float = 1.5,
        cpu_cores: float = 1.0,
        hardware_class: str = "nvidia_t4",
        workers_per_device: int = 1,
        samples: int = 1,
        confidence: float = 1.0,
        applied_run_ids: Optional[List[str]] = None,
        applied_job_ids: Optional[List[str]] = None,
        migrated_legacy_job_ids: Optional[List[str]] = None,
        legacy_job_ids: Optional[List[str]] = None,
        schema_version: Optional[int] = None,
    ) -> None:
        """Registers empirical execution profile isolated by hardware class and concurrency level.
        Execution IDs in applied_run_ids are preserved as authoritative execution identities
        without string-pattern heuristics. Legacy job markers from applied_job_ids or legacy_job_ids
        are tracked separately for one-time historical migration.
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        key = self._normalize_key(env_id, model, hardware_class, workers_per_device)

        # 1. applied_run_ids is authoritative: every entry is an execution ID
        clean_run_ids: List[str] = list(dict.fromkeys(applied_run_ids or []))

        # 2. migrated_legacy_job_ids records legacy markers already converted
        clean_migrated_ids: List[str] = list(dict.fromkeys(migrated_legacy_job_ids or []))

        # 3. legacy job markers come from legacy_job_ids or deprecated applied_job_ids
        raw_legacy = list(dict.fromkeys(legacy_job_ids or []))
        if applied_job_ids:
            for item in applied_job_ids:
                if item not in clean_run_ids and item not in clean_migrated_ids and item not in raw_legacy:
                    raw_legacy.append(item)

        clean_legacy_ids: List[str] = [
            item for item in raw_legacy
            if item not in clean_run_ids and item not in clean_migrated_ids
        ]

        # If any legacy job id already has a corresponding run in clean_run_ids, mark as migrated
        for leg_id in list(clean_legacy_ids):
            if any(rid.startswith(f"legacy_{leg_id}_") or rid == leg_id for rid in clean_run_ids):
                clean_legacy_ids.remove(leg_id)
                if leg_id not in clean_migrated_ids:
                    clean_migrated_ids.append(leg_id)

        self._profiles[key] = ProfileRecord(
            throughput=float(fps),
            samples=samples,
            confidence=max(0.0, min(1.0, confidence)),
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            hardware_class=hardware_class.strip().lower(),
            workers_per_device=max(1, workers_per_device),
            applied_run_ids=clean_run_ids,
            legacy_job_ids=clean_legacy_ids,
            migrated_legacy_job_ids=clean_migrated_ids,
        )

    def get_profile(
        self,
        env_id: str = "",
        model: str = "",
        hardware_class: str = "nvidia_t4",
        workers_per_device: int = 1,
    ) -> Optional[ProfileRecord]:
        """Looks up profile with fallback from exact match -> isolated 1-worker -> generic fallback."""
        norm_key = self._normalize_key(env_id, model, hardware_class, workers_per_device)
        if norm_key in self._profiles:
            return self._profiles[norm_key]

        # Fallback 1: Match same (env, model, hardware_class) with 1 worker
        single_worker_key = self._normalize_key(env_id, model, hardware_class, 1)
        if single_worker_key in self._profiles:
            base_rec = self._profiles[single_worker_key]
            # Apply conservative fallback degradation: 1.30x for 2 workers, etc.
            slowdown = 1.0 + 0.30 * (workers_per_device - 1)
            return ProfileRecord(
                throughput=base_rec.throughput / slowdown,
                samples=base_rec.samples,
                confidence=base_rec.confidence * 0.85,
                vram_gb=base_rec.vram_gb,
                ram_gb=base_rec.ram_gb,
                cpu_cores=base_rec.cpu_cores,
                hardware_class=hardware_class.strip().lower(),
                workers_per_device=workers_per_device,
            )

        # Fallback 2: Match any hardware class
        for (e, m, h, w), rec in self._profiles.items():
            if e == env_id.strip().lower() and m == model.strip().lower():
                slowdown = 1.0 + 0.30 * (workers_per_device - 1)
                return ProfileRecord(
                    throughput=rec.throughput / slowdown,
                    samples=rec.samples,
                    confidence=rec.confidence * 0.70,
                    vram_gb=rec.vram_gb,
                    ram_gb=rec.ram_gb,
                    cpu_cores=rec.cpu_cores,
                    hardware_class=hardware_class.strip().lower(),
                    workers_per_device=workers_per_device,
                )

        return None

    def estimate_duration_h(
        self,
        steps: int,
        env_id: str = "",
        model: str = "",
        hardware_class: str = "nvidia_t4",
        workers_per_device: int = 1,
    ) -> float:
        """Calculates expected execution time in hours."""
        rec = self.get_profile(env_id, model, hardware_class, workers_per_device)
        fps = rec.throughput if rec else (self.default_fps / (1.0 + 0.30 * (workers_per_device - 1)))
        return float(steps) / (fps * 3600.0)

    def create_job(
        self,
        job_id: str,
        steps: int,
        env_id: str = "",
        model: str = "",
        hardware_class: str = "nvidia_t4",
        workers_per_device: int = 1,
        vram_gb: Optional[float] = None,
        ram_gb: Optional[float] = None,
        cpu_cores: Optional[float] = None,
        priority: float = 1.0,
        resumable: bool = False,
        segments: Optional[List[float]] = None,
    ) -> Job:
        """Factory helper creating a Job with predicted duration and resource demands."""
        rec = self.get_profile(env_id, model, hardware_class, workers_per_device)
        slowdown = 1.0 + 0.30 * (workers_per_device - 1)
        fps = rec.throughput if rec else (self.default_fps / slowdown)
        dur = float(steps) / (fps * 3600.0)

        final_vram = vram_gb if vram_gb is not None else (rec.vram_gb if rec else 0.5)
        final_ram = ram_gb if ram_gb is not None else (rec.ram_gb if rec else 1.5)
        final_cpu = cpu_cores if cpu_cores is not None else (rec.cpu_cores if rec else 1.0)
        confidence = rec.confidence if rec else 0.40

        return Job(
            job_id=job_id,
            duration_h=dur,
            vram_gb=final_vram,
            ram_gb=final_ram,
            cpu_cores=final_cpu,
            priority=priority,
            resumable=resumable,
            segments=segments,
            hardware_class=hardware_class,
            metadata={
                "env": env_id,
                "model": model,
                "steps": steps,
                "fps": fps,
                "confidence": confidence,
                "hardware_class": hardware_class,
                "workers_per_device": workers_per_device,
                "has_profile": rec is not None,
            },
        )

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Exports all registered profiles to a serializable list of dictionaries."""
        result = []
        for (env_id, model, hw, workers), rec in self._profiles.items():
            result.append({
                "schema_version": 2,
                "env_id": env_id,
                "model": model,
                "fps": rec.throughput,
                "vram_gb": rec.vram_gb,
                "ram_gb": rec.ram_gb,
                "cpu_cores": rec.cpu_cores,
                "hardware_class": hw,
                "workers_per_device": workers,
                "samples": rec.samples,
                "confidence": rec.confidence,
                "applied_run_ids": list(rec.applied_run_ids),
                "applied_job_ids": list(rec.applied_job_ids),
                "migrated_legacy_job_ids": list(rec.migrated_legacy_job_ids),
            })
        return result

