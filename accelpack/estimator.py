"""Duration estimator based on empirical stepping throughput, hardware profiles, and confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from accelpack.models import Job


@dataclass
class ProfileRecord:
    throughput: float             # Steps/sec or FPS
    samples: int = 1
    confidence: float = 1.0       # 0.0 to 1.0 confidence score
    vram_gb: float = 0.5
    ram_gb: float = 1.5
    cpu_cores: float = 1.0


class ProfileDurationEstimator:
    """Estimates job duration and resource footprint from environment steps and benchmark profiles."""

    def __init__(self, default_fps: float = 100.0):
        self.default_fps = default_fps
        # Key: (env_id, model_architecture) -> ProfileRecord
        self._profiles: Dict[Tuple[str, str], ProfileRecord] = {}

    def register_profile(
        self,
        env_id: str,
        model: str,
        fps: float,
        vram_gb: float = 0.5,
        ram_gb: float = 1.5,
        cpu_cores: float = 1.0,
        samples: int = 1,
        confidence: float = 1.0,
    ) -> None:
        """Registers empirical execution profile."""
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        key = (env_id.strip().lower(), model.strip().lower())
        self._profiles[key] = ProfileRecord(
            throughput=float(fps),
            samples=samples,
            confidence=max(0.0, min(1.0, confidence)),
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
        )

    def estimate_duration_h(self, steps: int, env_id: str = "", model: str = "") -> float:
        """Calculates expected execution time in hours."""
        key = (env_id.strip().lower(), model.strip().lower())
        rec = self._profiles.get(key)
        fps = rec.throughput if rec else self.default_fps
        return float(steps) / (fps * 3600.0)

    def create_job(
        self,
        job_id: str,
        steps: int,
        env_id: str = "",
        model: str = "",
        vram_gb: Optional[float] = None,
        ram_gb: Optional[float] = None,
        cpu_cores: Optional[float] = None,
        priority: float = 1.0,
    ) -> Job:
        """Factory helper creating a Job with predicted duration and resource demands."""
        key = (env_id.strip().lower(), model.strip().lower())
        rec = self._profiles.get(key)

        fps = rec.throughput if rec else self.default_fps
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
            metadata={
                "env": env_id,
                "model": model,
                "steps": steps,
                "fps": fps,
                "confidence": confidence,
            },
        )
