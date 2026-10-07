"""Throughput database and duration estimator based on empirical steps and FPS."""

from __future__ import annotations

from typing import Dict, Tuple, Optional
from accelpack.models import Job


class ProfileDurationEstimator:
    """Estimates job duration from environment steps and throughput (FPS or updates/sec)."""

    def __init__(self, default_fps: float = 100.0):
        self.default_fps = default_fps
        # Key: (env_id, model_architecture) -> empirical FPS
        self._fps_table: Dict[Tuple[str, str], float] = {}

    def register_profile(self, env_id: str, model: str, fps: float) -> None:
        """Registers empirical execution profile."""
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        self._fps_table[(env_id.strip().lower(), model.strip().lower())] = float(fps)

    def estimate_duration_h(self, steps: int, env_id: str = "", model: str = "") -> float:
        """Calculates expected execution time in hours."""
        key = (env_id.strip().lower(), model.strip().lower())
        fps = self._fps_table.get(key, self.default_fps)
        return float(steps) / (fps * 3600.0)

    def create_job(
        self,
        job_id: str,
        steps: int,
        env_id: str = "",
        model: str = "",
        vram_gb: float = 0.5,
        ram_gb: float = 1.5,
        cpu_cores: float = 1.0,
    ) -> Job:
        """Factory helper creating a Job with predicted duration."""
        dur = self.estimate_duration_h(steps=steps, env_id=env_id, model=model)
        return Job(
            job_id=job_id,
            duration_h=dur,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            metadata={"env": env_id, "model": model, "steps": steps},
        )
