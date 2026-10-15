"""Unit tests for AccelPack models, scheduler, preflight validation, and reconciliation."""

import tempfile
from pathlib import Path
import unittest

from accelpack.models import (
    DeviceSlot,
    ExecutionPlan,
    Instance,
    InstancePlan,
    Job,
    ObjectiveMode,
    PackingStrategy,
)
from accelpack.scheduler import AccelPackScheduler, QuotaExceededException
from accelpack.estimator import ProfileDurationEstimator
from accelpack.preflight import PreflightValidator
from accelpack.reconciliation import ExecutionReconciler


class TestAccelPack(unittest.TestCase):
    def test_job_validation(self):
        with self.assertRaises(ValueError):
            Job(job_id="bad_job", duration_h=-1.0)
        j = Job(job_id="valid", duration_h=2.5)
        self.assertEqual(j.duration_h, 2.5)
        self.assertEqual(j.vram_gb, 0.5)

    def test_instance_validation(self):
        with self.assertRaises(ValueError):
            Instance(instance_id="bad", device_count=0)
        inst = Instance(instance_id="node1", device_count=2, remaining_quota_h=20.0, slots_per_device=2)
        self.assertEqual(inst.remaining_quota_h, 20.0)
        self.assertEqual(inst.max_concurrent_workers, 4)

    def test_preflight_validation_pipeline(self):
        validator = PreflightValidator(min_quota_buffer_h=2.0)
        inst_good = Instance(instance_id="node-good", device_count=2, remaining_quota_h=25.0, max_session_h=11.0)
        inst_tight = Instance(instance_id="node-tight", device_count=2, remaining_quota_h=1.5, max_session_h=11.0)

        # 1. Job exceeds session limit without checkpointing
        job_oversized = Job(job_id="huge", duration_h=12.0, resumable=False)
        rep = validator.validate([job_oversized], [inst_good])
        self.assertFalse(rep.passed)
        self.assertTrue(any(i.code == "JOB_EXCEEDS_SESSION_LIMIT" for i in rep.blockers))

        # 2. Resumable job passes
        job_resumable = Job(job_id="huge_resumable", duration_h=12.0, resumable=True, segments=[6.0, 6.0])
        rep = validator.validate([job_resumable], [inst_good])
        self.assertTrue(rep.passed)

        # 3. Quota buffer warning
        job_normal = Job(job_id="norm", duration_h=1.0)
        rep = validator.validate([job_normal], [inst_tight])
        self.assertTrue(rep.passed)  # Warning is not a blocker
        self.assertTrue(any(i.code == "QUOTA_BUFFER_EXHAUSTED" for i in rep.warnings))

    def test_reconciliation_and_online_learning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            h_file = Path(tmp_dir) / "history.json"
            reconciler = ExecutionReconciler(history_file=h_file, learning_rate=0.5)
            estimator = ProfileDurationEstimator(default_fps=100.0)

            # Record a job that ran faster than estimated (200 FPS actual vs 100 FPS predicted)
            rec = reconciler.record_job(
                job_id="test_run_1",
                env_id="CartPole",
                model="MLP",
                steps=720_000,
                predicted_duration_h=2.0,
                actual_duration_h=1.0,  # 720k / 3600s = 200 FPS
                success=True,
                instance_id="node-1",
                peak_vram_gb=1.2,
                peak_ram_gb=2.4,
                estimator=estimator,
            )
            self.assertEqual(rec.measured_fps, 200.0)

            # Check that estimator profile was calibrated online via moving average (100 -> 150 FPS)
            updated_dur = estimator.estimate_duration_h(steps=720_000, env_id="CartPole", model="MLP")
            self.assertAlmostEqual(updated_dur, 720000.0 / (150.0 * 3600.0), places=2)

    def test_concurrent_slot_packing_speedup(self):
        inst_serial = [Instance(instance_id="node_serial", device_count=2, slots_per_device=1, remaining_quota_h=30.0)]
        # Default realistic slowdown (1.30x for 2 workers per GPU)
        inst_concurrent_realistic = [Instance(instance_id="node_concurrent", device_count=2, slots_per_device=2, remaining_quota_h=30.0)]
        # Ideal scaling (zero slowdown)
        inst_concurrent_ideal = [Instance(instance_id="node_ideal", device_count=2, slots_per_device=2, remaining_quota_h=30.0, concurrency_slowdown_factors={1: 1.0, 2: 1.0})]

        jobs = [Job(job_id=f"job_{i}", duration_h=1.0, vram_gb=2.0, ram_gb=2.0) for i in range(4)]

        # Serial (1 worker per GPU = 2 parallel)
        sched_serial = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10)
        plan_serial = sched_serial.schedule(jobs, inst_serial)
        self.assertAlmostEqual(plan_serial.global_makespan_h, 2.10, places=2)

        # Concurrent with realistic slowdown (1.30x -> makespan 1.30 + 0.10 = 1.40h)
        sched_concurrent = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10, slots_per_device=2)
        plan_concurrent = sched_concurrent.schedule(jobs, inst_concurrent_realistic)
        self.assertAlmostEqual(plan_concurrent.global_makespan_h, 1.40, places=2)

        # Concurrent with ideal scaling (1.0x -> makespan 1.00 + 0.10 = 1.10h)
        plan_ideal = sched_concurrent.schedule(jobs, inst_concurrent_ideal)
        self.assertAlmostEqual(plan_ideal.global_makespan_h, 1.10, places=2)


if __name__ == "__main__":
    unittest.main()
