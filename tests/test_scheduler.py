"""Unit tests for AccelPack models and multi-accelerator scheduler using standard unittest."""

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

    def test_concentrate_minimal_instance_preservation(self):
        """Verify that in CONCENTRATE mode, jobs are packed onto minimal instances."""
        instances = [
            Instance(instance_id="node-1", device_count=2, remaining_quota_h=25.0, max_session_h=11.0),
            Instance(instance_id="node-2", device_count=2, remaining_quota_h=25.0, max_session_h=11.0),
            Instance(instance_id="node-3", device_count=2, remaining_quota_h=25.0, max_session_h=11.0),
        ]

        jobs = [Job(job_id=f"job_{i}", duration_h=0.2) for i in range(6)]

        scheduler = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10)
        plan = scheduler.schedule(jobs, instances)

        self.assertEqual(len(plan.active_instances), 1)
        self.assertEqual(len(plan.untouched_instances), 2)
        self.assertEqual(plan.active_instances[0].instance.instance_id, "node-1")

    def test_balance_waterfill_equal_usage_default(self):
        """Verify that in BALANCE mode (default), jobs are distributed across instances to equalize remaining quotas."""
        instances = [
            Instance(instance_id="used_node", device_count=2, remaining_quota_h=20.0, max_session_h=11.0),
            Instance(instance_id="fresh_node_1", device_count=2, remaining_quota_h=29.0, max_session_h=11.0),
            Instance(instance_id="fresh_node_2", device_count=2, remaining_quota_h=29.0, max_session_h=11.0),
        ]

        jobs = [Job(job_id=f"job_{i}", duration_h=1.0) for i in range(6)]

        scheduler = AccelPackScheduler(strategy=PackingStrategy.BALANCE, setup_overhead_h=0.10)
        plan = scheduler.schedule(jobs, instances)

        for ip in plan.active_instances:
            self.assertLessEqual(ip.load_imbalance_h, 1.0)

    def test_concurrent_slot_packing_speedup(self):
        """Verify that setting slots_per_device=2 halves the physical makespan by packing 4 workers."""
        inst_serial = [Instance(instance_id="node_serial", device_count=2, slots_per_device=1, remaining_quota_h=30.0)]
        inst_concurrent = [Instance(instance_id="node_concurrent", device_count=2, slots_per_device=2, remaining_quota_h=30.0)]

        jobs = [Job(job_id=f"job_{i}", duration_h=1.0, vram_gb=2.0, ram_gb=2.0) for i in range(4)]

        # Serial (1 worker per GPU = 2 parallel)
        sched_serial = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10)
        plan_serial = sched_serial.schedule(jobs, inst_serial)
        self.assertAlmostEqual(plan_serial.global_makespan_h, 2.10, places=2)

        # Concurrent (2 workers per GPU = 4 parallel)
        sched_concurrent = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10, slots_per_device=2)
        plan_concurrent = sched_concurrent.schedule(jobs, inst_concurrent)
        self.assertAlmostEqual(plan_concurrent.global_makespan_h, 1.10, places=2)

    def test_aggregate_vram_and_ram_protection(self):
        """Verify that jobs exceeding physical device VRAM or host RAM cannot be packed concurrently."""
        inst = [Instance(instance_id="gpu_node", device_count=2, vram_per_device_gb=4.0, host_ram_gb=8.0, slots_per_device=2)]

        heavy_jobs = [Job(job_id=f"heavy_{i}", duration_h=1.0, vram_gb=2.5, ram_gb=2.0) for i in range(4)]

        scheduler = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10, slots_per_device=2)
        plan = scheduler.schedule(heavy_jobs, inst, allow_partial=True)

        for ip in plan.active_instances:
            for d in range(ip.instance.device_count):
                self.assertLessEqual(ip.peak_vram_per_device_gb[d], ip.instance.vram_per_device_gb)


if __name__ == "__main__":
    unittest.main()
