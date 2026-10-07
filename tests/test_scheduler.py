import pytest
from accelpack.models import Job, Instance, DeviceSlot, InstancePlan, ExecutionPlan, PackingStrategy
from accelpack.scheduler import AccelPackScheduler, QuotaExceededException
from accelpack.estimator import ProfileDurationEstimator

def test_job_validation():
    with pytest.raises(ValueError):
        Job(job_id="bad_job", duration_h=-1.0)
    j = Job(job_id="valid", duration_h=2.5)
    assert j.duration_h == 2.5
    assert j.vram_gb == 0.5


def test_instance_validation():
    with pytest.raises(ValueError):
        Instance(instance_id="bad", device_count=0)
    inst = Instance(instance_id="node1", device_count=2, remaining_quota_h=20.0)
    assert inst.remaining_quota_h == 20.0


def test_minimal_instance_preservation():
    """Verify that 6 small jobs are packed onto 1 instance instead of spreading across 3."""
    instances = [
        Instance(instance_id="node-1", device_count=2, remaining_quota_h=25.0, max_session_h=11.0),
        Instance(instance_id="node-2", device_count=2, remaining_quota_h=25.0, max_session_h=11.0),
        Instance(instance_id="node-3", device_count=2, remaining_quota_h=25.0, max_session_h=11.0),
    ]

def test_concentrate_minimal_instance_preservation():
    """Verify that in CONCENTRATE mode, jobs are packed onto minimal instances."""
    instances = [
        Instance(instance_id="node-1", device_count=2, remaining_quota_h=25.0, max_session_h=11.0),
        Instance(instance_id="node-2", device_count=2, remaining_quota_h=25.0, max_session_h=11.0),
        Instance(instance_id="node-3", device_count=2, remaining_quota_h=25.0, max_session_h=11.0),
    ]

    jobs = [Job(job_id=f"job_{i}", duration_h=0.2) for i in range(6)]

    scheduler = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10)
    plan = scheduler.schedule(jobs, instances)

    assert len(plan.active_instances) == 1
    assert len(plan.untouched_instances) == 2
    assert plan.active_instances[0].instance.instance_id == "node-1"


def test_balance_waterfill_equal_usage_default():
    """Verify that in BALANCE mode (default), jobs are distributed across instances to equalize remaining quotas."""
    instances = [
        Instance(instance_id="used_node", device_count=2, remaining_quota_h=20.0, max_session_h=11.0),
        Instance(instance_id="fresh_node_1", device_count=2, remaining_quota_h=29.0, max_session_h=11.0),
        Instance(instance_id="fresh_node_2", device_count=2, remaining_quota_h=29.0, max_session_h=11.0),
    ]

    # 6 jobs of 1.0 hour each.
    # BALANCE mode should spread across the fresh nodes (29h remaining) rather than overloading the used node (20h).
    jobs = [Job(job_id=f"job_{i}", duration_h=1.0) for i in range(6)]

    scheduler = AccelPackScheduler(strategy=PackingStrategy.BALANCE, setup_overhead_h=0.10)
    plan = scheduler.schedule(jobs, instances)

    # Waterfill should target fresh_node_1 and fresh_node_2 first!
    # Each fresh node gets 2 or 4 jobs, leaving used_node with minimal or 0 new burden.
    active_ids = {ip.instance.instance_id for ip in plan.active_instances}
    assert "fresh_node_1" in active_ids
    assert "fresh_node_2" in active_ids
    # Both fresh nodes get work, preserving balance
    for ip in plan.active_instances:
        # Dual devices must be balanced inside each container
        assert ip.load_imbalance_h <= 1.0


def test_dual_device_load_balancing():
    """Verify that jobs with unequal lengths are balanced across devices to minimize idle device billing."""
    instances = [Instance(instance_id="node-1", device_count=2, remaining_quota_h=25.0)]
    
    jobs = [
        Job(job_id="j1", duration_h=2.0),
        Job(job_id="j2", duration_h=1.5),
        Job(job_id="j3", duration_h=1.0),
        Job(job_id="j4", duration_h=0.5),
    ]

    scheduler = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10)
    plan = scheduler.schedule(jobs, instances)

    slots = plan.active_instances[0].slots
    assert slots[0].load_h == pytest.approx(2.5)
    assert slots[1].load_h == pytest.approx(2.5)
    assert plan.active_instances[0].load_imbalance_h == pytest.approx(0.0)

def test_session_limit_spillover():
    """Verify that when workload exceeds one instance's session limit (11h), it spills cleanly into a second instance."""
    instances = [
        Instance(instance_id="node-1", device_count=2, remaining_quota_h=30.0, max_session_h=4.0),
        Instance(instance_id="node-2", device_count=2, remaining_quota_h=30.0, max_session_h=4.0),
    ]

    # 4 jobs of 3.0h each. Total work = 12h.
    # Node 1 can hold at most 2 * 4.0h = 8h. So it must open node-2!
    jobs = [Job(job_id=f"job_{i}", duration_h=3.0) for i in range(4)]

    scheduler = AccelPackScheduler(setup_overhead_h=0.10)
    plan = scheduler.schedule(jobs, instances)

    assert len(plan.active_instances) == 2
    assert len(plan.untouched_instances) == 0


def test_quota_safety_buffer():
    """Verify scheduler refuses to assign jobs that would violate an account's quota buffer."""
    instances = [
        Instance(instance_id="low_quota_node", device_count=2, remaining_quota_h=3.0, max_session_h=11.0),
    ]

    # 1 job of 2.0h -> billed resource time = 2 * (2.0 + 0.1) = 4.2h > (3.0 - 2.0) = 1.0h
    jobs = [Job(job_id="j1", duration_h=2.0)]

    scheduler = AccelPackScheduler(quota_buffer_h=2.0)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs, instances)


def test_profile_duration_estimator():
    estimator = ProfileDurationEstimator(default_fps=100.0)
    estimator.register_profile("cartpole", "linear_ema", fps=2000.0)

    # 1,024,000 steps at 2000 FPS: 1024000 / (2000 * 3600) = 0.1422h (~8.5 min)
    dur = estimator.estimate_duration_h(steps=1_024_000, env_id="cartpole", model="linear_ema")
    assert dur == pytest.approx(0.14222, rel=1e-3)

    job = estimator.create_job("j1", steps=1_024_000, env_id="cartpole", model="linear_ema")
    assert job.duration_h == pytest.approx(0.14222, rel=1e-3)
    assert job.metadata["env"] == "cartpole"
