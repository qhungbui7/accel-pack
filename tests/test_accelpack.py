"""Comprehensive regression and unit tests for AccelPack multi-accelerator scheduling engine."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from accelpack.estimator import ProfileDurationEstimator
from accelpack.models import (
    DeviceSlot,
    ExecutionPlan,
    Instance,
    InstancePlan,
    Job,
    ObjectiveMode,
    PackingStrategy,
)
from accelpack.preflight import PreflightValidator
from accelpack.reconciliation import ExecutionReconciler
from accelpack.scheduler import AccelPackScheduler, QuotaExceededException


# ---------------------------------------------------------------------------
# Core Regression Tests from Review
# ---------------------------------------------------------------------------

def test_cpu_demand_exceeds_100_percent_allowed():
    """Regression Test 1: CPU demand exceeds 100% nominal capacity; placement remains possible without hard ceiling."""
    instances = [
        Instance(instance_id="node-1", device_count=2, host_cpu_cores=4.0, remaining_quota_h=20.0),
    ]

    # Two concurrent jobs demanding 3.0 cores each = 6.0 cores on 4.0 vCPU host (150% demand)
    jobs = [
        Job(job_id="j1", duration_h=1.0, cpu_cores=3.0, vram_gb=1.0, ram_gb=2.0),
        Job(job_id="j2", duration_h=1.0, cpu_cores=3.0, vram_gb=1.0, ram_gb=2.0),
    ]

    scheduler = AccelPackScheduler(slots_per_device=1)
    plan = scheduler.schedule(jobs, instances)

    assert len(plan.active_instances) == 1
    ip = plan.active_instances[0]
    assert ip.peak_host_cpu_cores == 6.0
    assert ip.cpu_oversubscription_ratio == 1.5
    assert ip.is_cpu_oversubscribed is True
    assert ip.cpu_contention_penalty == pytest.approx(0.25)


def test_cpu_demand_exceeds_emergency_cap_rejected():
    """Regression Test 2: CPU demand exceeds configured emergency cap; placement is rejected."""
    instances = [
        Instance(
            instance_id="node-1",
            device_count=2,
            host_cpu_cores=4.0,
            max_session_h=1.5,  # Prevents 1.0h + 1.0h from running serially in one slot
            remaining_quota_h=20.0,
            max_cpu_oversubscription_ratio=1.40,  # Max 4 * 1.4 = 5.6 cores
        ),
    ]

    # Two concurrent jobs demanding 3.0 cores each = 6.0 cores > 5.6 cores ceiling -> Rejection!
    jobs = [
        Job(job_id="j1", duration_h=1.0, cpu_cores=3.0),
        Job(job_id="j2", duration_h=1.0, cpu_cores=3.0),
    ]

    scheduler = AccelPackScheduler(slots_per_device=1)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs, instances, allow_partial=False)


def test_high_host_ram_sequential_queue_accepted():
    """Regression Test 3: High host RAM, sequential queue.
    A job can use an existing slot if its sequential peak fits, verifying that
    BALANCE mode no longer falsely rejects sequential jobs by naively adding
    candidate RAM to existing peak RAM.
    """
    # 30 GB host RAM, safety margin 1.0 GB -> 29 GB usable.
    instances = [
        Instance(instance_id="node-1", device_count=1, host_ram_gb=30.0, remaining_quota_h=20.0),
    ]

    # Job 1 uses 18 GB RAM.
    # Job 2 uses 16 GB RAM.
    # If added together naively (18 + 16 = 34 > 30), it would be rejected.
    # But because they run sequentially in the single queue (slot 0), the peak RAM is max(18, 16) = 18 GB <= 29 GB!
    jobs = [
        Job(job_id="j1", duration_h=1.0, ram_gb=18.0, vram_gb=2.0),
        Job(job_id="j2", duration_h=1.0, ram_gb=16.0, vram_gb=2.0),
    ]

    scheduler = AccelPackScheduler(strategy=PackingStrategy.BALANCE, safety_ram_margin_gb=1.0)
    plan = scheduler.schedule(jobs, instances)

    assert len(plan.active_instances) == 1
    assert len(plan.active_instances[0].slots[0].jobs) == 2
    assert plan.active_instances[0].peak_host_ram_gb == 18.0


def test_shared_account_inconsistent_balances_normalized_or_rejected():
    """Regression Test 4: Shared account has inconsistent balances.
    Scheduling blocks when strict, or normalizes safely to the authoritative minimum.
    """
    instances = [
        Instance(instance_id="node_a", account_id="shared_acc", device_count=1, remaining_quota_h=5.0),
        Instance(instance_id="node_b", account_id="shared_acc", device_count=1, remaining_quota_h=20.0),
    ]

    # 1. Strict mode blocks
    jobs = [Job(job_id="j1", duration_h=1.0)]
    strict_sched = AccelPackScheduler(reject_conflicting_account_quotas=True)
    with pytest.raises(QuotaExceededException):
        strict_sched.schedule(jobs, instances)

    # 2. Normalizing mode uses min balance (5.0h) instead of generous declaration (20.0h)
    norm_sched = AccelPackScheduler(reject_conflicting_account_quotas=False, quota_buffer_h=2.0)
    # With device_count=1, two 2-hour jobs require 4.1h billed makespan, which cannot fit within 5.0h - 2.0h buffer!
    two_jobs = [Job(job_id="j1", duration_h=2.0), Job(job_id="j2", duration_h=2.0)]
    with pytest.raises(QuotaExceededException):
        norm_sched.schedule(two_jobs, instances, allow_partial=False)


def test_reset_timestamp_invalid_or_stale_fails_safely():
    """Regression Test 5: Reset timestamp invalid or stale.
    Enforced scheduling fails safely (does NOT fail open).
    """
    now = datetime.now(timezone.utc)
    past_reset = (now - timedelta(hours=2.0)).isoformat()

    # Case A: Reset timestamp in the past
    stale_inst = [
        Instance(instance_id="stale_node", reset_at=past_reset, must_finish_before_reset=True, remaining_quota_h=20.0),
    ]
    scheduler = AccelPackScheduler(enforce_reset_deadline=True)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule([Job(job_id="j1", duration_h=1.0)], stale_inst)

    # Case B: Malformed reset timestamp
    malformed_inst = [
        Instance(instance_id="bad_node", reset_at="NOT_A_DATE", must_finish_before_reset=True, remaining_quota_h=20.0),
    ]
    with pytest.raises(QuotaExceededException):
        scheduler.schedule([Job(job_id="j1", duration_h=1.0)], malformed_inst)


def test_segmented_job_scheduled_correctly():
    """Regression Test 6: Segmented job exceeds session duration.
    Job duration is 12h, but max session limit is 10h.
    Because segments=[6.0, 6.0] are declared, scheduler expands the segments into
    session-fitting units and schedules both successfully!
    """
    instances = [
        Instance(instance_id="node-1", device_count=1, max_session_h=10.0, remaining_quota_h=30.0),
        Instance(instance_id="node-2", device_count=1, max_session_h=10.0, remaining_quota_h=30.0),
    ]

    job_12h = Job(job_id="long_sim", duration_h=12.0, resumable=True, segments=[6.0, 6.0])

    # With expand_segments=True (default), each 6.0h segment fits within 10.0h session limit!
    scheduler = AccelPackScheduler(expand_segments=True)
    plan = scheduler.schedule([job_12h], instances)

    # Both segments must be scheduled
    scheduled_job_ids = [j.job_id for ip in plan.active_instances for s in ip.slots for j in s.jobs]
    assert "long_sim_seg0" in scheduled_job_ids
    assert "long_sim_seg1" in scheduled_job_ids
    assert len(plan.unassigned_jobs) == 0

    # If expand_segments is False, 12h job cannot fit into 10h session limit and is rejected
    no_expand_sched = AccelPackScheduler(expand_segments=False)
    with pytest.raises(QuotaExceededException):
        no_expand_sched.schedule([job_12h], instances, allow_partial=False)


def test_hardware_class_mismatch():
    """Regression Test 7: Hardware class mismatch.
    An incompatible instance is never selected, and preflight blocks if no compatible node exists.
    """
    instances = [
        Instance(instance_id="cpu_node", hardware_class="cpu_only", device_count=1, remaining_quota_h=20.0),
        Instance(instance_id="t4_node", hardware_class="nvidia_t4", device_count=1, remaining_quota_h=20.0),
    ]

    # Job requires nvidia_t4
    gpu_job = Job(job_id="gpu_work", duration_h=1.0, hardware_class="nvidia_t4")

    scheduler = AccelPackScheduler()
    plan = scheduler.schedule([gpu_job], instances)

    # Must be scheduled on t4_node, NEVER cpu_node
    assert len(plan.active_instances) == 1
    assert plan.active_instances[0].instance.instance_id == "t4_node"

    # Preflight blocks if only incompatible hardware exists
    validator = PreflightValidator()
    report = validator.validate([gpu_job], [instances[0]])  # cpu_node only
    assert report.passed is False
    assert any(i.code == "NO_COMPATIBLE_HARDWARE" for i in report.blockers)


def test_efficiency_objective_distinct_decision():
    """Regression Test 8: Efficiency objective produces a meaningfully distinct, tested decision.
    In MAXIMIZE_EFFICIENCY mode, the scheduler prioritizes packing jobs onto an already-active instance
    and into concurrent slots to maximize useful work per billed hour.
    """
    # Two nodes: node-1 is already open with job 1; node-2 is untouched.
    # In MAXIMIZE_EFFICIENCY, packing into node-1 avoids paying another cold-start setup overhead.
    instances = [
        Instance(instance_id="node-1", device_count=2, slots_per_device=2, remaining_quota_h=25.0),
        Instance(instance_id="node-2", device_count=2, slots_per_device=2, remaining_quota_h=25.0),
    ]

    jobs = [
        Job(job_id=f"job_{i}", duration_h=1.0) for i in range(4)
    ]

    sched_eff = AccelPackScheduler(
        strategy=PackingStrategy.BALANCE,
        objective=ObjectiveMode.MAXIMIZE_EFFICIENCY,
        slots_per_device=2,
    )
    plan_eff = sched_eff.schedule(jobs, instances)

    # Packing all 4 jobs onto node-1's 4 concurrent slots yields highest productivity (1 instance active)
    assert len(plan_eff.active_instances) == 1
    assert plan_eff.active_instances[0].instance.instance_id == "node-1"
    assert plan_eff.quota_productivity_ratio > 1.0


# ---------------------------------------------------------------------------
# Additional Unit & Lifecycle Tests
# ---------------------------------------------------------------------------

def test_job_validation():
    with pytest.raises(ValueError):
        Job(job_id="bad_job", duration_h=-1.0)

    j = Job(job_id="valid", duration_h=2.5)
    assert j.duration_h == 2.5
    assert j.vram_gb == 0.5

    with pytest.raises(ValueError):
        Job(job_id="seg_mismatch", duration_h=5.0, segments=[2.0, 2.0])


def test_instance_validation():
    with pytest.raises(ValueError):
        Instance(instance_id="bad", device_count=0)

    inst = Instance(instance_id="node1", device_count=2, remaining_quota_h=20.0)
    assert inst.remaining_quota_h == 20.0
    assert inst.account_id == "node1"
    assert inst.get_concurrency_slowdown(1) == 1.0
    assert inst.get_concurrency_slowdown(2) == 1.30


def test_vram_and_ram_hard_constraints_enforced():
    instances = [
        Instance(
            instance_id="node-1",
            device_count=2,
            vram_per_device_gb=16.0,
            host_ram_gb=30.0,
            remaining_quota_h=20.0,
        ),
    ]

    # VRAM limit violation
    jobs_vram_exceeded = [Job(job_id="heavy_vram", duration_h=1.0, vram_gb=15.8)]
    scheduler = AccelPackScheduler(safety_vram_margin_gb=0.5)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs_vram_exceeded, instances)

    # Host RAM limit violation
    jobs_ram_exceeded = [Job(job_id="heavy_ram", duration_h=1.0, ram_gb=29.5)]
    scheduler_ram = AccelPackScheduler(safety_ram_margin_gb=1.0)
    with pytest.raises(QuotaExceededException):
        scheduler_ram.schedule(jobs_ram_exceeded, instances)


def test_concurrency_slowdown_models():
    instances = [
        Instance(
            instance_id="node-1",
            device_count=1,
            slots_per_device=2,
            remaining_quota_h=20.0,
            concurrency_slowdown_factors={1: 1.0, 2: 1.30},
        ),
    ]

    jobs = [
        Job(job_id="j1", duration_h=1.0),
        Job(job_id="j2", duration_h=1.0),
    ]

    scheduler = AccelPackScheduler(setup_overhead_h=0.10)
    plan = scheduler.schedule(jobs, instances)

    ip = plan.active_instances[0]
    assert ip.device_loads_h[0] == pytest.approx(1.30)
    assert ip.makespan_h == pytest.approx(1.40)
    assert plan.quota_productivity_ratio == pytest.approx(2.0 / 1.40)


def test_shared_account_quota_pool():
    instances = [
        Instance(instance_id="node-a", account_id="kaggle_user", device_count=1, remaining_quota_h=8.0),
        Instance(instance_id="node-b", account_id="kaggle_user", device_count=1, remaining_quota_h=8.0),
    ]

    jobs = [
        Job(job_id="j1", duration_h=3.0),
        Job(job_id="j2", duration_h=3.0),
    ]

    scheduler = AccelPackScheduler(quota_buffer_h=3.0, setup_overhead_h=0.10)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs, instances, allow_partial=False)


def test_reconciliation_prediction_error_report(tmp_path):
    history_file = tmp_path / "history.json"
    reconciler = ExecutionReconciler(history_file=history_file)

    reconciler.record_job(
        job_id="j1",
        env_id="cartpole",
        model="gru128",
        steps=100000,
        predicted_duration_h=1.0,
        actual_duration_h=1.1,
        success=True,
        instance_id="node1",
        device_index=0,
        concurrent_workers_on_device=2,
        hardware_class="nvidia_t4",
        peak_vram_gb=2.0,
        peak_ram_gb=3.0,
        cpu_busy_pct=85.0,
    )

    reconciler.record_job(
        job_id="j2",
        env_id="cartpole",
        model="gru128",
        steps=100000,
        predicted_duration_h=1.0,
        actual_duration_h=1.2,
        success=True,
        instance_id="node1",
        device_index=0,
        concurrent_workers_on_device=2,
        hardware_class="nvidia_t4",
    )

    assert "peak_vram_gb" in reconciler.history[1].missing_telemetry
    assert "cpu_busy_pct" in reconciler.history[1].missing_telemetry

    report = reconciler.get_prediction_error_report()
    assert report["total_records"] == 2
    assert report["overall"]["mae_hours"] == pytest.approx(0.15)
    assert report["overall"]["success_rate"] == 1.0


def test_cli_preflight_blocks_and_export(tmp_path, monkeypatch):
    from accelpack import cli

    inst_file = tmp_path / "instances.json"
    inst_file.write_text(
        json.dumps([{"instance_id": "inst_1", "device_count": 2, "remaining_quota_h": 20.0, "max_session_h": 10.0}])
    )

    bad_jobs_file = tmp_path / "bad_jobs.json"
    bad_jobs_file.write_text(json.dumps([{"job_id": "job_too_big", "duration_h": 15.0, "resumable": False}]))

    monkeypatch.setattr("sys.argv", ["accelpack.cli", "--instances", str(inst_file), "--jobs", str(bad_jobs_file)])

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1

    good_jobs_file = tmp_path / "good_jobs.json"
    good_jobs_file.write_text(json.dumps([{"job_id": "job_1", "duration_h": 1.0}]))
    output_file = tmp_path / "plan_out.json"

    monkeypatch.setattr(
        "sys.argv",
        ["accelpack.cli", "--instances", str(inst_file), "--jobs", str(good_jobs_file), "--output", str(output_file)],
    )

    cli.main()
    assert output_file.exists()
    plan_data = json.loads(output_file.read_text())
    assert plan_data["total_work_h"] == 1.0
    assert len(plan_data["instances"]) == 1


def test_preflight_inconsistent_account_quota_and_reset_warnings():
    validator = PreflightValidator()
    now = datetime.now(timezone.utc)
    past_reset = (now - timedelta(hours=2.0)).isoformat()
    future_reset = (now + timedelta(hours=5.0)).isoformat()

    instances = [
        Instance(instance_id="node_1", account_id="shared_acc", remaining_quota_h=20.0, reset_at=past_reset),
        Instance(instance_id="node_2", account_id="shared_acc", remaining_quota_h=25.0, reset_at=future_reset),
    ]

    jobs = [Job(job_id="j1", duration_h=1.0)]
    report = validator.validate(jobs, instances)

    assert any(i.code == "INCONSISTENT_ACCOUNT_QUOTA" for i in report.issues)
    assert any(i.code == "RESET_TIME_PASSED" for i in report.issues)


def test_readme_quickstart_example():
    """Verify that the quickstart example in the README produces exact expected outputs."""
    instances = [
        Instance(instance_id="node-01", device_count=2, remaining_quota_h=24.0, max_session_h=11.0),
        Instance(instance_id="node-02", device_count=2, remaining_quota_h=24.0, max_session_h=11.0),
        Instance(instance_id="node-03", device_count=2, remaining_quota_h=24.0, max_session_h=11.0),
    ]

    jobs = [
        Job(job_id="train_model_s1", duration_h=1.8, vram_gb=0.6, ram_gb=1.5),
        Job(job_id="train_model_s2", duration_h=1.8, vram_gb=0.6, ram_gb=1.5),
        Job(job_id="train_model_s3", duration_h=1.8, vram_gb=0.6, ram_gb=1.5),
        Job(job_id="eval_model_s1", duration_h=0.5, vram_gb=0.3, ram_gb=1.0),
        Job(job_id="eval_model_s2", duration_h=0.5, vram_gb=0.3, ram_gb=1.0),
        Job(job_id="eval_model_s3", duration_h=0.5, vram_gb=0.3, ram_gb=1.0),
    ]

    scheduler = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10, quota_buffer_h=2.0)
    plan = scheduler.schedule(jobs, instances)

    assert len(plan.active_instances) == 1
    assert plan.active_instances[0].instance.instance_id == "node-01"
    assert plan.global_makespan_h == pytest.approx(3.70)
    assert plan.total_billed_resource_h == pytest.approx(3.70)
    assert plan.quota_productivity_ratio == pytest.approx(6.90 / 3.70, rel=1e-3)
    assert len(plan.unassigned_jobs) == 0


def test_gang_scheduling_unsupported():
    """Verify that single-job multi-device gang scheduling explicitly raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Job(job_id="gang_job", duration_h=1.0, device_count=2)

