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


def test_session_limit_includes_setup_overhead_rejection():
    """Verify that jobs whose duration + setup overhead exceeds session limit are rejected."""
    # Instance max session is 1.0 hour, setup overhead is 0.10h.
    # Job duration is 0.95h.
    # 0.95h + 0.10h = 1.05h > 1.00h session limit -> Must be rejected!
    inst = [Instance(instance_id="n1", max_session_h=1.0, remaining_quota_h=20.0)]
    job = [Job(job_id="j_tight", duration_h=0.95)]

    # BALANCE strategy
    sched_bal = AccelPackScheduler(strategy=PackingStrategy.BALANCE, setup_overhead_h=0.10)
    with pytest.raises(QuotaExceededException):
        sched_bal.schedule(job, inst, allow_partial=False)

    # CONCENTRATE strategy
    sched_conc = AccelPackScheduler(strategy=PackingStrategy.CONCENTRATE, setup_overhead_h=0.10)
    with pytest.raises(QuotaExceededException):
        sched_conc.schedule(job, inst, allow_partial=False)


def test_json_concurrency_slowdown_string_keys_normalized():
    """Verify that concurrency slowdown factor dictionaries with string keys (from JSON) are normalized to int."""
    json_factors = {"1": 1.0, "2": 1.35, "3": 1.70}
    inst = Instance(instance_id="n1", concurrency_slowdown_factors=json_factors)

    # Must convert string keys to int
    assert 1 in inst.concurrency_slowdown_factors
    assert 2 in inst.concurrency_slowdown_factors
    assert inst.get_concurrency_slowdown(2) == 1.35
    assert inst.get_concurrency_slowdown(4) > 1.70  # Fallback calculation works without TypeError


def test_segmented_jobs_strict_ordering_and_sequential_slot_binding():
    """Verify that segmented jobs are topologically ordered and assigned sequentially in the same slot."""
    # Instance has 2 devices with 1 slot each. Session limit is 10.0h.
    # Job has two segments: seg0 (4.0h) and seg1 (5.0h).
    # If scheduled in parallel across device 0 and device 1, seg1 would start at t=0 concurrently with seg0.
    # The dependency satisfaction rule requires seg1 to be placed in the SAME slot as seg0 (4.0h + 5.0h = 9.0h <= 10.0h).
    instances = [Instance(instance_id="node1", device_count=2, max_session_h=10.0, remaining_quota_h=25.0)]
    job = Job(job_id="resumable_task", duration_h=9.0, resumable=True, segments=[4.0, 5.0])

    sched = AccelPackScheduler(expand_segments=True)
    plan = sched.schedule([job], instances)

    assert len(plan.active_instances) == 1
    ip = plan.active_instances[0]

    # Find which slot has seg0 and seg1
    slot_with_seg0 = next(s for s in ip.slots if any(j.job_id == "resumable_task_seg0" for j in s.jobs))
    slot_with_seg1 = next(s for s in ip.slots if any(j.job_id == "resumable_task_seg1" for j in s.jobs))

    # They MUST be assigned to the same slot
    assert slot_with_seg0 is slot_with_seg1
    # seg0 must appear BEFORE seg1 in the sequential queue
    job_ids = [j.job_id for j in slot_with_seg0.jobs]
    assert job_ids.index("resumable_task_seg0") < job_ids.index("resumable_task_seg1")


def test_closed_loop_telemetry_persists_profiles_and_modifies_schedule(tmp_path, monkeypatch):
    """Verify that recording a job execution persists calibrated profiles, which subsequently update scheduling."""
    from accelpack import cli

    profiles_file = tmp_path / "profiles.json"
    history_file = tmp_path / "history.json"
    inst_file = tmp_path / "inst.json"
    jobs_file = tmp_path / "jobs.json"
    out_file = tmp_path / "out.json"

    # Initial profile: CartPole MLP has 100 FPS (720,000 steps = 2.0h)
    initial_profiles = [
        {"env_id": "cartpole", "model": "mlp", "fps": 100.0, "hardware_class": "nvidia_t4", "workers_per_device": 1}
    ]
    profiles_file.write_text(json.dumps(initial_profiles), encoding="utf-8")

    # Record run that actually completed twice as fast (200 FPS, 1.0h)
    job_run = {
        "job_id": "prev_run",
        "env_id": "cartpole",
        "model": "mlp",
        "steps": 720000,
        "predicted_duration_h": 2.0,
        "actual_duration_h": 1.0,
        "success": True,
        "instance_id": "node_prev",
        "hardware_class": "nvidia_t4",
        "concurrent_workers": 1,
    }
    record_file = tmp_path / "run.json"
    record_file.write_text(json.dumps(job_run), encoding="utf-8")

    inst_file.write_text(json.dumps([{"instance_id": "n1", "device_count": 1, "remaining_quota_h": 10.0}]))
    jobs_file.write_text(json.dumps([
        {"job_id": "new_job", "duration_h": 2.0, "metadata": {"env": "cartpole", "model": "mlp", "steps": 720000}}
    ]))

    # Step 1: Run CLI with --record-job and --profiles
    monkeypatch.setattr(
        "sys.argv",
        [
            "accelpack.cli",
            "--instances", str(inst_file),
            "--jobs", str(jobs_file),
            "--history", str(history_file),
            "--profiles", str(profiles_file),
            "--record-job", str(record_file),
            "--output", str(out_file),
        ],
    )
    cli.main()

    # Verify profiles file was updated and persisted
    saved_profiles = json.loads(profiles_file.read_text(encoding="utf-8"))
    assert len(saved_profiles) >= 1
    # Moving average: (1 - 0.25)*100 + 0.25*200 = 125 FPS
    updated_prof = next(p for p in saved_profiles if p["env_id"] == "cartpole")
    assert updated_prof["fps"] == pytest.approx(125.0)

    # Step 2: Run CLI with --reestimate, proving updated profiles modify scheduled duration
    monkeypatch.setattr(
        "sys.argv",
        [
            "accelpack.cli",
            "--instances", str(inst_file),
            "--jobs", str(jobs_file),
            "--profiles", str(profiles_file),
            "--reestimate",
            "--output", str(out_file),
        ],
    )
    cli.main()

    plan_data = json.loads(out_file.read_text(encoding="utf-8"))
    # Re-estimated duration: 720,000 / (125 * 3600) = 1.60h (reduced from initial 2.0h!)
    assert plan_data["total_work_h"] == pytest.approx(1.60, rel=1e-2)


def test_cross_instance_critical_path_makespan():
    """Verify that cross-instance dependent segments calculate true critical path makespan (e.g. 6.1h + 6.1h = 12.2h)."""
    # Two nodes with 7.0h max session limit each (cannot co-locate two 6.0h jobs on the same node).
    instances = [
        Instance(instance_id="node-1", device_count=1, max_session_h=7.0, remaining_quota_h=20.0),
        Instance(instance_id="node-2", device_count=1, max_session_h=7.0, remaining_quota_h=20.0),
    ]

    jobs = [
        Job(job_id="seg0", duration_h=6.0),
        Job(job_id="seg1", duration_h=6.0, depends_on="seg0"),
    ]

    scheduler = AccelPackScheduler(setup_overhead_h=0.10)
    plan = scheduler.schedule(jobs, instances)

    assert len(plan.active_instances) == 2
    # Individual instance makespans are 6.10h each
    assert plan.active_instances[0].makespan_h == pytest.approx(6.10)
    assert plan.active_instances[1].makespan_h == pytest.approx(6.10)

    # Dependency-aware instance timings
    start_times, finish_times = plan.instance_schedule_timings()
    assert start_times["node-1"] == pytest.approx(0.0)
    assert finish_times["node-1"] == pytest.approx(6.10)
    assert start_times["node-2"] == pytest.approx(6.10)
    assert finish_times["node-2"] == pytest.approx(12.20)

    # True global makespan must be the sequential critical path: 12.20h (NOT 6.10h!)
    assert plan.global_makespan_h == pytest.approx(12.20)

    # Summary string must display start and finish offsets
    summary_text = plan.summary()
    assert "Start: +6.10h, Finish: 12.20h" in summary_text


def test_cross_instance_reset_deadline_enforcement():
    """Verify that downstream dependent sessions validate reset deadlines accounting for predecessor finish time."""
    now = datetime.now(timezone.utc)
    # node-1 reset is in 24 hours (plenty of time for 6.1h job)
    node1_reset = (now + timedelta(hours=24.0)).isoformat()
    # node-2 reset is in 10.0 hours. A 5.0h job (+0.10h setup = 5.1h) would fit if run at t=0,
    # but since it depends on seg0 on node-1 (finishing at t=6.1h), it finishes at 6.1h + 5.1h = 11.2h > 10.0h!
    node2_reset = (now + timedelta(hours=10.0)).isoformat()

    instances = [
        Instance(instance_id="node-1", device_count=1, max_session_h=7.0, remaining_quota_h=20.0, reset_at=node1_reset),
        Instance(instance_id="node-2", device_count=1, max_session_h=7.0, remaining_quota_h=20.0, reset_at=node2_reset),
    ]

    jobs = [
        Job(job_id="seg0", duration_h=6.0),
        Job(job_id="seg1", duration_h=5.0, depends_on="seg0"),
    ]

    scheduler = AccelPackScheduler(setup_overhead_h=0.10, enforce_reset_deadline=True)
    # node-2 must be rejected for seg1 because completion time 11.2h exceeds node-2's 10.0h reset deadline
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs, instances, allow_partial=False)


def test_dependency_graph_validation_errors():
    """Verify explicit validation errors for self-dependencies, unknown prerequisites, and cycles."""
    inst = [Instance(instance_id="node-1", device_count=1, remaining_quota_h=20.0)]
    scheduler = AccelPackScheduler()

    # 1. Self dependency
    job_self = [Job(job_id="task_a", duration_h=1.0, depends_on="task_a")]
    with pytest.raises(ValueError, match="self-dependency detected"):
        scheduler.schedule(job_self, inst)

    # 2. Unknown prerequisite
    job_unknown = [Job(job_id="task_b", duration_h=1.0, depends_on="missing_prereq")]
    with pytest.raises(ValueError, match="unknown prerequisite 'missing_prereq'"):
        scheduler.schedule(job_unknown, inst)

    # 3. Circular dependency cycle (task_1 -> task_2 -> task_1)
    job_cycle = [
        Job(job_id="task_1", duration_h=1.0, depends_on="task_2"),
        Job(job_id="task_2", duration_h=1.0, depends_on="task_1"),
    ]
    with pytest.raises(ValueError, match="Circular dependency cycle detected"):
        scheduler.schedule(job_cycle, inst)


def test_cli_history_replay_idempotence_and_precedence(tmp_path, monkeypatch):
    """Verify that loading profiles first and replaying history is idempotent and does not double-count."""
    from accelpack import cli

    profiles_file = tmp_path / "profiles.json"
    history_file = tmp_path / "history.json"
    inst_file = tmp_path / "inst.json"
    jobs_file = tmp_path / "jobs.json"
    out_file = tmp_path / "out.json"

    # Pre-existing profile snapshot with job_1 already applied (FPS = 100.0)
    initial_profiles = [
        {
            "env_id": "cartpole",
            "model": "mlp",
            "fps": 100.0,
            "hardware_class": "nvidia_t4",
            "workers_per_device": 1,
            "samples": 1,
            "confidence": 1.0,
            "applied_job_ids": ["job_1"],
        }
    ]
    profiles_file.write_text(json.dumps(initial_profiles), encoding="utf-8")

    # History file contains:
    # 1) job_1 (which was already incorporated in initial_profiles)
    # 2) job_2 (new unapplied execution with FPS = 200.0)
    history_records = [
        {
            "job_id": "job_1",
            "env_id": "cartpole",
            "model": "mlp",
            "steps": 360000,
            "predicted_duration_h": 1.0,
            "actual_duration_h": 1.0,
            "success": True,
            "instance_id": "node_prev",
            "hardware_class": "nvidia_t4",
            "concurrent_workers_on_device": 1,
            "measured_fps": 100.0,
        },
        {
            "job_id": "job_2",
            "env_id": "cartpole",
            "model": "mlp",
            "steps": 360000,
            "predicted_duration_h": 1.0,
            "actual_duration_h": 0.5,
            "success": True,
            "instance_id": "node_prev",
            "hardware_class": "nvidia_t4",
            "concurrent_workers_on_device": 1,
            "measured_fps": 200.0,
        },
    ]
    history_file.write_text(json.dumps(history_records), encoding="utf-8")

    inst_file.write_text(json.dumps([{"instance_id": "n1", "device_count": 1, "remaining_quota_h": 10.0}]))
    jobs_file.write_text(json.dumps([{"job_id": "eval_job", "duration_h": 1.0}]))

    # Run CLI: should skip job_1 and only replay job_2!
    monkeypatch.setattr(
        "sys.argv",
        [
            "accelpack.cli",
            "--instances", str(inst_file),
            "--jobs", str(jobs_file),
            "--profiles", str(profiles_file),
            "--history", str(history_file),
            "--output", str(out_file),
        ],
    )
    cli.main()

    saved_profiles = json.loads(profiles_file.read_text(encoding="utf-8"))
    prof = saved_profiles[0]
    # Check that job_1 was NOT reapplied, only job_2 was applied
    # (1 - 0.25)*100.0 + 0.25*200.0 = 125.0 FPS
    assert prof["fps"] == pytest.approx(125.0)
    assert set(prof["applied_job_ids"]) == {"job_1", "job_2"}

    # Run CLI a second time: both job_1 and job_2 are already applied!
    monkeypatch.setattr(
        "sys.argv",
        [
            "accelpack.cli",
            "--instances", str(inst_file),
            "--jobs", str(jobs_file),
            "--profiles", str(profiles_file),
            "--history", str(history_file),
            "--output", str(out_file),
        ],
    )
    cli.main()

    # Profiles must remain identical (idempotent, no double counting)
    saved_profiles_2 = json.loads(profiles_file.read_text(encoding="utf-8"))
    assert saved_profiles_2[0]["fps"] == pytest.approx(125.0)


def test_instance_level_dependency_cycle_raises_explicit_error():
    """Verify that cross-instance dependency cycles raise ValueError instead of fabricating a 0.0 makespan."""
    # Two instances
    inst1 = Instance(instance_id="node-A", device_count=1)
    inst2 = Instance(instance_id="node-B", device_count=1)

    # Acyclic job DAG:
    # ja1 (none) -> jb1 (depends on ja1)
    # jb2 (none) -> ja2 (depends on jb2)
    # Job-level graph is completely acyclic!
    ja1 = Job(job_id="ja1", duration_h=1.0)
    jb1 = Job(job_id="jb1", duration_h=1.0, depends_on="ja1")
    jb2 = Job(job_id="jb2", duration_h=1.0)
    ja2 = Job(job_id="ja2", duration_h=1.0, depends_on="ja2_prereq_jb2", parent_job_id="jb2")
    # Explicitly set dependency to induce cycle across nodes
    ja2.depends_on = "jb2"

    slot_a = DeviceSlot(instance_id="node-A", device_index=0, slot_index=0, jobs=[ja1, ja2])
    slot_b = DeviceSlot(instance_id="node-B", device_index=0, slot_index=0, jobs=[jb1, jb2])

    plan = ExecutionPlan(
        instance_plans=[
            InstancePlan(instance=inst1, slots=[slot_a]),
            InstancePlan(instance=inst2, slots=[slot_b]),
        ]
    )

    # Because node-A depends on node-B (for ja2) and node-B depends on node-A (for jb1),
    # instance_schedule_timings MUST detect the deadlock cycle and raise ValueError!
    with pytest.raises(ValueError, match="Instance dependency cycle detected"):
        plan.instance_schedule_timings()


def test_downstream_instance_unrelated_job_evaluates_start_offset():
    """Verify that an instance already delayed by a predecessor dependency evaluates reset deadlines
    for subsequent unrelated jobs using the instance's full start offset."""
    now = datetime.now(timezone.utc)
    # node-1 reset in 24h
    node1_reset = (now + timedelta(hours=24.0)).isoformat()
    # node-2 reset in 10.0h
    node2_reset = (now + timedelta(hours=10.0)).isoformat()

    instances = [
        Instance(instance_id="node-1", device_count=1, max_session_h=7.0, remaining_quota_h=20.0, reset_at=node1_reset),
        Instance(instance_id="node-2", device_count=1, max_session_h=7.0, remaining_quota_h=20.0, reset_at=node2_reset),
    ]

    # Workload:
    # 1. seg0 on node-1: 6.0h + 0.1h = 6.1h finish
    # 2. seg1 on node-2: depends on seg0, 3.0h duration (+ 0.1h setup = 3.1h makespan)
    #    On node-2, start offset is 6.1h. Finish time is 6.1 + 3.1 = 9.2h <= 10.0h (FITS!)
    # 3. job_unrelated: 2.0h duration, depends_on=None.
    #    Cannot fit on node-1 (6.0h + 2.0h = 8.0h > 7.0h session limit).
    #    If considered for node-2: hypothetical makespan on node-2 = 3.0h + 2.0h + 0.1h = 5.1h.
    #    If checked from t=0, 5.1h <= 10.0h would falsely pass.
    #    BUT with node-2 start offset = 6.1h: finish offset is 6.1h + 5.1h = 11.2h > 10.0h!
    #    Must be rejected due to reset deadline!
    jobs = [
        Job(job_id="seg0", duration_h=6.0),
        Job(job_id="seg1", duration_h=3.0, depends_on="seg0"),
        Job(job_id="job_unrelated", duration_h=2.0),
    ]

    scheduler = AccelPackScheduler(setup_overhead_h=0.10, enforce_reset_deadline=True)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs, instances, allow_partial=False)


def test_cli_and_expansion_preserves_external_depends_on(tmp_path, monkeypatch):
    """Verify that CLI JSON loading and segment expansion preserve explicit job dependencies."""
    from accelpack import cli

    # 1. Verify segment expansion inherits parent's external depends_on
    sched = AccelPackScheduler(expand_segments=True)
    prep_job = Job(job_id="prep_step", duration_h=1.0)
    segmented_job = Job(job_id="sim_task", duration_h=6.0, resumable=True, segments=[3.0, 3.0], depends_on="prep_step")

    expanded = sched._expand_segmented_jobs([prep_job, segmented_job])
    seg0 = next(j for j in expanded if j.job_id == "sim_task_seg0")
    seg1 = next(j for j in expanded if j.job_id == "sim_task_seg1")

    assert seg0.depends_on == "prep_step"
    assert seg1.depends_on == "sim_task_seg0"

    # 2. Verify CLI preserves depends_on from JSON input
    jobs_file = tmp_path / "dep_jobs.json"
    inst_file = tmp_path / "dep_inst.json"
    out_file = tmp_path / "dep_out.json"

    jobs_file.write_text(
        json.dumps([
            {"job_id": "step_a", "duration_h": 2.0},
            {"job_id": "step_b", "duration_h": 2.0, "depends_on": "step_a"},
        ]),
        encoding="utf-8",
    )
    inst_file.write_text(
        json.dumps([{"instance_id": "node_x", "device_count": 2, "remaining_quota_h": 20.0}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "accelpack.cli",
            "--instances", str(inst_file),
            "--jobs", str(jobs_file),
            "--output", str(out_file),
        ],
    )
    cli.main()

    plan_data = json.loads(out_file.read_text(encoding="utf-8"))
    scheduled_b = next(
        j for inst in plan_data["instances"]
        for slot in inst["slots"]
        for j in slot["jobs"]
        if j["job_id"] == "step_b"
    )
    assert scheduled_b["depends_on"] == "step_a"


def test_telemetry_run_id_idempotence_and_distinct_executions(tmp_path):
    """Verify that record_job is idempotent for duplicate run_ids and accurately records distinct runs of the same job."""
    history_file = tmp_path / "hist.json"
    reconciler = ExecutionReconciler(history_file=history_file)
    estimator = ProfileDurationEstimator(default_fps=100.0)

    # Run A: First execution of logical job "train_cifar"
    rec1 = reconciler.record_job(
        job_id="train_cifar",
        env_id="cifar",
        model="resnet",
        steps=360000,
        predicted_duration_h=1.0,
        actual_duration_h=1.0,  # 100 FPS
        success=True,
        instance_id="node1",
        run_id="run_001",
        estimator=estimator,
    )
    assert len(reconciler.history) == 1
    prof1 = estimator.get_profile(env_id="cifar", model="resnet")
    assert prof1.throughput == pytest.approx(100.0)
    assert prof1.samples == 1
    assert "run_001" in prof1.applied_run_ids

    # Run A duplicate submission (same run_id): must be idempotent!
    rec1_dup = reconciler.record_job(
        job_id="train_cifar",
        env_id="cifar",
        model="resnet",
        steps=360000,
        predicted_duration_h=1.0,
        actual_duration_h=1.0,
        success=True,
        instance_id="node1",
        run_id="run_001",
        estimator=estimator,
    )
    assert rec1_dup is rec1
    assert len(reconciler.history) == 1
    assert prof1.samples == 1  # Samples did NOT increment!
    assert prof1.throughput == pytest.approx(100.0)  # No duplicate moving-average shift!

    # Run B: Second distinct execution of the same logical job "train_cifar" (run_id="run_002", ran faster at 200 FPS)
    reconciler.record_job(
        job_id="train_cifar",
        env_id="cifar",
        model="resnet",
        steps=360000,
        predicted_duration_h=1.0,
        actual_duration_h=0.5,  # 200 FPS
        success=True,
        instance_id="node1",
        run_id="run_002",
        estimator=estimator,
    )
    assert len(reconciler.history) == 2
    prof2 = estimator.get_profile(env_id="cifar", model="resnet")
    assert prof2.samples == 2
    assert "run_002" in prof2.applied_run_ids
    # Calibrated: (1 - 0.25)*100.0 + 0.25*200.0 = 125.0 FPS
    assert prof2.throughput == pytest.approx(125.0)


def test_duplicate_job_ids_and_segment_id_collisions_rejected():
    """Verify that duplicate user job IDs and collisions with generated segment IDs fail explicitly."""
    inst = [Instance(instance_id="node-1", device_count=1, remaining_quota_h=20.0)]
    scheduler = AccelPackScheduler(expand_segments=True)

    # 1. Duplicate user job IDs
    dup_jobs = [
        Job(job_id="train_job_1", duration_h=1.0),
        Job(job_id="train_job_1", duration_h=2.0),
    ]
    with pytest.raises(ValueError, match="Duplicate job ID detected in workload"):
        scheduler.schedule(dup_jobs, inst)

    # 2. Collision with generated segment ID
    colliding_jobs = [
        Job(job_id="sim_work", duration_h=4.0, segments=[2.0, 2.0]),
        Job(job_id="sim_work_seg0", duration_h=1.0),
    ]
    with pytest.raises(ValueError, match="collides with an existing job ID"):
        scheduler.schedule(colliding_jobs, inst)




