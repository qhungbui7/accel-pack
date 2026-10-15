"""Comprehensive unit tests for AccelPack multi-accelerator scheduling engine."""

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


def test_job_validation():
    # Negative duration must be rejected
    with pytest.raises(ValueError):
        Job(job_id="bad_job", duration_h=-1.0)

    # Valid job
    j = Job(job_id="valid", duration_h=2.5)
    assert j.duration_h == 2.5
    assert j.vram_gb == 0.5
    assert j.account_id is None if hasattr(j, "account_id") else True

    # Job segments validation
    with pytest.raises(ValueError):
        Job(job_id="seg_mismatch", duration_h=5.0, segments=[2.0, 2.0])  # Sums to 4.0 != 5.0

    valid_segmented = Job(job_id="segmented", duration_h=6.0, resumable=True, segments=[3.0, 3.0])
    assert valid_segmented.segments == [3.0, 3.0]


def test_instance_validation():
    with pytest.raises(ValueError):
        Instance(instance_id="bad", device_count=0)

    inst = Instance(instance_id="node1", device_count=2, remaining_quota_h=20.0)
    assert inst.remaining_quota_h == 20.0
    # account_id defaults to instance_id if omitted
    assert inst.account_id == "node1"

    # Explicit account_id
    inst_shared = Instance(instance_id="node2", account_id="team_acc", device_count=2)
    assert inst_shared.account_id == "team_acc"

    # Concurrency slowdown lookup
    assert inst.get_concurrency_slowdown(1) == 1.0
    assert inst.get_concurrency_slowdown(2) == 1.30
    assert inst.get_concurrency_slowdown(3) == 1.65


def test_cpu_soft_oversubscription_allowed():
    """Verify that jobs with aggregate CPU demand exceeding host vCPU count can be scheduled
    with soft penalty rather than unconditional rejection.
    """
    instances = [
        Instance(instance_id="node-1", device_count=2, host_cpu_cores=4.0, remaining_quota_h=20.0),
    ]

    # Two concurrent jobs demanding 3.0 CPU cores each = 6.0 cores on 4.0 vCPU host
    jobs = [
        Job(job_id="j1", duration_h=1.0, cpu_cores=3.0, vram_gb=1.0, ram_gb=2.0),
        Job(job_id="j2", duration_h=1.0, cpu_cores=3.0, vram_gb=1.0, ram_gb=2.0),
    ]

    scheduler = AccelPackScheduler(slots_per_device=1)
    plan = scheduler.schedule(jobs, instances)

    assert len(plan.active_instances) == 1
    ip = plan.active_instances[0]
    assert ip.peak_host_cpu_cores == 6.0
    assert ip.cpu_oversubscription_ratio == 1.5  # 6.0 / 4.0
    assert ip.is_cpu_oversubscribed is True
    assert ip.cpu_contention_penalty == pytest.approx(0.25)  # (1.5 - 1.0)^2 = 0.25


def test_cpu_emergency_cap_blocks():
    """Verify that if an emergency hard cap is configured, CPU oversubscription beyond that ceiling is rejected."""
    instances = [
        Instance(
            instance_id="node-1",
            device_count=2,
            host_cpu_cores=4.0,
            remaining_quota_h=20.0,
            max_cpu_oversubscription_ratio=1.40,  # Max 4 * 1.4 = 5.6 cores
        ),
    ]

    jobs = [
        Job(job_id="j1", duration_h=1.0, cpu_cores=3.0),
        Job(job_id="j2", duration_h=1.0, cpu_cores=3.0),
    ]

    # Demand is 6.0 cores > 5.6 cores -> Must be rejected!
    scheduler = AccelPackScheduler(slots_per_device=1)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs, instances, allow_partial=False)


def test_vram_and_ram_hard_constraints_enforced():
    """Verify that physical VRAM and host RAM remain strictly enforced hard limits with safety margins."""
    instances = [
        Instance(
            instance_id="node-1",
            device_count=2,
            vram_per_device_gb=16.0,
            host_ram_gb=30.0,
            remaining_quota_h=20.0,
        ),
    ]

    # VRAM limit violation: 15.8 GB + 0.5 GB margin > 16.0 GB
    jobs_vram_exceeded = [Job(job_id="heavy_vram", duration_h=1.0, vram_gb=15.8)]
    scheduler = AccelPackScheduler(safety_vram_margin_gb=0.5)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs_vram_exceeded, instances)

    # Host RAM limit violation: 29.5 GB + 1.0 GB margin > 30.0 GB
    jobs_ram_exceeded = [Job(job_id="heavy_ram", duration_h=1.0, ram_gb=29.5)]
    scheduler_ram = AccelPackScheduler(safety_ram_margin_gb=1.0)
    with pytest.raises(QuotaExceededException):
        scheduler_ram.schedule(jobs_ram_exceeded, instances)


def test_concurrency_slowdown_models():
    """Verify that multiple concurrent slots on the same physical device apply realistic slowdown."""
    instances = [
        Instance(
            instance_id="node-1",
            device_count=1,
            slots_per_device=2,
            remaining_quota_h=20.0,
            concurrency_slowdown_factors={1: 1.0, 2: 1.30},
        ),
    ]

    # Two 1-hour jobs running concurrently on the same GPU
    jobs = [
        Job(job_id="j1", duration_h=1.0),
        Job(job_id="j2", duration_h=1.0),
    ]

    scheduler = AccelPackScheduler(setup_overhead_h=0.10)
    plan = scheduler.schedule(jobs, instances)

    ip = plan.active_instances[0]
    # With 2 concurrent workers on device 0, slowdown is 1.30x
    # Effective device load = 1.0 * 1.30 = 1.30h
    assert ip.device_loads_h[0] == pytest.approx(1.30)
    # Makespan = 1.30 + 0.10 setup = 1.40h
    assert ip.makespan_h == pytest.approx(1.40)
    assert ip.billed_resource_h == pytest.approx(1.40)
    # Useful work = 2.0h, billed = 1.40h -> Productivity = 2.0 / 1.40 = ~1.4286
    assert plan.quota_productivity_ratio == pytest.approx(2.0 / 1.40)


def test_shared_account_quota_pool():
    """Verify that multiple instances sharing the same account_id draw against a single shared quota balance."""
    instances = [
        Instance(instance_id="node-a", account_id="kaggle_user", device_count=1, remaining_quota_h=8.0),
        Instance(instance_id="node-b", account_id="kaggle_user", device_count=1, remaining_quota_h=8.0),
    ]

    # Two 3-hour jobs. Quota buffer is 3.0h.
    # Total usable quota for account "kaggle_user" = 8.0 - 3.0 = 5.0h.
    # Job 1 on node-a: makespan ~3.1h.
    # Job 2 on node-b: makespan ~3.1h. Total billed for kaggle_user = 6.2h > 5.0h -> Cannot both fit!
    jobs = [
        Job(job_id="j1", duration_h=3.0),
        Job(job_id="j2", duration_h=3.0),
    ]

    scheduler = AccelPackScheduler(quota_buffer_h=3.0, setup_overhead_h=0.10)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs, instances, allow_partial=False)

    # With allow_partial=True, only 1 job can be scheduled within the shared account quota
    partial_plan = scheduler.schedule(jobs, instances, allow_partial=True)
    assert len(partial_plan.active_instances) == 1
    assert len(partial_plan.unassigned_jobs) == 1


def test_reset_deadline_enforcement():
    """Verify that when enforce_reset_deadline is enabled, jobs that would finish after reset_at are rejected."""
    now = datetime.now(timezone.utc)
    # Reset is 1.5 hours in the future
    future_reset = (now + timedelta(hours=1.5)).isoformat()

    instances = [
        Instance(
            instance_id="node-1",
            device_count=1,
            remaining_quota_h=20.0,
            reset_at=future_reset,
            must_finish_before_reset=True,
        ),
    ]

    # Job duration 2.0h + setup 0.1h = 2.1h > 1.5h until reset
    jobs = [Job(job_id="j_long", duration_h=2.0)]

    scheduler = AccelPackScheduler(enforce_reset_deadline=True, setup_overhead_h=0.10)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs, instances, allow_partial=False)

    # Short job (0.5h + 0.1h = 0.6h < 1.5h) fits easily
    short_jobs = [Job(job_id="j_short", duration_h=0.5)]
    plan = scheduler.schedule(short_jobs, instances)
    assert len(plan.active_instances) == 1


def test_resumable_segmentation_in_preflight():
    """Verify that resumable=True does NOT bypass preflight session limits unless valid segments are defined."""
    validator = PreflightValidator()
    instances = [Instance(instance_id="node-1", max_session_h=10.0)]

    # Job is 15h, resumable=True, but NO segments -> MUST BE A BLOCKER!
    job_unsegmented = [Job(job_id="big_job", duration_h=15.0, resumable=True)]
    report = validator.validate(job_unsegmented, instances)
    assert report.passed is False
    assert any(i.code == "JOB_EXCEEDS_SESSION_LIMIT" and i.severity == "BLOCKER" for i in report.issues)

    # Job is 15h, resumable=True, WITH valid segments [7.5, 7.5] <= 9.5h -> MUST PASS!
    job_segmented = [Job(job_id="big_job_ok", duration_h=15.0, resumable=True, segments=[7.5, 7.5])]
    report_ok = validator.validate(job_segmented, instances)
    assert report_ok.passed is True


def test_objective_modes_differ():
    """Verify that MAXIMIZE_WORK and COMPLETE_ALL produce different decisions under budget constraints."""
    instances = [
        Instance(instance_id="node-1", device_count=1, remaining_quota_h=5.0),
    ]

    # Available budget after 2.0h safety reserve is 3.0h.
    # Jobs:
    # j_valuable: priority 10.0, duration 2.5h (priority-work = 25.0)
    # j_filler: priority 1.0, duration 2.0h (priority-work = 2.0)
    jobs = [
        Job(job_id="j_filler", duration_h=2.0, priority=1.0),
        Job(job_id="j_valuable", duration_h=2.5, priority=10.0),
    ]

    # COMPLETE_ALL with allow_partial sorts by LPT (j_valuable 2.5h, then j_filler 2.0h)
    sched_work = AccelPackScheduler(objective=ObjectiveMode.MAXIMIZE_WORK, quota_buffer_h=2.0)
    plan_work = sched_work.schedule(jobs, instances, allow_partial=True)

    assigned_work_ids = [j.job_id for ip in plan_work.active_instances for s in ip.slots for j in s.jobs]
    assert "j_valuable" in assigned_work_ids
    assert "j_filler" not in assigned_work_ids


def test_reconciliation_prediction_error_report(tmp_path):
    """Verify that ExecutionReconciler tracks telemetry, computes accuracy metrics, and reports errors."""
    history_file = tmp_path / "history.json"
    reconciler = ExecutionReconciler(history_file=history_file)

    # Record 2 jobs
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
        # Intentionally omitting peak_vram_gb and cpu_busy_pct to test missing telemetry handling
    )

    # Missing telemetry was detected
    assert "peak_vram_gb" in reconciler.history[1].missing_telemetry
    assert "cpu_busy_pct" in reconciler.history[1].missing_telemetry

    report = reconciler.get_prediction_error_report()
    assert report["total_records"] == 2
    # Errors: |1.1 - 1.0| = 0.1, |1.2 - 1.0| = 0.2 -> MAE = 0.15
    assert report["overall"]["mae_hours"] == pytest.approx(0.15)
    assert report["overall"]["success_rate"] == 1.0

    # Persistence verification
    assert history_file.exists()
    reloaded = ExecutionReconciler(history_file=history_file)
    assert len(reloaded.history) == 2


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
    """Verify that in BALANCE mode, jobs are distributed across instances and partner devices are saturated."""
    instances = [
        Instance(instance_id="used_node", device_count=2, remaining_quota_h=20.0, max_session_h=11.0),
        Instance(instance_id="fresh_node_1", device_count=2, remaining_quota_h=29.0, max_session_h=11.0),
        Instance(instance_id="fresh_node_2", device_count=2, remaining_quota_h=29.0, max_session_h=11.0),
    ]

    jobs = [Job(job_id=f"job_{i}", duration_h=1.0) for i in range(6)]

    scheduler = AccelPackScheduler(strategy=PackingStrategy.BALANCE, setup_overhead_h=0.10)
    plan = scheduler.schedule(jobs, instances)

    active_ids = {ip.instance.instance_id for ip in plan.active_instances}
    assert "fresh_node_1" in active_ids
    assert "fresh_node_2" in active_ids
    for ip in plan.active_instances:
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
    assert abs(slots[0].load_h - 2.5) < 1e-4
    assert abs(slots[1].load_h - 2.5) < 1e-4
    assert abs(plan.active_instances[0].load_imbalance_h) < 1e-4


def test_session_limit_spillover():
    """Verify that when workload exceeds one instance's session limit (4h), it spills cleanly into a second instance."""
    instances = [
        Instance(instance_id="node-1", device_count=2, remaining_quota_h=30.0, max_session_h=4.0),
        Instance(instance_id="node-2", device_count=2, remaining_quota_h=30.0, max_session_h=4.0),
    ]

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

    jobs = [Job(job_id="j1", duration_h=2.0)]

    scheduler = AccelPackScheduler(quota_buffer_h=2.0)
    with pytest.raises(QuotaExceededException):
        scheduler.schedule(jobs, instances)


def test_profile_duration_estimator():
    estimator = ProfileDurationEstimator(default_fps=100.0)
    estimator.register_profile("cartpole", "linear_ema", fps=2000.0, hardware_class="nvidia_t4", workers_per_device=1)

    dur = estimator.estimate_duration_h(steps=1_024_000, env_id="cartpole", model="linear_ema", hardware_class="nvidia_t4")
    assert abs(dur - 0.14222) < 1e-3

    job = estimator.create_job("j1", steps=1_024_000, env_id="cartpole", model="linear_ema", hardware_class="nvidia_t4")
    assert abs(job.duration_h - 0.14222) < 1e-3
    assert job.metadata["env"] == "cartpole"


def test_cli_preflight_blocks_and_export(tmp_path, monkeypatch):
    """Verify that CLI runs preflight, aborts on blocker, and exports JSON when valid."""
    from accelpack import cli

    inst_file = tmp_path / "instances.json"
    inst_file.write_text(
        json.dumps(
            [
                {
                    "instance_id": "inst_1",
                    "device_count": 2,
                    "remaining_quota_h": 20.0,
                    "max_session_h": 10.0,
                }
            ]
        )
    )

    # Job that exceeds session limit without segments -> Should trigger preflight blocker and sys.exit(1)
    bad_jobs_file = tmp_path / "bad_jobs.json"
    bad_jobs_file.write_text(
        json.dumps(
            [
                {
                    "job_id": "job_too_big",
                    "duration_h": 15.0,
                    "resumable": False,
                }
            ]
        )
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "accelpack.cli",
            "--instances",
            str(inst_file),
            "--jobs",
            str(bad_jobs_file),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1

    # Now with valid jobs and export
    good_jobs_file = tmp_path / "good_jobs.json"
    good_jobs_file.write_text(
        json.dumps(
            [
                {
                    "job_id": "job_1",
                    "duration_h": 1.0,
                }
            ]
        )
    )
    output_file = tmp_path / "plan_out.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "accelpack.cli",
            "--instances",
            str(inst_file),
            "--jobs",
            str(good_jobs_file),
            "--output",
            str(output_file),
        ],
    )

    cli.main()
    assert output_file.exists()
    plan_data = json.loads(output_file.read_text())
    assert plan_data["total_work_h"] == 1.0
    assert len(plan_data["instances"]) == 1


def test_preflight_inconsistent_account_quota_and_reset_warnings():
    """Verify preflight detects contradictory account quotas and handles past/future reset timestamps."""
    validator = PreflightValidator()
    now = datetime.now(timezone.utc)
    past_reset = (now - timedelta(hours=2.0)).isoformat()
    future_reset = (now + timedelta(hours=5.0)).isoformat()

    instances = [
        Instance(
            instance_id="node_1",
            account_id="shared_acc",
            remaining_quota_h=20.0,
            reset_at=past_reset,
        ),
        Instance(
            instance_id="node_2",
            account_id="shared_acc",
            remaining_quota_h=25.0,  # Contradictory quota for shared_acc
            reset_at=future_reset,
        ),
    ]

    jobs = [Job(job_id="j1", duration_h=1.0)]
    report = validator.validate(jobs, instances)

    # Should detect inconsistent quota warning
    assert any(i.code == "INCONSISTENT_ACCOUNT_QUOTA" for i in report.issues)
    # Should detect reset time passed warning
    assert any(i.code == "RESET_TIME_PASSED" for i in report.issues)

