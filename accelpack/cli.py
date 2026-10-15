"""CLI interface for AccelPack scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

from accelpack.estimator import ProfileDurationEstimator
from accelpack.models import Instance, Job, ObjectiveMode, PackingStrategy
from accelpack.preflight import PreflightValidator
from accelpack.reconciliation import (
    ExecutionReconciler,
    HistoryLoadException,
    HistoryPersistenceException,
    ReplayResult,
    atomic_save_json,
    file_lock,
    reconcile_and_persist_profiles,
)
from accelpack.scheduler import AccelPackScheduler, QuotaExceededException


def load_or_assign_execution_record(record_path: Path | str) -> Dict[str, Any]:
    """Loads a job record JSON, ensuring a stable, canonical run_id exists under file lock.
    If run_id is missing, generates one, persists it atomically, and returns the canonical dict.
    Fails closed if the record cannot be read or if a generated run_id cannot be persisted.
    """
    path = Path(record_path)
    with file_lock(path):
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
        except Exception as e:
            raise RuntimeError(f"Failed to read job record file {path}: {e}") from e

        if data.get("run_id"):
            return data

        jid = data.get("job_id", "job")
        assigned_run_id = f"run_{jid}_{uuid.uuid4().hex[:12]}"
        data["run_id"] = assigned_run_id
        try:
            atomic_save_json(path, data)
        except Exception as e:
            raise RuntimeError(
                f"Cannot persist generated run_id to {path}: {e}. "
                f"Aborting execution recording to prevent breaking idempotence."
            ) from e

        return data


def main():
    parser = argparse.ArgumentParser(description="AccelPack: Multi-Accelerator Quota-Preserving Scheduler")
    parser.add_argument("--instances", type=str, required=True, help="Path to JSON file with available instances/nodes")
    parser.add_argument("--jobs", type=str, required=True, help="Path to JSON file with jobs to schedule")
    parser.add_argument("--strategy", choices=["balance", "concentrate"], default="balance", help="Packing strategy")
    parser.add_argument(
        "--objective",
        choices=["complete_all", "maximize_work", "maximize_efficiency"],
        default="complete_all",
        help="Objective mode",
    )
    parser.add_argument("--slots-per-device", type=int, default=None, help="Concurrent worker streams per device (e.g. 2)")
    parser.add_argument("--setup-overhead", type=float, default=0.10, help="Cold-start setup overhead in hours")
    parser.add_argument("--quota-buffer", type=float, default=2.0, help="Minimum safety quota buffer in hours")
    parser.add_argument("--max-session", type=float, default=None, help="Global max session override in hours")
    parser.add_argument("--cpu-penalty-weight", type=float, default=0.5, help="Weight for CPU oversubscription soft penalty")
    parser.add_argument(
        "--max-cpu-oversubscription",
        type=float,
        default=None,
        help="Emergency hard ceiling ratio for CPU oversubscription (e.g. 2.0)",
    )
    parser.add_argument("--enforce-reset-deadline", action="store_true", help="Reject assignments exceeding reset_at deadline")
    parser.add_argument(
        "--reject-conflicting-quotas",
        action="store_true",
        help="Reject scheduling if instances sharing an account declare conflicting quotas",
    )
    parser.add_argument("--no-expand-segments", action="store_true", help="Disable automatic expansion of segmented jobs")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial scheduling without error")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight validation")
    parser.add_argument("--ignore-blockers", action="store_true", help="Proceed with scheduling even if preflight has blockers")
    parser.add_argument("--output", type=str, default=None, help="Path to export schedule JSON")

    # Telemetry and profiles integration
    parser.add_argument("--history", type=str, default=None, help="Path to execution history JSON for online calibration")
    parser.add_argument("--profiles", type=str, default=None, help="Path to throughput profiles database JSON")
    parser.add_argument("--reestimate", action="store_true", help="Recalculate job durations using loaded profiles database")
    parser.add_argument("--record-job", type=str, default=None, help="Path to completed job run summary JSON to record into history")
    parser.add_argument("--accuracy-report", action="store_true", help="Print prediction accuracy and error report from history")
    args = parser.parse_args()

    # 1. Telemetry & Profiling Initialization
    reconciler = ExecutionReconciler(history_file=args.history) if args.history else None
    estimator = ProfileDurationEstimator(default_fps=100.0)

    # Step A: Load profiles first (if provided) as baseline/prior state
    if args.profiles and Path(args.profiles).exists():
        try:
            prof_data = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
            for p in prof_data:
                estimator.register_profile(
                    env_id=p["env_id"],
                    model=p["model"],
                    fps=p["fps"],
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
                )
        except Exception as e:
            sys.stderr.write(f"[WARN AccelPack] Could not load profiles database {args.profiles}: {e}\n")

    # Step B: Replay unapplied historical execution records into estimator (if history provided)
    if reconciler and reconciler.history:
        try:
            if args.profiles:
                replay_res = reconcile_and_persist_profiles(args.profiles, reconciler, estimator, save_fn=atomic_save_json)
            else:
                replay_res = reconciler.replay_into_estimator(estimator)
            if replay_res.replayed_count > 0:
                print(f"[OK] Replayed {replay_res.replayed_count} unapplied historical execution records to calibrate throughput profiles.")
            if replay_res.migrations_count > 0:
                print(f"[OK] Migrated {replay_res.migrations_count} legacy profile marker(s) to historical execution run IDs.")
        except HistoryLoadException as e:
            sys.stderr.write(f"[ERROR AccelPack] Failed to load history file: {e}\n")
            return
        except Exception as e:
            sys.stderr.write(f"[ERROR AccelPack] Failed to replay history into profiles: {e}\n")

    # Record completed job if requested
    if args.record_job and Path(args.record_job).exists() and reconciler:
        run_path = Path(args.record_job)
        try:
            run_data = load_or_assign_execution_record(run_path)
        except RuntimeError as e:
            sys.stderr.write(f"[ERROR AccelPack] {e}\n")
            return
        except Exception as e:
            sys.stderr.write(f"[ERROR AccelPack] Failed to read job record file {run_path}: {e}\n")
            return

        # Step 1: Record into history
        history_persisted = False
        try:
            reconciler.record_job(
                job_id=run_data["job_id"],
                env_id=run_data.get("env_id", ""),
                model=run_data.get("model", ""),
                steps=run_data.get("steps", 0),
                predicted_duration_h=run_data.get("predicted_duration_h", 0.0),
                actual_duration_h=run_data.get("actual_duration_h", 0.0),
                success=run_data.get("success", True),
                instance_id=run_data.get("instance_id", ""),
                run_id=run_data.get("run_id"),
                device_index=run_data.get("device_index", 0),
                concurrent_workers_on_device=run_data.get("concurrent_workers") or run_data.get("concurrent_workers_on_device", 1),
                hardware_class=run_data.get("hardware_class", "nvidia_t4"),
                peak_vram_gb=run_data.get("peak_vram_gb"),
                peak_ram_gb=run_data.get("peak_ram_gb"),
                cpu_busy_pct=run_data.get("cpu_busy_pct"),
                estimator=estimator,
            )
            history_persisted = True
            print(f"[OK] Recorded completed job {run_data['job_id']} (run_id: {run_data['run_id']}) into history.")
        except HistoryLoadException as e:
            sys.stderr.write(f"[ERROR AccelPack] Failed to load history file: {e}\n")
            return
        except HistoryPersistenceException as e:
            sys.stderr.write(f"[ERROR AccelPack] Failed to persist job execution history: {e}\n")
            return
        except Exception as e:
            sys.stderr.write(f"[ERROR AccelPack] Failed to record job summary: {e}\n")
            return

        # Step 2: Persist updated profiles atomically and under profile lock if requested
        if history_persisted and args.profiles:
            try:
                reconcile_and_persist_profiles(args.profiles, reconciler, estimator, save_fn=atomic_save_json)
                print(f"[OK] Persisted {len(estimator.to_dict_list())} updated profiles to {args.profiles}")
            except Exception as e:
                sys.stderr.write(
                    f"[ERROR AccelPack] History record was saved, but failed to persist updated profiles to {args.profiles}: {e}. "
                    f"Subsequent replay will recover profile calibrations from history.\n"
                )

    # Print accuracy report if requested
    if args.accuracy_report:
        if reconciler:
            acc_rep = reconciler.get_prediction_error_report()
            print("\n" + "=" * 70)
            print("ACCELPACK PREDICTION ACCURACY REPORT")
            print(f"Total Records: {acc_rep['total_records']}")
            print(f"Overall MAE: {acc_rep['overall']['mae_hours']:.4f}h | "
                  f"RMSE: {acc_rep['overall']['rmse_hours']:.4f}h | "
                  f"MAPE: {acc_rep['overall']['mape_pct']:.1f}% | "
                  f"Success Rate: {acc_rep['overall']['success_rate'] * 100:.1f}%")
            if acc_rep["by_stratum"]:
                print("-" * 70)
                print("Breakdown by Task / Hardware / Concurrency:")
                for k, v in acc_rep["by_stratum"].items():
                    print(f"  * {k}: count={v['count']}, MAE={v['mae_hours']:.4f}h, success={v['success_rate']*100:.0f}%")
            print("=" * 70 + "\n")
        else:
            print("[WARN] --accuracy-report requires --history <file>.")

    # 2. Parse instances and jobs
    instances_data = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    instances = [
        Instance(
            instance_id=i["instance_id"],
            account_id=i.get("account_id"),
            device_count=i.get("device_count", 2),
            remaining_quota_h=i.get("remaining_quota_h", 30.0),
            max_session_h=i.get("max_session_h", 11.0),
            vram_per_device_gb=i.get("vram_per_device_gb", 16.0),
            host_ram_gb=i.get("host_ram_gb", 30.0),
            host_cpu_cores=i.get("host_cpu_cores", 4.0),
            slots_per_device=i.get("slots_per_device", 1),
            debit_rate=i.get("debit_rate", 1.0),
            reset_at=i.get("reset_at"),
            must_finish_before_reset=i.get("must_finish_before_reset", False),
            hardware_class=i.get("hardware_class", "nvidia_t4"),
            concurrency_slowdown_factors=i.get("concurrency_slowdown_factors", {1: 1.0, 2: 1.30, 3: 1.65, 4: 2.0}),
            max_cpu_oversubscription_ratio=i.get("max_cpu_oversubscription_ratio"),
            metadata=i.get("metadata", {}),
        )
        for i in instances_data
    ]

    jobs_data = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    jobs = []
    for j in jobs_data:
        duration_h = j["duration_h"]
        # Re-estimate duration from loaded profiles if requested and steps provided
        if args.reestimate and "metadata" in j and "steps" in j["metadata"]:
            steps = j["metadata"]["steps"]
            env_id = j["metadata"].get("env", "")
            model = j["metadata"].get("model", "")
            hw = j.get("hardware_class", "nvidia_t4")
            duration_h = estimator.estimate_duration_h(steps, env_id=env_id, model=model, hardware_class=hw)

        jobs.append(
            Job(
                job_id=j["job_id"],
                duration_h=duration_h,
                vram_gb=j.get("vram_gb", 0.5),
                ram_gb=j.get("ram_gb", 1.5),
                cpu_cores=j.get("cpu_cores", 1.0),
                priority=j.get("priority", 1.0),
                resumable=j.get("resumable", False),
                segments=j.get("segments"),
                hardware_class=j.get("hardware_class"),
                depends_on=j.get("depends_on"),
                parent_job_id=j.get("parent_job_id"),
                segment_index=j.get("segment_index"),
                metadata=j.get("metadata", {}),
            )
        )

    # 3. Preflight Validation stage
    if not args.skip_preflight:
        validator = PreflightValidator(
            min_quota_buffer_h=args.quota_buffer,
            enforce_reset_deadline=args.enforce_reset_deadline,
            strict_account_quotas=args.reject_conflicting_quotas,
            allow_segmented_jobs=not args.no_expand_segments,
        )
        report = validator.validate(jobs, instances)
        print(report.format_text())
        if not report.passed and not args.ignore_blockers:
            print(f"[ERROR] Preflight validation failed with {len(report.blockers)} blocker(s). Use --ignore-blockers to override.")
            sys.exit(1)

    # 4. Scheduling stage
    scheduler = AccelPackScheduler(
        strategy=PackingStrategy(args.strategy),
        objective=ObjectiveMode(args.objective),
        setup_overhead_h=args.setup_overhead,
        quota_buffer_h=args.quota_buffer,
        max_session_h=args.max_session,
        slots_per_device=args.slots_per_device,
        cpu_penalty_weight=args.cpu_penalty_weight,
        max_cpu_oversubscription_ratio=args.max_cpu_oversubscription,
        enforce_reset_deadline=args.enforce_reset_deadline,
        reject_conflicting_account_quotas=args.reject_conflicting_quotas,
        expand_segments=not args.no_expand_segments,
    )

    try:
        plan = scheduler.schedule(jobs, instances, allow_partial=args.allow_partial)
    except QuotaExceededException as e:
        print(f"[ERROR] Scheduling failed: {e}")
        sys.exit(2)

    print(plan.summary())

    if args.output:
        acct_rem_map = plan.account_remaining_quota()
        export_data = {
            "global_makespan_h": plan.global_makespan_h,
            "total_work_h": plan.total_work_h,
            "total_billed_resource_h": plan.total_billed_resource_h,
            "quota_productivity_ratio": plan.quota_productivity_ratio,
            "account_billed_quota": plan.account_billed_quota(),
            "account_remaining_quota": acct_rem_map,
            "instances": [
                {
                    "instance_id": ip.instance.instance_id,
                    "account_id": ip.instance.account_id,
                    "makespan_h": ip.makespan_h,
                    "billed_resource_h": ip.billed_resource_h,
                    "quota_remaining_after_h": acct_rem_map.get(ip.instance.account_id or ip.instance.instance_id, ip.quota_remaining_after_h),
                    "account_quota_remaining_after_h": acct_rem_map.get(ip.instance.account_id or ip.instance.instance_id, ip.quota_remaining_after_h),
                    "instance_alone_quota_remaining_h": ip.quota_remaining_after_h,
                    "peak_concurrent_workers": ip.peak_concurrent_workers,
                    "cpu_oversubscription_ratio": ip.cpu_oversubscription_ratio,
                    "is_cpu_oversubscribed": ip.is_cpu_oversubscribed,
                    "device_slowdowns": ip.device_slowdowns,
                    "slots": [
                        {
                            "device_index": s.device_index,
                            "slot_index": s.slot_index,
                            "load_h": s.load_h,
                            "jobs": [
                                {
                                    "job_id": j.job_id,
                                    "duration_h": j.duration_h,
                                    "vram_gb": j.vram_gb,
                                    "depends_on": j.depends_on,
                                    "parent_job_id": j.parent_job_id,
                                    "segment_index": j.segment_index,
                                    "metadata": j.metadata,
                                }
                                for j in s.jobs
                            ],
                        }
                        for s in ip.slots
                    ],
                }
                for ip in plan.active_instances
            ],
            "unassigned_jobs": [
                {"job_id": j.job_id, "reason": j.metadata.get("unassigned_reason", "Unknown")}
                for j in plan.unassigned_jobs
            ],
        }
        Path(args.output).write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        print(f"\n[OK] Plan exported to {args.output}")


if __name__ == "__main__":
    main()
