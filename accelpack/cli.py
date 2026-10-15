"""CLI interface for AccelPack scheduler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from accelpack.models import Instance, Job, ObjectiveMode, PackingStrategy
from accelpack.scheduler import AccelPackScheduler


def main():
    parser = argparse.ArgumentParser(description="AccelPack: Multi-Accelerator Quota-Preserving Scheduler")
    parser.add_argument("--instances", type=str, required=True, help="Path to JSON file with available instances/nodes")
    parser.add_argument("--jobs", type=str, required=True, help="Path to JSON file with jobs to schedule")
    parser.add_argument("--strategy", choices=["balance", "concentrate"], default="balance", help="Packing strategy")
    parser.add_argument("--objective", choices=["complete_all", "maximize_work", "maximize_efficiency"], default="complete_all", help="Objective mode")
    parser.add_argument("--slots-per-device", type=int, default=None, help="Concurrent worker streams per device (e.g. 2)")
    parser.add_argument("--setup-overhead", type=float, default=0.10, help="Cold-start setup overhead in hours")
    parser.add_argument("--quota-buffer", type=float, default=2.0, help="Minimum safety quota buffer in hours")
    parser.add_argument("--max-session", type=float, default=None, help="Global max session override in hours")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial scheduling without error")
    parser.add_argument("--output", type=str, default=None, help="Path to export schedule JSON")
    args = parser.parse_args()

    instances_data = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    instances = [
        Instance(
            instance_id=i["instance_id"],
            device_count=i.get("device_count", 2),
            remaining_quota_h=i.get("remaining_quota_h", 30.0),
            max_session_h=i.get("max_session_h", 11.0),
            vram_per_device_gb=i.get("vram_per_device_gb", 16.0),
            host_ram_gb=i.get("host_ram_gb", 30.0),
            host_cpu_cores=i.get("host_cpu_cores", 4.0),
            slots_per_device=i.get("slots_per_device", 1),
            debit_rate=i.get("debit_rate", 1.0),
            reset_at=i.get("reset_at"),
            metadata=i.get("metadata", {}),
        )
        for i in instances_data
    ]

    jobs_data = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    jobs = [
        Job(
            job_id=j["job_id"],
            duration_h=j["duration_h"],
            vram_gb=j.get("vram_gb", 0.5),
            ram_gb=j.get("ram_gb", 1.5),
            cpu_cores=j.get("cpu_cores", 1.0),
            priority=j.get("priority", 1.0),
            resumable=j.get("resumable", False),
            metadata=j.get("metadata", {}),
        )
        for j in jobs_data
    ]

    scheduler = AccelPackScheduler(
        strategy=PackingStrategy(args.strategy),
        objective=ObjectiveMode(args.objective),
        setup_overhead_h=args.setup_overhead,
        quota_buffer_h=args.quota_buffer,
        max_session_h=args.max_session,
        slots_per_device=args.slots_per_device,
    )

    plan = scheduler.schedule(jobs, instances, allow_partial=args.allow_partial)
    print(plan.summary())

    if args.output:
        # Export executable execution plan JSON
        export_data = {
            "global_makespan_h": plan.global_makespan_h,
            "total_work_h": plan.total_work_h,
            "total_billed_resource_h": plan.total_billed_resource_h,
            "efficiency_ratio": plan.efficiency_ratio,
            "instances": [
                {
                    "instance_id": ip.instance.instance_id,
                    "makespan_h": ip.makespan_h,
                    "billed_resource_h": ip.billed_resource_h,
                    "quota_remaining_after_h": ip.quota_remaining_after_h,
                    "peak_concurrent_workers": ip.peak_concurrent_workers,
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
            "unassigned_jobs": [j.job_id for j in plan.unassigned_jobs],
        }
        Path(args.output).write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        print(f"\n[OK] Plan exported to {args.output}")


if __name__ == "__main__":
    main()
