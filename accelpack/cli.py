"""Command-line interface for AccelPack scheduler."""

import argparse
import json
import sys
from pathlib import Path

from accelpack.models import Job, Instance
from accelpack.scheduler import AccelPackScheduler


def main():
    parser = argparse.ArgumentParser(description="AccelPack: Multi-Accelerator Workload Packing & Quota-Preserving Scheduler")
    parser.add_argument("--jobs-json", type=Path, required=True, help="Path to JSON file containing list of jobs")
    parser.add_argument("--instances-json", type=Path, required=True, help="Path to JSON file containing compute instances")
    parser.add_argument("--setup-overhead", type=float, default=0.10, help="Per-container startup overhead in hours (default: 0.10h)")
    parser.add_argument("--quota-buffer", type=float, default=2.0, help="Safety quota reserve per account in hours (default: 2.0h)")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional output path to save JSON execution plan")

    args = parser.parse_args()

    with open(args.jobs_json, "r", encoding="utf-8") as f:
        raw_jobs = json.load(f)
    with open(args.instances_json, "r", encoding="utf-8") as f:
        raw_instances = json.load(f)

    jobs = [Job(**j) for j in raw_jobs]
    instances = [Instance(**inst) for inst in raw_instances]

    scheduler = AccelPackScheduler(
        setup_overhead_h=args.setup_overhead,
        quota_buffer_h=args.quota_buffer,
    )

    plan = scheduler.schedule(jobs, instances)
    print(plan.summary())

    if args.out_json:
        # Export structured plan
        data = {
            "total_work_h": plan.total_work_h,
            "total_billed_resource_h": plan.total_billed_resource_h,
            "efficiency_ratio": plan.efficiency_ratio,
            "active_instances": [
                {
                    "instance_id": ip.instance.instance_id,
                    "makespan_h": ip.makespan_h,
                    "billed_h": ip.billed_resource_h,
                    "slots": [
                        {
                            "device_index": s.device_index,
                            "load_h": s.load_h,
                            "jobs": [j.job_id for j in s.jobs],
                        }
                        for s in ip.slots
                    ],
                }
                for ip in plan.active_instances
            ],
            "untouched_instances": [ip.instance.instance_id for ip in plan.untouched_instances],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved execution plan to {args.out_json}")


if __name__ == "__main__":
    main()
