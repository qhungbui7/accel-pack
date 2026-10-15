# AccelPack

**Generalized Multi-Accelerator Workload Packing and Quota-Preserving Job Scheduler**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

AccelPack is an algorithmic scheduler and resource-allocation engine designed for distributed computing, cloud GPU clusters, multi-node compute fleets, and shared quota-bounded execution environments.

It formulates workload scheduling across multi-device instances as a **Constrained Multi-Dimensional Bin Packing & Job Scheduling Problem with Instance Preservation Objectives**.

---

## The Problem AccelPack Solves

In modern cloud computing, containerized multi-device nodes (e.g., dual-accelerator or quad-accelerator instances) are billed by **instance wall-clock session time multiplied by a platform debit rate**:

$$\\text{Billed Resource Time} = \\text{debit\\_rate} \\times \\text{Session Wall-Clock Time}$$

When experiment suites are dispatched naively (e.g., fanning out one job per instance):
1. **Unused Accelerators Incur Full Costs:** If 1 job runs on Device 0 while Device 1 sits idle, the idle device continues to burn billed quota.
2. **Setup Overhead Multiplies:** Cold-start provisioning (container boot, cloning, dependency installation) is paid redundantly across every activated instance.
3. **Account / Fleet Quota Fragmentation:** Chipping away at every account or cluster partition consumes emergency compute headroom needed for unexpected retraining runs.

AccelPack optimizes execution to:
- **Minimize opened instances** via heuristic packing (preserving untouched nodes for future workloads).
- **Level loads between accelerators** on the same instance to minimize idle-device billing waste.
- **Enforce hard session timeouts, reset deadlines, and safety margins** ($M_a \\le S_{\\max}$, $Q_{\\text{billed}} \\le Q_{\\text{rem}} - \\beta$).
- **Respect hardware boundaries** (CPU cores, RAM/RSS, accelerator VRAM per device, and target hardware classes).

> **Scope Note (v0.2):** AccelPack schedules independent single-device jobs (`Job.device_count == 1`) across multi-device instances, supporting multiple concurrent worker slots per device with empirical slowdown factors. Multi-device single-job gang scheduling (`Job.device_count != 1`, e.g., single jobs spanning multiple GPUs via DDP or FSDP) raises `NotImplementedError` and is not supported in v0.2.

---

## Scheduling Strategies & Objectives

AccelPack provides clear heuristic placement strategies:
- **`CONCENTRATE` (Greedy LPT Packing):** Fills one instance completely before opening another. Maximizes untouched instance preservation and minimizes redundant startup overhead.
- **`BALANCE` (Water-Filling):** Evenly distributes device load and remaining quota across active instances to balance wall-clock makespan.

Supported optimization objectives:
- **`COMPLETE_ALL`:** Default full assignment ensuring all candidate jobs are placed within quotas and session limits.
- **`MAXIMIZE_EFFICIENCY`:** Prioritizes packing onto already-active instances and concurrent slots to maximize useful worker-hours per billed quota-hour.
- **`MAXIMIZE_WORK`:** Priority-weighted duration packing when partial scheduling (`allow_partial=True`) is permitted.

---

## Installation

```bash
pip install accelpack
```

Or from source:
```bash
git clone https://github.com/qhungbui7/accel-pack.git
cd accel-pack
pip install -e .
```

---

## Quickstart

```python
from accelpack import Job, Instance, AccelPackScheduler, PackingStrategy

# 1. Define instances (e.g., dual-accelerator nodes with quotas)
instances = [
    Instance(instance_id="node-01", device_count=2, remaining_quota_h=24.0, max_session_h=11.0),
    Instance(instance_id="node-02", device_count=2, remaining_quota_h=24.0, max_session_h=11.0),
    Instance(instance_id="node-03", device_count=2, remaining_quota_h=24.0, max_session_h=11.0),
]

# 2. Define jobs with predicted or profiled runtimes
jobs = [
    Job(job_id="train_model_s1", duration_h=1.8, vram_gb=0.6, ram_gb=1.5),
    Job(job_id="train_model_s2", duration_h=1.8, vram_gb=0.6, ram_gb=1.5),
    Job(job_id="train_model_s3", duration_h=1.8, vram_gb=0.6, ram_gb=1.5),
    Job(job_id="eval_model_s1", duration_h=0.5, vram_gb=0.3, ram_gb=1.0),
    Job(job_id="eval_model_s2", duration_h=0.5, vram_gb=0.3, ram_gb=1.0),
    Job(job_id="eval_model_s3", duration_h=0.5, vram_gb=0.3, ram_gb=1.0),
]

# 3. Pack the workload using CONCENTRATE strategy
scheduler = AccelPackScheduler(
    strategy=PackingStrategy.CONCENTRATE,
    setup_overhead_h=0.10,
    quota_buffer_h=2.0,
)
plan = scheduler.schedule(jobs, instances)

# 4. Inspect generated execution plan
print(plan.summary())
```

### Output:
```text
======================================================================
ACCELPACK WORKLOAD SCHEDULE SUMMARY
Total Jobs: 6
Active Instances: 1 / 3 (2 preserved untouched)
Total Work: 6.90 worker-hours
Total Billed Resource Time: 3.70 quota-hours
Quota Productivity: 186.5% (1.86 worker-hours / billed quota-hour)
Global Makespan: 3.70 hours
----------------------------------------------------------------------
[node-01 (Account: node-01)] Makespan: 3.70h | Billed: 3.70h | Quota Remaining After: 20.30h | Workers: 2/2
  CPU Demand: 2.0/4.0 cores (50%)
  ├── Device 0 (Wall-load: 3.60h, Peak VRAM: 0.60/16.0 GB):
  │     [Queue] train_model_s3 (1.80h), train_model_s1 (1.80h)
  ├── Device 1 (Wall-load: 3.30h, Peak VRAM: 0.60/16.0 GB):
  │     [Queue] train_model_s2 (1.80h), eval_model_s3 (0.50h), eval_model_s2 (0.50h), eval_model_s1 (0.50h)
----------------------------------------------------------------------
Preserved Instances (Untouched): node-02 (24.0h), node-03 (24.0h)
======================================================================
```

---

## Core Capabilities & Safety Guarantees

- **Sequential RAM Reuse:** Sequential jobs assigned to the same worker slot track peak RAM usage without spurious aggregate sum rejections.
- **Fail-Safe Reset Enforcement:** Blocks execution safely if reset deadlines have passed or timestamp metadata is missing/malformed.
- **Shared Account Normalization:** Eliminates conflicting quota declarations for instances sharing an `account_id` by adopting the authoritative minimum.
- **Resumable Checkpoint Segments:** Automatically expands workloads exceeding session timeouts into dependency-ordered discrete segment assignments (`expand_segments=True`). AccelPack generates sequential execution plans across worker slots and multi-session instances with dependency-aware critical-path timing; the execution controller (e.g., runner or container orchestrator) is responsible for artifact staging, checkpoint saving/resumption, and honoring inter-session dependency metadata.

### Execution Orchestration Boundary

AccelPack is an algorithmic planning and scheduling engine, not a workflow orchestrator:
- **What AccelPack does:** Generates dependency-ordered, resource-checked assignments, calculates critical-path makespan across multi-session stages, binds sequential segments to worker slots, and evaluates quota safety.
- **What the caller / execution controller does:** Dispatches container instances, monitors process liveness, serializes and stages checkpoint artifacts between sessions, verifies checkpoint availability before launching downstream dependent segments, and feeds back actual runtimes to the telemetry reconciler.
- **Hardware-Class Compatibility:** Enforces target hardware family matching before checking physical resource envelopes.
- **Configurable Concurrency & Slowdown:** Multi-slot device packing with realistic concurrency slowdown degradation models.
- **Soft CPU Contention Penalty:** Quadratic penalty for CPU oversubscription with configurable emergency hard ceilings.
- **Closed-Loop Telemetry & Profiling:** CLI and SDK support for `ExecutionReconciler` and `ProfileDurationEstimator` to continuously calibrate runtime predictions from completed runs.

---

## License

Apache License 2.0.
