# AccelPack Integration, Repository Standards & Fleet Optimization Guide

This guide establishes the architectural, operational, and data storage requirements that any repository or experiment pipeline must satisfy to integrate seamlessly with the `accelpack` scheduling and lifecycle management engine.

---

## 1. Core Operating Philosophy: Validate, Schedule, Reconcile

`accelpack` operates on a zero-waste, staged lifecycle principle:

```
[1. Preflight Validation]  -->  [2. Feasibility & Scheduling]  -->  [3. Lightweight Execution]  -->  [4. Reconcile & Learn]
 (Zero GPU cost checks)           (Quota, Slots & Feasibility)       (Lightweight Telemetry)          (Continuous Online Calibration)
```

1. **Preflight Validation (Zero GPU Cost):**  
   Validate accounts, remaining quotas, reset timestamps, active sessions, and hardware feasibility before making scheduling decisions. **Never start a GPU container solely to run diagnostic checks.**
2. **Selective Benchmarking:**  
   Run dedicated profiling **only** when expected future compute savings exceed benchmark cost and risk:
   $$\text{Expected Savings} > \text{Profiling Cost} + \text{Risk}$$
   For normal workloads, reuse existing historical profiles or start conservative (1 job/GPU).
3. **Continuous Online Learning:**  
   Every production training run is a profiling sample. Reconcile predicted vs. actual duration and memory after each session to calibrate the empirical throughput model automatically.

---

## 2. What a Repository Should Store

To enable `accelpack` to make optimal, quota-preserving packing decisions, a target experiment repository must persist and maintain the following clean directory structure and artifacts:

```
repo-root/
├── configs/
│   ├── fleet_manifest.json          # Target account IDs, quotas, and accelerator shapes
│   └── jobs_manifest.json           # Declarative list of workload jobs to schedule
├── profiles/
│   ├── throughput_db.json           # Cached empirical FPS, VRAM, and RAM records per (task, model)
│   └── execution_history.json       # Historical rolling log of completed job runs and reconciled metrics
└── artifacts/
    └── {run_slug}/
        ├── run_summary.json         # Per-job final metrics (updates, ret50, elapsed_sec, fps)
        └── resource-telemetry.csv   # Per-node lightweight resource samples (CPU %, RSS, VRAM)
```

### Required Schemas

#### A. Fleet Instance Manifest (`configs/fleet_manifest.json`)
```json
[
  {
    "instance_id": "account_primary",
    "device_count": 2,
    "remaining_quota_h": 28.5,
    "max_session_h": 11.0,
    "vram_per_device_gb": 16.0,
    "host_ram_gb": 30.0,
    "host_cpu_cores": 4.0,
    "slots_per_device": 2,
    "debit_rate": 1.0,
    "reset_at": "2026-10-17T00:00:00Z"
  }
]
```

#### B. Workload Jobs Manifest (`configs/jobs_manifest.json`)
```json
[
  {
    "job_id": "cartpole_gru_s100",
    "duration_h": 1.25,
    "vram_gb": 2.2,
    "ram_gb": 2.5,
    "cpu_cores": 1.0,
    "priority": 1.0,
    "resumable": true,
    "metadata": {
      "env": "cartpole",
      "model": "gru128",
      "steps": 1024000
    }
  }
]
```

#### C. Post-Execution Job Summary (`artifacts/{run_slug}/run_summary.json`)
Every completed job must emit a summary JSON for the reconciler:
```json
{
  "job_id": "cartpole_gru_s100",
  "status": "complete",
  "final_metrics": {
    "loop/update": 1000,
    "train/ret50": 0.8842,
    "loss/entropy": 0.3210,
    "loss/value_loss": 0.00145,
    "perf/fps": 122.4,
    "perf/elapsed_sec": 4200.0
  },
  "training_stats": {
    "best_ret50": 0.9450,
    "peak_vram_gb": 2.15,
    "peak_host_ram_gb": 2.45
  }
}
```

---

## 3. What a Repository Should Do (Execution Rules & Best Practices)

### Rule 1: Mandatory Thread Pool Capping
When running concurrent worker processes on multi-accelerator nodes (e.g. 4 workers across dual GPUs on a 4-vCPU host), all worker scripts MUST cap CPU thread pools to 1:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```
*Why:* Prevents $4\text{ workers} \times 4\text{ threads} = 16\text{ threads}$ thrashing on 4 vCPUs, which degrades rollout stepping by over $40\%$.

### Rule 2: Device and Log Isolation
- Never let all processes inherit the default `cuda:0` device. The launcher or notebook must explicitly bind each worker:
  ```python
  # Worker 0 (Slot 0 on GPU 0): CUDA_VISIBLE_DEVICES="0" or torch.cuda.set_device(0)
  # Worker 1 (Slot 1 on GPU 0): CUDA_VISIBLE_DEVICES="0" or torch.cuda.set_device(0)
  # Worker 2 (Slot 0 on GPU 1): CUDA_VISIBLE_DEVICES="1" or torch.cuda.set_device(1)
  # Worker 3 (Slot 1 on GPU 1): CUDA_VISIBLE_DEVICES="1" or torch.cuda.set_device(1)
  ```
- Redirect each worker's stdout/stderr to an isolated log file (`logs/{job_id}.log`).

### Rule 3: Resumable Checkpointing Before Session Limits
- For jobs with duration $> 2.0\text{ hours}$, write periodic state checkpoints (`artifacts/{run_slug}/ckpt.pt`) every 200 PPO updates.
- If a session approaches its 11.0h limit, wrap up gracefully, package the `.part` archive, and allow `accelpack` to schedule the remaining steps in the next session.

---

## 4. Standard Preflight & Scheduling Workflow in Python

```python
from pathlib import Path
from accelpack import (
    AccelPackScheduler,
    Instance,
    Job,
    ObjectiveMode,
    PackingStrategy,
    PreflightValidator,
    ProfileDurationEstimator,
    ExecutionReconciler,
)

# 1. Load profiles and estimate job durations
estimator = ProfileDurationEstimator(default_fps=100.0)
estimator.register_profile("cartpole", "gru128", fps=120.0, vram_gb=2.2, ram_gb=2.5)

jobs = [
    estimator.create_job(job_id=f"job_{i}", steps=1_024_000, env_id="cartpole", model="gru128")
    for i in range(8)
]

instances = [
    Instance(
        instance_id="account_1",
        device_count=2,
        remaining_quota_h=25.0,
        max_session_h=11.0,
        slots_per_device=2,  # 2 concurrent workers per physical GPU
    )
]

# 2. Preflight validation (Zero GPU cost)
validator = PreflightValidator(min_quota_buffer_h=2.0)
report = validator.validate(jobs, instances)
if not report.passed:
    print(report.format_text())
    raise SystemExit("Preflight blockers detected. Launch aborted.")

# 3. Schedule with multi-accelerator packing
scheduler = AccelPackScheduler(
    strategy=PackingStrategy.BALANCE,
    objective=ObjectiveMode.COMPLETE_ALL,
    setup_overhead_h=0.10,
    quota_buffer_h=2.0,
)
plan = scheduler.schedule(jobs, instances)
print(plan.summary())

# 4. Post-Execution Reconciliation (Learn continuously)
reconciler = ExecutionReconciler(history_file="profiles/execution_history.json")
# After container finishes:
reconciler.record_job(
    job_id="job_0",
    env_id="cartpole",
    model="gru128",
    steps=1_024_000,
    predicted_duration_h=2.37,
    actual_duration_h=2.15,
    success=True,
    instance_id="account_1",
    peak_vram_gb=2.18,
    peak_ram_gb=2.40,
    estimator=estimator,  # Continuously calibrates estimator with moving average
)
```
