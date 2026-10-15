# AccelPack Next-Step Checklist

Use this as the implementation checklist for the next iteration.

## P0 — Baseline
- [x] Confirm the intended local branch contains the latest changes; reconcile it with GitHub `main`.
- [x] Run the full test suite and record baseline results.
- [x] Keep representative workload configs for CPU-heavy, GPU-heavy, VRAM-limited, quota-limited, and reset-boundary cases.

## P1 — CPU and GPU policy
- [x] In `accelpack/scheduler.py::_check_hardware_fit()`, remove CPU demand > vCPU count as an unconditional rejection.
- [x] Add a configurable CPU oversubscription penalty/warning; treat it as a soft preference.
- [x] Keep host RAM and per-device VRAM limits hard, with safety margins.
- [x] Explain CPU percentage conventions and distinguish GPU compute utilization from VRAM capacity.
- [x] Add tests showing CPU overcommit can be scheduled but RAM/VRAM violations cannot.

## P2 — Realistic concurrent runtime
- [x] Review `DeviceSlot.assign()` and `InstancePlan.device_loads_h` runtime assumptions.
- [x] Add concurrency-aware duration/throughput estimates, grouped by workload/model, hardware class, and workers per GPU.
- [x] Use an explicit conservative/configurable fallback when profile evidence is missing.
- [x] Replace the test that assumes 4 × 1-hour jobs always finish in 1.1 hours with tests for varied slowdown assumptions.

## P3 — Quota, account identity, reset, sessions
- [x] Add an account/quota-pool identifier so the same account's quota cannot be counted multiple times.
- [x] Define and document provider-specific quota debit semantics; use them consistently in planning and preflight.
- [x] Make reset boundaries affect scheduling, not just produce informational preflight output.
- [x] Make preflight quota estimates consistent with `debit_rate`, setup overhead, and the planned makespan.
- [x] Do not let `resumable=True` bypass the session blocker unless actual segments/checkpoint support are modeled.

## P4 — Workflow and telemetry
- [x] Run preflight in the normal CLI flow before schedule export.
- [x] Record predicted/actual duration, session makespan, quota debit, success/failure, hardware class, concurrency, and observed memory peaks where available.
- [x] Do not require standalone profiling runs.
- [x] Report missing telemetry and persistence errors instead of silently discarding records.
- [x] Isolate performance profiles by hardware and concurrency configuration.

## P5 — Objective semantics
- [x] Define separate acceptance tests for `COMPLETE_ALL`, `MAXIMIZE_WORK`, and `MAXIMIZE_EFFICIENCY`.
- [x] Use simple deterministic heuristics first; defer MILP/CP-SAT until empirical results justify the complexity.
- [x] Replace or qualify the current efficiency metric if worker-hours divided by billed quota-hours is not meaningful for concurrent jobs.

## P6 — Release quality
- [x] Add tests for resource aggregation, shared quota, reset times, concurrency fallback, checkpoint semantics, objective modes, and reconciliation persistence.
- [x] Update README claims to match actual behavior.
- [x] Align `pyproject.toml` version with the release when ready.
- [x] Compare before/after on the same workloads using useful work per quota-hour, actual makespan, failure/OOM rate, and wasted interrupted work.

## Done means
- [x] The planner can accept reasonable CPU oversubscription without ignoring memory safety.
- [x] Concurrent GPU jobs are not assumed to have perfect scaling without evidence.
- [x] Quota, account identity, reset, and session limits agree across preflight and scheduling.
- [x] Ordinary runs improve the scheduler's estimates without a mandatory profiling stage.
- [x] README examples and tests reflect implemented behavior.
