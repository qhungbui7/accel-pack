# AccelPack: Next Steps After the Latest Review

**Purpose:** Turn the current scheduler into a practical, quota-aware workload planner for Kaggle sessions, without assuming that higher utilization always means higher throughput or requiring separate profiling runs.

**Recommended order:** correctness first, then resource-policy flexibility, then concurrency calibration, then optimizer sophistication. Do not start by building a more complex optimizer around runtime assumptions that are not yet trustworthy.

## Executive summary

1. Remove CPU oversubscription as an unconditional placement rejection. Treat CPU demand as a soft cost; keep RAM and per-GPU VRAM as hard limits.
2. Fix the concurrency model so multiple workers on one GPU do not automatically get the same runtime as isolated workers.
3. Make quota accounting, reset boundaries, and per-session time limits explicit and consistent between scheduling, preflight, and reporting.
4. Do not treat `resumable=True` as checkpoint support unless the scheduler can actually represent and schedule resumable segments.
5. Wire preflight into the normal CLI workflow and use telemetry from ordinary runs to improve predictions.
6. Make objective modes truthful, strengthen tests, and update documentation/versioning only after behavior is validated.

---

## P0. Establish a reproducible baseline

Before changing behavior:

- [ ] Work on the intended local branch and confirm whether the latest changes are pushed to GitHub. The public `main` currently declares package version `0.1.0` in `pyproject.toml`; update version metadata only when a release is actually ready.
- [ ] Run the full existing test suite and save the result as the baseline.
- [ ] Keep a small set of representative planning inputs: one GPU-bound workload, one CPU-heavy workload, one VRAM-constrained workload, several jobs sharing one account quota, and a workload near a session/reset boundary.
- [ ] Record both the generated plan and, where available, actual wall-clock duration and quota debit. Treat planned results as predictions, not execution guarantees.

**Exit criterion:** Existing behavior is reproducible, and subsequent changes can be compared against the same inputs.

## P1. Make CPU a soft constraint, while keeping memory limits hard

### Current issue

`accelpack/scheduler.py::_check_hardware_fit()` rejects a candidate when summed estimated CPU demand exceeds `host_cpu_cores`. That makes a prediction such as 5 CPU cores of demand on a 4-vCPU host a hard failure, even though operating systems can schedule more runnable work than there are cores.

### Change

- [ ] Keep hard rejection for estimated aggregate host RAM and per-device VRAM, including configurable safety margins.
- [ ] Replace the unconditional CPU rejection with a CPU contention estimate and a configurable penalty.
- [ ] Use a simple initial measure:

  `cpu_ratio = estimated_concurrent_cpu_demand / available_vcpus`

  `cpu_penalty = max(0, cpu_ratio - 1) ** 2`

- [ ] Apply the penalty as a secondary scheduling preference, or use it to adjust predicted completion time only where there is a defensible estimate. Do not let an arbitrary penalty outweigh a clearly better useful-work outcome.
- [ ] Keep an optional user-configured emergency cap if desired, but do not make a 1.0 ratio a universal feasibility boundary.
- [ ] Document that CPU core-percent readings vary by tool: a process using two full cores can appear near 200%; a 4-core host can reach roughly 400% in tools that report percent per core.

### GPU terminology and policy

- [ ] Do not add together per-process GPU utilization percentages and treat the sum as physical GPU capacity. Standard per-device `nvidia-smi` utilization is reported on a 0–100% scale; a sum across multiple GPUs or a different monitoring metric can exceed 100%.
- [ ] Do not reject GPU placements merely because nominal compute demand or utilization estimates exceed 100%. Treat compute contention as a runtime/throughput estimation problem.
- [ ] Keep per-GPU VRAM as a hard constraint, because memory capacity cannot be safely time-shared in the same way as compute.
- [ ] Keep `slots_per_device` configurable. Do not hard-code two slots per GPU as universally optimal.

**Exit criterion:** A workload whose only issue is CPU demand modestly above nominal vCPU capacity can still be scheduled; RAM/VRAM violations remain blocked; the plan reports CPU oversubscription clearly.

## P2. Correct the concurrency/runtime assumption

### Current issue

`accelpack/models.py::DeviceSlot.assign()` adds each job's isolated `duration_h` to a slot, while `InstancePlan.device_loads_h` takes the maximum load across slots on a physical device. This can model jobs in separate slots as running in parallel with no slowdown. The test `test_concurrent_slot_packing_speedup` currently encodes the ideal result: four 1-hour jobs on two GPUs finish in about 1.1 hours with two slots per GPU. That is only valid if measured execution supports that assumption.

### Change

- [ ] Keep a clean distinction between the number of physical GPUs and the number of logical worker slots per GPU.
- [ ] Add a runtime/throughput estimate for concurrent workers on the same GPU. Prefer empirical profiles from ordinary executions; do not launch mandatory standalone benchmark jobs.
- [ ] Key relevant measurements by at least workload/model family, hardware class, and concurrent workers per GPU. Avoid mixing single-worker and multi-worker observations into one undifferentiated throughput average.
- [ ] When there is no useful concurrency history, use an explicitly conservative or configurable fallback instead of silently assuming zero slowdown. Make the uncertainty visible in the plan.
- [ ] Use the concurrency-adjusted duration when estimating per-slot load, makespan, session feasibility, and quota debit. Make sure each of these uses the same runtime assumptions.
- [ ] Replace the idealized concurrency test with multiple cases: ideal scaling, moderate slowdown, severe slowdown, memory contention, and unknown profile. Assert calculations against the selected assumptions rather than assuming a universal speedup.

**Exit criterion:** Adding slots may improve a plan, but cannot automatically claim perfect speedup without evidence or an explicit idealized configuration.

## P3. Unify quota, reset, and session-limit logic

### Current issues to close

- The scheduler estimates billed quota with `debit_rate * makespan_h`, while preflight's aggregate warning compares total worker-hours with summed quota hours. Those are not equivalent in general.
- `PreflightValidator` notices when `reset_at` is in the past, but does not use a future reset boundary to determine whether the proposed plan can finish or whether quota observations need refreshing.
- An instance carries its own `remaining_quota_h`. If multiple hardware/session entries represent the same Kaggle account, separate entries could accidentally duplicate the account's quota unless account-level ownership is explicit.
- `resumable=True` currently allows preflight to pass an overlong job, but this flag does not itself implement checkpointing or split the job into valid session-sized segments.

### Change

- [ ] Define an explicit account/quota-pool identifier. Aggregate every session or hardware configuration attached to the same account against one shared remaining quota balance.
- [ ] Document and configure how the target provider debits quota. Keep `debit_rate` only if the accounting model is well-defined; do not assume `device_count * wall_time` or `wall_time` without validating the actual rule for the target workload/provider.
- [ ] Make preflight evaluate a plan-compatible estimate of billed quota and the configured safety reserve instead of using raw sum of job worker-hours as if it were quota debit.
- [ ] Define the reset policy: when an account's quota reset is known, decide whether work must finish before reset, whether a run may cross the boundary, and when account quota must be refreshed. Do not infer provider behavior from the timestamp alone.
- [ ] Wire reset-aware checks into actual scheduling decisions, not just informational messages.
- [ ] Until checkpoint segmentation is implemented, make an over-session job a blocker even if `resumable=True`, unless its input already explicitly represents valid segments that fit the session limit.
- [ ] If checkpoint segmentation is implemented later, represent segment duration, checkpoint overhead, restart cost, and continuation dependency explicitly.

**Exit criterion:** Scheduler, preflight, and summary agree on predicted quota debit; shared account quota cannot be counted twice; reset/session behavior is explicit; no job is marked feasible merely because a boolean claims it is resumable.

## P4. Connect preflight and reconciliation to the normal workflow

### Change

- [ ] Integrate `PreflightValidator` into the normal CLI path before scheduling or exporting a plan. Show blockers and warnings clearly; block on real feasibility errors by default.
- [ ] Preserve a lightweight workflow: read supplied configuration, validate it, schedule it, emit a plan. Do not require expensive device diagnostics or dedicated benchmark runs.
- [ ] After real runs, record predicted versus actual duration, actual session makespan, actual quota debit, success/failure, peak VRAM/RAM, hardware class, and concurrency level where available.
- [ ] Include CPU observations when they are readily available, but do not require a special profiler to collect them.
- [ ] Handle missing telemetry explicitly; absence of a metric must not be interpreted as zero resource usage.
- [ ] Surface persistence/serialization failures rather than silently discarding the whole telemetry history.
- [ ] Add a report showing prediction error and success/failure rates by workload/hardware/concurrency category.

**Exit criterion:** The default CLI catches invalid plans before export, and completed runs produce reusable measurements without a separate profiling workflow.

## P5. Make optimization objectives match their names

Current objective handling mainly changes job ordering: `MAXIMIZE_WORK` has a different sort, while the other modes follow LPT. That is not enough to guarantee different optimization behavior.

- [ ] Define the primary outcome for each mode:
  - `COMPLETE_ALL`: schedule all jobs if feasible; minimize predicted makespan/quota cost under hard constraints.
  - `MAXIMIZE_WORK`: maximize completed priority-weighted work before relevant quota/session/reset boundaries; make partial scheduling explicit.
  - `MAXIMIZE_EFFICIENCY`: maximize useful completed work per predicted quota debit, without relying on a misleading ratio that is clipped at 100%.
- [ ] Keep hard feasibility separate from soft preferences.
- [ ] Start with simple, testable heuristics. Add MILP/CP-SAT only if measured scenarios show that a better optimizer is necessary.
- [ ] Make tie-breaking deterministic so schedules and tests are reproducible.

**Exit criterion:** Small constructed examples demonstrate that the objective modes make different, justifiable decisions.

## P6. Testing and documentation pass

Add or update tests for:

- [ ] CPU overcommit allowed with a penalty or warning.
- [ ] RAM and per-GPU VRAM remain hard constraints, including safety margins.
- [ ] Multiple worker slots share one physical GPU's VRAM capacity.
- [ ] Concurrency profiles affect estimated runtime/makespan.
- [ ] Missing or low-confidence concurrency profiles use the documented fallback.
- [ ] Shared account quota is not double-counted across instances/sessions.
- [ ] Debit rate and setup overhead are reflected consistently in scheduler, preflight, and reporting.
- [ ] Reset timestamps in the past, in the future, missing, and malformed.
- [ ] Over-session non-resumable jobs fail; `resumable=True` is not a false pass without segmentation.
- [ ] Partial scheduling reports every unassigned job and its reason.
- [ ] Objective modes differ on cases where their goals conflict.
- [ ] Prediction-versus-actual telemetry survives save/load and malformed history is reported.

Then update the README and package metadata to describe only implemented behavior. In particular, revise claims that CPU oversubscription is always blocked, avoid presenting theoretical concurrent speedup as guaranteed, document quota-debit assumptions, and align `pyproject.toml`'s version with the intended release.

**Exit criterion:** Tests pass, examples match the implemented semantics, and documentation does not promise unsupported checkpointing or ideal concurrency.

---

## Suggested implementation sequence

1. Baseline tests and branch/version check.
2. CPU soft constraint and resource-policy tests.
3. Concurrency slowdown representation and realistic tests.
4. Quota-account ownership, billing consistency, and reset/session feasibility.
5. Fix resumable-job feasibility semantics.
6. CLI preflight integration and run reconciliation.
7. Telemetry profile isolation and prediction-error reporting.
8. Objective-mode behavior, README cleanup, and version release.

This order deliberately postpones optimizer complexity until resource estimates are honest.

## What not to do yet

- Do not assume that `slots_per_device=2` is always better than `1`.
- Do not create a mandatory profiling phase that consumes quota before useful work.
- Do not treat CPU demand above vCPU count as an automatic failure.
- Do not relax RAM/VRAM safety checks simply because utilization looks low in one snapshot.
- Do not claim reset-aware execution or checkpoint-resume support until the scheduler actually enforces it.
- Do not optimize for a high utilization percentage alone. Measure completed useful work per wall-clock hour and per unit of quota, alongside failure/OOM risk.

## Practical success metrics

Track these from normal executions:

- Useful training steps or completed jobs per wall-clock hour.
- Useful training steps or completed jobs per quota-hour debited.
- Planned versus actual session makespan and quota debit.
- Job completion/failure rate, OOM rate, and interrupted-work waste.
- Peak host RAM and per-GPU VRAM versus configured margins.
- Results stratified by GPU hardware class and workers per GPU.

The primary success criterion is **more useful work completed within the available quota and session boundaries without materially increasing failed or interrupted jobs**, not maximizing CPU/GPU utilization by itself.
