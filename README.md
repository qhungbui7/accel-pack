# AccelPack

**Generalized Multi-Accelerator Workload Packing and Quota-Preserving Job Scheduler**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

AccelPack is an algorithmic scheduler and resource-allocation engine designed for distributed computing, cloud GPU clusters, multi-node compute fleets, and shared quota-bounded execution environments. 

It formulates workload scheduling across multi-device instances as a **Constrained Multi-Dimensional Bin Packing & Job Scheduling Problem with Instance Preservation Objectives**.

---

## The Problem AccelPack Solves

In modern cloud computing, containerized multi-device nodes (e.g., dual-accelerator or quad-accelerator instances) are frequently billed by **instance wall-clock time multiplied by device count ($G$)**:

$$\text{Billed Resource Time} = G \times \text{Session Wall-Clock Time}$$

When experiment suites are dispatched naively (e.g., fanning out one job per instance):
1. **Unused Accelerators Incur Full Costs:** If 1 job runs on Device 0 while Device 1 sits idle, the idle device is billed at full cost.
2. **Setup Overhead Multiplies:** Cold-start provisioning (container boot, cloning, dependency installation) is paid redundantly per instance.
3. **Account / Fleet Quota Fragmentation:** Chipping away at every account or cluster partition consumes emergency compute headroom needed for unexpected heavy retraining.

AccelPack optimizes execution to:
- **Minimize the number of opened instances** ($y_a \in \{0, 1\}$).
- **Balance loads between accelerators** on the same instance ($|D_{a,0} - D_{a,1}| \to 0$) to eliminate idle-device billing waste.
- **Enforce hard session timeout and safety margins** ($M_a \le S_{\max}$, $Q_{\text{billed}} \le Q_{\text{rem}} - \beta$).
- **Respect hardware boundaries** (CPU cores, RAM/RSS, VRAM per process).

---

## Mathematical Formulation

Given a workload of $N$ jobs $\mathcal{J} = \{j_1, \dots, j_N\}$ with estimated execution runtimes $\hat{T}_j$, and a set of compute instances $\mathcal{A} = \{a_1, \dots, a_M\}$ with device count $G_a$ and remaining quota $Q_a$:

$$\min_{\mathbf{x}, \mathbf{y}} \quad \underbrace{\alpha \sum_{a \in \mathcal{A}} y_a}_{\text{Minimize Opened Instances}} + \underbrace{\mu \sum_{a \in \mathcal{A}} \sum_{k=0}^{G_a-1} D_{a, k}}_{\text{Minimize Total Device Hours}} + \underbrace{\gamma \sum_{a \in \mathcal{A}} \max_{k, l} |D_{a, k} - D_{a, l}|}_{\text{Inter-Device Load Balancing}} + \underbrace{\lambda \max_{a \in \mathcal{A}} M_a}_{\text{Bounded Wall-Clock Time}}$$

Subject to:
1. **Complete Assignment:** $\sum_{a \in \mathcal{A}} \sum_{k=0}^{G_a-1} x_{j, a, k} = 1 \quad \forall j \in \mathcal{J}$
2. **Instance Activation:** $x_{j, a, k} \le y_a \quad \forall j, a, k$
3. **Session Deadline:** $M_a = \max_k D_{a, k} + t_{\text{setup}} \le S_{\max} \cdot y_a \quad \forall a$
4. **Quota Capacity:** $G_a \cdot M_a \le Q_a - \beta \quad \forall a$
5. **Hardware Constraints:** Host RAM $\le \text{RAM}_{\max}$, CPU workers $\le \text{vCPU}_{\max}$.

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
from accelpack import Job, Instance, AccelPackScheduler

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

# 3. Optimize and pack the workload
scheduler = AccelPackScheduler(setup_overhead_h=0.10, quota_buffer_h=2.0)
plan = scheduler.schedule(jobs, instances)

# 4. Inspect optimal allocation
print(plan.summary())
```

### Output:
```text
======================================================================
ACCELPACK SCHEDULE SUMMARY
Total Workload: 6 Jobs (6.90 device-hours)
Instances Opened: 1 / 3 (2 instances preserved untouched)
Makespan: 3.55 hours
Total Billed Resource Time: 7.10 device-hours (Efficiency: 97.2%)

[node-01] Makespan: 3.55h | Billed: 7.10h | Quota Remaining After: 16.90h
  ├── Device 0 (Load: 3.45h):
  │     1. train_model_s1 (1.80h)
  │     2. eval_model_s1 (0.50h)
  │     3. eval_model_s2 (0.50h)
  └── Device 1 (Load: 3.45h):
        1. train_model_s2 (1.80h)
        2. train_model_s3 (1.80h)

Preserved Instances (Untouched): node-02 (24.0h), node-03 (24.0h)
======================================================================
```

---

## Features

- **Multi-Device Co-Scheduling:** Optimal assignment across devices on heterogeneous or homogeneous multi-GPU/accelerator nodes.
- **Hardware-Aware Bounds:** Guards against host RAM exhaustion, vCPU saturation, and VRAM fragmentation.
- **Idle-Device Waste Minimization:** Balances parallel device queues to prevent idle devices from inflating billed instance time.
- **Instance Preservation:** Greedy LPT packing keeps instances untouched for future compute needs.
- **Extensible Profiling:** Built-in hooks for throughput (FPS, steps/sec) and telemetry interpolation.

---

## License

Apache License 2.0.
