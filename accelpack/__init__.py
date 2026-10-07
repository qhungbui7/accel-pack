"""AccelPack: Multi-Accelerator Workload Packing & Quota-Preserving Job Scheduler."""

from accelpack.models import Job, Instance, DeviceSlot, ExecutionPlan
from accelpack.scheduler import AccelPackScheduler

__all__ = ["Job", "Instance", "DeviceSlot", "ExecutionPlan", "AccelPackScheduler"]
__version__ = "0.1.0"
