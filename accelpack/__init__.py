"""AccelPack: Multi-accelerator bin-packing, quota-preserving scheduler, and execution lifecycle."""

__version__ = "0.2.0"

from accelpack.models import (
    DeviceSlot,
    ExecutionPlan,
    Instance,
    InstancePlan,
    Job,
    ObjectiveMode,
    PackingStrategy,
)
from accelpack.scheduler import AccelPackScheduler, QuotaExceededException
from accelpack.estimator import ProfileDurationEstimator, ProfileRecord
from accelpack.preflight import PreflightValidator, PreflightReport, PreflightIssue
from accelpack.reconciliation import (
    ExecutionReconciler,
    JobExecutionRecord,
    SessionReconciliationReport,
    load_and_validate_profiles,
    reconcile_and_persist_profiles,
    ProfileLoadException,
    HistoryLoadException,
    HistoryPersistenceException,
    LockAcquisitionException,
)

__all__ = [
    "DeviceSlot",
    "ExecutionPlan",
    "Instance",
    "InstancePlan",
    "Job",
    "ObjectiveMode",
    "PackingStrategy",
    "AccelPackScheduler",
    "QuotaExceededException",
    "ProfileDurationEstimator",
    "ProfileRecord",
    "PreflightValidator",
    "PreflightReport",
    "PreflightIssue",
    "ExecutionReconciler",
    "JobExecutionRecord",
    "SessionReconciliationReport",
    "load_and_validate_profiles",
    "reconcile_and_persist_profiles",
    "ProfileLoadException",
    "HistoryLoadException",
    "HistoryPersistenceException",
    "LockAcquisitionException",
]
