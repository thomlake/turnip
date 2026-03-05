from dataclasses import dataclass
from typing import Any


class ExperimentRuntimeError(RuntimeError):
    """Base exception for experiment runtime errors."""


class MissingRunContextError(ExperimentRuntimeError):
    """Raised when a @scope function is called outside Program.map context."""


class DataCorruptionError(ExperimentRuntimeError):
    """Raised when persisted data cannot be deserialized."""


@dataclass(slots=True)
class FailedMapItem:
    index: int
    key: str
    trial: int
    error: BaseException


class MapExecutionError(ExperimentRuntimeError):
    def __init__(self, stage: str, failures: list[FailedMapItem]) -> None:
        self.stage = stage
        self.failures = failures
        parts = [f"index={f.index}, key={f.key}, trial={f.trial}, error={type(f.error).__name__}" for f in failures]
        super().__init__(f"map failed for stage '{stage}' with {len(failures)} failures: " + "; ".join(parts))


class SerializationError(ExperimentRuntimeError):
    def __init__(self, scope: str, value: Any) -> None:
        value_type = type(value).__name__
        super().__init__(f"scope '{scope}' returned non-JSON-serializable value of type {value_type}")
