from dataclasses import dataclass
from enum import StrEnum


class ScopeStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


@dataclass(slots=True)
class ScopeRecord:
    experiment: str
    stage: str
    key: str
    trial: int
    scope: str
    attempt: int
    status: ScopeStatus
    data_json: str | None
    exception_type: str | None
    exception_message: str | None
    traceback_text: str | None
