"""Per-run call accounting: how many model calls a story cost and how long they took."""
import threading
import time
from dataclasses import dataclass, field

from .llm import LLM


@dataclass
class Metrics:
    calls: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, stage: str, elapsed: float) -> None:
        with self._lock:
            self.calls[stage] = self.calls.get(stage, 0) + 1
            self.seconds[stage] = self.seconds.get(stage, 0.0) + elapsed

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())

    def summary(self) -> str:
        stages = ", ".join(
            f"{stage} x{count} ({self.seconds[stage]:.0f}s)"
            for stage, count in self.calls.items()
        )
        return f"{self.total_calls} model calls: {stages}"


class MeteredLLM:
    """Wraps any LLM and records each call under its stage name."""

    def __init__(self, llm: LLM, metrics: Metrics | None = None):
        self.llm = llm
        self.metrics = metrics or Metrics()

    def text(self, system, prompt, effort="medium"):
        return self._timed("write", self.llm.text, system, prompt, effort)

    def structured(self, system, prompt, schema, effort="medium"):
        return self._timed(
            schema.__name__, self.llm.structured, system, prompt, schema, effort
        )

    def _timed(self, stage, call, *args):
        start = time.monotonic()
        try:
            return call(*args)
        finally:
            self.metrics.record(stage, time.monotonic() - start)
