from __future__ import annotations

from abc import ABC, abstractmethod
from jev_benchmark.models import BenchmarkCase, ProviderResult


class BenchmarkProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def classify(self, case: BenchmarkCase) -> ProviderResult:
        raise NotImplementedError
