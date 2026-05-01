from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class ForecastModel(ABC):
    """Common interface: fit on a sales history, predict over a horizon index."""

    name: str = "base"

    @abstractmethod
    def fit(self, sales: pd.DataFrame, features: pd.DataFrame) -> "ForecastModel":
        ...

    @abstractmethod
    def predict(self, horizon_index: pd.DatetimeIndex, features: pd.DataFrame) -> pd.Series:
        """Return Revenue forecast indexed by horizon_index (linear scale, ≥0)."""

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "ForecastModel":
        raise NotImplementedError
