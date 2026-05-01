from __future__ import annotations

import numpy as np


def mae(actual, forecast) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual, forecast) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def r2(actual, forecast) -> float:
    a = np.asarray(actual)
    f = np.asarray(forecast)
    ss_res = float(np.sum((a - f) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def evaluate(actual, forecast) -> dict[str, float]:
    return {"mae": mae(actual, forecast), "rmse": rmse(actual, forecast), "r2": r2(actual, forecast)}
