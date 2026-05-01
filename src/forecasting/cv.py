from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.features import compute_sample_weights
from forecasting.metrics import mae
from forecasting.models import predict_lgb, train_lgb

CV_FOLDS = [
    ("A_primary", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("B_stability", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("C_horizon", "2021-06-30", "2021-07-01", "2022-06-30"),
]


def time_series_cv(
    feat_train: pd.DataFrame,
    sales: pd.DataFrame,
    feature_cols: list[str],
    target: str = "Revenue",
) -> dict[str, dict[str, float]]:
    results = {}
    for fold_name, train_end, val_start, val_end in CV_FOLDS:
        train_mask = sales["Date"] <= train_end
        val_mask = (sales["Date"] >= val_start) & (sales["Date"] <= val_end)
        X_tr = feat_train.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
        y_tr = np.log(sales.loc[train_mask, target].clip(lower=1).to_numpy())
        X_vl = feat_train.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)
        y_vl = sales.loc[val_mask, target].to_numpy()
        weights = compute_sample_weights(sales.loc[train_mask, "Date"], scheme="high_era")

        model = train_lgb(X_tr, y_tr, sample_weight=weights)
        pred = np.exp(predict_lgb(model, X_vl))
        results[fold_name] = {
            "mae": mae(y_vl, pred),
            "n_val": int(val_mask.sum()),
            "best_iter": int(model["best_iter"]),
        }
    return results
