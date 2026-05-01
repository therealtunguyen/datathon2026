from __future__ import annotations

import numpy as np
import pandas as pd


RUN_CONFIG = {
    "name": "baseline",
    "weight_scheme": "high_era",
    "q_boost": 3.0,
    "alpha": 0.60,
    "cr": 1.32,
    "cc": 1.36,
    "use_diversity_models": False,
    "diversity_rev_weight": 0.10,
    "diversity_cog_weight": 0.10,
    "write_candidate_sweep": False,
    "candidate_grid": [
        {"name": "best_repeat", "cr": 1.32, "cc": 1.36, "alpha": 0.60},
        {"name": "rev_133_cogs_136", "cr": 1.33, "cc": 1.36, "alpha": 0.60},
        {"name": "rev_1325_cogs_136", "cr": 1.325, "cc": 1.36, "alpha": 0.60},
        {"name": "rev_1315_cogs_136", "cr": 1.315, "cc": 1.36, "alpha": 0.60},
        {"name": "rev_132_cogs_135", "cr": 1.32, "cc": 1.35, "alpha": 0.60},
        {"name": "rev_133_cogs_135", "cr": 1.33, "cc": 1.35, "alpha": 0.60},
        {"name": "rev_132_alpha055", "cr": 1.32, "cc": 1.36, "alpha": 0.55},
        {"name": "rev_132_alpha065", "cr": 1.32, "cc": 1.36, "alpha": 0.65},
    ],
}


def blend_base_and_specialist(base_pred: np.ndarray, specialist_pred: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * specialist_pred + (1 - alpha) * base_pred


def make_raw_predictions(
    preds: dict[str, np.ndarray],
    alpha: float = 0.60,
    diversity_rev_weight: float = 0.0,
    diversity_cog_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    lgb_blend_rev = blend_base_and_specialist(preds["p_lgb_rev"], preds["p_spec_rev"], alpha)
    lgb_blend_cog = blend_base_and_specialist(preds["p_lgb_cogs"], preds["p_spec_cogs"], alpha)
    cat_blend_rev = blend_base_and_specialist(preds["p_cat_rev"], preds["p_cat_spec_rev"], alpha)
    cat_blend_cog = blend_base_and_specialist(preds["p_cat_cogs"], preds["p_cat_spec_cogs"], alpha)

    raw_rev = 0.10 * preds["p_ridge_rev"] + 0.45 * cat_blend_rev + 0.45 * lgb_blend_rev
    raw_cog = 0.10 * preds["p_ridge_cogs"] + 0.45 * cat_blend_cog + 0.45 * lgb_blend_cog

    if diversity_rev_weight > 0 and "p_lgb_recent_rev" in preds and "p_cat_post_rev" in preds:
        div_rev = 0.50 * preds["p_lgb_recent_rev"] + 0.50 * preds["p_cat_post_rev"]
        raw_rev = (1 - diversity_rev_weight) * raw_rev + diversity_rev_weight * div_rev

    if diversity_cog_weight > 0 and "p_lgb_recent_cogs" in preds and "p_cat_post_cogs" in preds:
        div_cog = 0.50 * preds["p_lgb_recent_cogs"] + 0.50 * preds["p_cat_post_cogs"]
        raw_cog = (1 - diversity_cog_weight) * raw_cog + diversity_cog_weight * div_cog

    return raw_rev, raw_cog


def candidate_tag(config: dict) -> str:
    return (
        f"{config.get('name', 'candidate')}_"
        f"cr{int(round(config['cr'] * 100)):03d}_"
        f"cc{int(round(config['cc'] * 100)):03d}_"
        f"alpha{int(round(config['alpha'] * 100)):03d}"
    )


def build_submission(test: pd.DataFrame, raw_rev: np.ndarray, raw_cog: np.ndarray, cr: float, cc: float) -> pd.DataFrame:
    final_rev = np.clip(cr * raw_rev, 1.0, None)
    final_cog = np.clip(cc * raw_cog, 1.0, None)
    return pd.DataFrame(
        {
            "Date": test["Date"].dt.strftime("%Y-%m-%d"),
            "Revenue": np.round(final_rev, 2),
            "COGS": np.round(final_cog, 2),
        }
    )


def summarize_submission(submission: pd.DataFrame, label: str = "submission") -> dict[str, float | int | str]:
    df = submission.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return {
        "label": label,
        "rows": int(len(df)),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "rev_min": float(df["Revenue"].min()),
        "rev_mean": float(df["Revenue"].mean()),
        "rev_max": float(df["Revenue"].max()),
        "cogs_min": float(df["COGS"].min()),
        "cogs_mean": float(df["COGS"].mean()),
        "cogs_max": float(df["COGS"].max()),
        "mean_cogs_rev_ratio": float((df["COGS"] / df["Revenue"]).mean()),
        "cogs_ge_revenue_days": int((df["COGS"] >= df["Revenue"]).sum()),
    }


def predictions_frame_to_dict(predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    return {col: predictions[col].to_numpy() for col in predictions.columns if col != "Date"}
