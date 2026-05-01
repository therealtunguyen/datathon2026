from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forecasting.config import ARTIFACT_DIR, CONFIG_DIR, DATA_DIR, SUBMISSION_PATH
from forecasting.cv import time_series_cv
from forecasting.data import load_competition_data
from forecasting.ensemble import (
    RUN_CONFIG,
    build_submission,
    make_raw_predictions,
    predictions_frame_to_dict,
    summarize_submission,
)
from forecasting.features import build_business_signals, build_train_test_features, compute_sample_weights
from forecasting.models import (
    predict_catboost,
    predict_lgb,
    predict_ridge,
    train_catboost,
    train_lgb,
    train_ridge,
)


def train_and_predict(
    config: dict | None = None,
    data_dir: Path | str | None = None,
    out_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict]:
    run_config = dict(RUN_CONFIG)
    if config:
        run_config.update(config)

    data = load_competition_data(data_dir or DATA_DIR)
    sales = data["sales"]
    test = data["test"]
    business_signals = build_business_signals(data)
    feat_train, feat_test, feature_cols = build_train_test_features(sales, test, business_signals)

    cv_results = time_series_cv(feat_train, sales, feature_cols, target="Revenue")
    X_train = feat_train[feature_cols].to_numpy(dtype=np.float32)
    X_test = feat_test[feature_cols].to_numpy(dtype=np.float32)
    y_train_rev = np.log(sales["Revenue"].clip(lower=1).to_numpy())
    y_train_cogs = np.log(sales["COGS"].clip(lower=1).to_numpy())

    base_weights = compute_sample_weights(sales["Date"], scheme=run_config["weight_scheme"])

    ridge_rev, stats_rev = train_ridge(X_train, y_train_rev, alpha=3.0)
    ridge_cogs, stats_cogs = train_ridge(X_train, y_train_cogs, alpha=3.0)
    p_ridge_rev = np.exp(predict_ridge(ridge_rev, X_test, stats_rev))
    p_ridge_cogs = np.exp(predict_ridge(ridge_cogs, X_test, stats_cogs))

    lgb_rev = train_lgb(X_train, y_train_rev, sample_weight=base_weights)
    lgb_cogs = train_lgb(X_train, y_train_cogs, sample_weight=base_weights)
    p_lgb_rev = np.exp(predict_lgb(lgb_rev, X_test))
    p_lgb_cogs = np.exp(predict_lgb(lgb_cogs, X_test))

    q_boost = run_config["q_boost"]
    q_train = feat_train["Date"].apply(lambda value: pd.Timestamp(value).quarter).to_numpy()
    q_test = feat_test["Date"].apply(lambda value: pd.Timestamp(value).quarter).to_numpy()

    cat_rev = train_catboost(X_train, y_train_rev, sample_weight=base_weights)
    cat_cogs = train_catboost(X_train, y_train_cogs, sample_weight=base_weights)
    p_cat_rev = np.exp(predict_catboost(cat_rev, X_test))
    p_cat_cogs = np.exp(predict_catboost(cat_cogs, X_test))

    p_cat_spec_rev, p_cat_spec_cogs = _quarter_specialist_predictions(
        X_train,
        X_test,
        y_train_rev,
        y_train_cogs,
        base_weights,
        q_train,
        q_test,
        q_boost,
        model_type="catboost",
    )
    p_spec_rev, p_spec_cogs = _quarter_specialist_predictions(
        X_train,
        X_test,
        y_train_rev,
        y_train_cogs,
        base_weights,
        q_train,
        q_test,
        q_boost,
        model_type="lgb",
        lgb_best_iters=(lgb_rev["best_iter"], lgb_cogs["best_iter"]),
    )

    preds = {
        "p_ridge_rev": p_ridge_rev,
        "p_ridge_cogs": p_ridge_cogs,
        "p_lgb_rev": p_lgb_rev,
        "p_lgb_cogs": p_lgb_cogs,
        "p_cat_rev": p_cat_rev,
        "p_cat_cogs": p_cat_cogs,
        "p_spec_rev": p_spec_rev,
        "p_spec_cogs": p_spec_cogs,
        "p_cat_spec_rev": p_cat_spec_rev,
        "p_cat_spec_cogs": p_cat_spec_cogs,
    }

    if run_config.get("use_diversity_models", False):
        w_blend_recent = compute_sample_weights(sales["Date"], scheme="blend_recent")
        w_post2019 = compute_sample_weights(sales["Date"], scheme="post2019_focus")
        lgb_recent_rev = train_lgb(X_train, y_train_rev, sample_weight=w_blend_recent, n_rounds_override=lgb_rev["best_iter"])
        lgb_recent_cogs = train_lgb(X_train, y_train_cogs, sample_weight=w_blend_recent, n_rounds_override=lgb_cogs["best_iter"])
        cat_post_rev = train_catboost(X_train, y_train_rev, sample_weight=w_post2019)
        cat_post_cogs = train_catboost(X_train, y_train_cogs, sample_weight=w_post2019)
        preds.update(
            {
                "p_lgb_recent_rev": np.exp(predict_lgb(lgb_recent_rev, X_test)),
                "p_lgb_recent_cogs": np.exp(predict_lgb(lgb_recent_cogs, X_test)),
                "p_cat_post_rev": np.exp(predict_catboost(cat_post_rev, X_test)),
                "p_cat_post_cogs": np.exp(predict_catboost(cat_post_cogs, X_test)),
            }
        )

    raw_rev, raw_cog = make_raw_predictions(
        preds,
        alpha=run_config["alpha"],
        diversity_rev_weight=run_config.get("diversity_rev_weight", 0.0)
        if run_config.get("use_diversity_models", False)
        else 0.0,
        diversity_cog_weight=run_config.get("diversity_cog_weight", 0.0)
        if run_config.get("use_diversity_models", False)
        else 0.0,
    )
    submission = build_submission(test, raw_rev, raw_cog, cr=run_config["cr"], cc=run_config["cc"])
    out_path = Path(out_path) if out_path is not None else SUBMISSION_PATH
    submission.to_csv(out_path, index=False)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)
    pred_frame = pd.DataFrame({"Date": test["Date"], **preds})
    pred_frame.to_parquet(ARTIFACT_DIR / "v3_raw_predictions.parquet", index=False)
    test.to_parquet(ARTIFACT_DIR / "v3_test_dates.parquet", index=False)
    summary = summarize_submission(submission, label="selected_default")
    report = {
        "config": run_config,
        "cv": cv_results,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "submission": summary,
    }
    with (CONFIG_DIR / "cv_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    with (CONFIG_DIR / "v3_run_config.yaml").open("w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)
    return submission, cv_results, report


def build_submission_from_artifacts(
    out_path: Path | str | None = None,
    config: dict | None = None,
) -> Path:
    run_config = dict(RUN_CONFIG)
    config_path = CONFIG_DIR / "v3_run_config.yaml"
    if config_path.exists():
        with config_path.open() as f:
            stored = yaml.safe_load(f) or {}
        run_config.update(stored)
    if config:
        run_config.update(config)

    predictions = pd.read_parquet(ARTIFACT_DIR / "v3_raw_predictions.parquet")
    test = pd.read_parquet(ARTIFACT_DIR / "v3_test_dates.parquet")
    preds = predictions_frame_to_dict(predictions)
    raw_rev, raw_cog = make_raw_predictions(
        preds,
        alpha=run_config["alpha"],
        diversity_rev_weight=run_config.get("diversity_rev_weight", 0.0)
        if run_config.get("use_diversity_models", False)
        else 0.0,
        diversity_cog_weight=run_config.get("diversity_cog_weight", 0.0)
        if run_config.get("use_diversity_models", False)
        else 0.0,
    )
    submission = build_submission(test, raw_rev, raw_cog, cr=run_config["cr"], cc=run_config["cc"])
    path = Path(out_path) if out_path is not None else SUBMISSION_PATH
    submission.to_csv(path, index=False)
    return path


def _quarter_specialist_predictions(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_rev: np.ndarray,
    y_train_cogs: np.ndarray,
    base_weights: np.ndarray,
    q_train: np.ndarray,
    q_test: np.ndarray,
    q_boost: float,
    model_type: str,
    lgb_best_iters: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred_rev = np.zeros(len(X_test))
    pred_cogs = np.zeros(len(X_test))
    for quarter in [1, 2, 3, 4]:
        weights = base_weights.copy()
        weights[q_train == quarter] *= q_boost
        if model_type == "catboost":
            model_rev = train_catboost(X_train, y_train_rev, sample_weight=weights)
            model_cogs = train_catboost(X_train, y_train_cogs, sample_weight=weights)
            q_pred_rev = np.exp(predict_catboost(model_rev, X_test))
            q_pred_cogs = np.exp(predict_catboost(model_cogs, X_test))
        elif model_type == "lgb":
            if lgb_best_iters is None:
                raise ValueError("lgb_best_iters is required for LightGBM specialists")
            model_rev = train_lgb(X_train, y_train_rev, sample_weight=weights, n_rounds_override=lgb_best_iters[0])
            model_cogs = train_lgb(X_train, y_train_cogs, sample_weight=weights, n_rounds_override=lgb_best_iters[1])
            q_pred_rev = np.exp(predict_lgb(model_rev, X_test))
            q_pred_cogs = np.exp(predict_lgb(model_cogs, X_test))
        else:
            raise ValueError(f"Unknown specialist model type: {model_type}")
        mask = q_test == quarter
        pred_rev[mask] = q_pred_rev[mask]
        pred_cogs[mask] = q_pred_cogs[mask]
    return pred_rev, pred_cogs
