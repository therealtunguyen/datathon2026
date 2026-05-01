from __future__ import annotations

import numpy as np


def train_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 3.0):
    from sklearn.linear_model import Ridge

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    model = Ridge(alpha=alpha, random_state=42, max_iter=5000)
    model.fit((X_train - mu) / sigma, y_train)
    return model, (mu, sigma)


def predict_ridge(model, X_test: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mu, sigma = stats
    return model.predict((X_test - mu) / sigma)


def train_lgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    params: dict | None = None,
    n_rounds_override: int | None = None,
) -> dict:
    import lightgbm as lgb

    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "lambda_l1": 0.1,
        "seed": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    if params:
        lgb_params.update(params)

    if n_rounds_override is None:
        n_val = min(180, max(1, int(len(X_train) * 0.1)))
        fit_mask = np.ones(len(X_train), dtype=bool)
        fit_mask[-n_val:] = False
        val_mask = ~fit_mask
        dtrain = lgb.Dataset(
            X_train[fit_mask],
            y_train[fit_mask],
            weight=sample_weight[fit_mask] if sample_weight is not None else None,
        )
        dval = lgb.Dataset(
            X_train[val_mask],
            y_train[val_mask],
            weight=sample_weight[val_mask] if sample_weight is not None else None,
        )
        booster_es = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)],
        )
        best_iter = booster_es.best_iteration
    else:
        best_iter = n_rounds_override

    dtrain_full = lgb.Dataset(X_train, y_train, weight=sample_weight)
    booster_final = lgb.train(
        lgb_params,
        dtrain_full,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(0)],
    )
    return {"model": booster_final, "best_iter": best_iter}


def predict_lgb(lgb_result: dict, X_test: np.ndarray) -> np.ndarray:
    return lgb_result["model"].predict(X_test)


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    iterations: int = 2000,
):
    from catboost import CatBoostRegressor

    n_val = min(180, max(1, int(len(X_train) * 0.1)))
    X_tr, X_vl = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_vl = y_train[:-n_val], y_train[-n_val:]
    w_tr = sample_weight[:-n_val] if sample_weight is not None else None
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=300,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_vl, y_vl), verbose=False)
    best_iter = max(1, model.best_iteration_ or iterations)

    final_model = CatBoostRegressor(
        iterations=best_iter,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=0,
    )
    final_model.fit(X_train, y_train, sample_weight=sample_weight)
    return final_model


def predict_catboost(model, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test)
