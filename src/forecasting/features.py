from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)

TET_DATES = {
    2012: "2012-01-23",
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
}
TET_TS = {year: pd.Timestamp(value) for year, value in TET_DATES.items()}

PROMO_SCHEDULE_DEFAULT = [
    ("spring_sale", 3, 18, 30, 12, True),
    ("mid_year", 6, 23, 29, 18, True),
    ("fall_launch", 8, 30, 32, 10, True),
    ("year_end", 11, 18, 45, 20, True),
    ("urban_blowout", 7, 30, 33, None, "odd"),
    ("rural_special", 1, 30, 30, 15, "odd"),
]

VN_FIXED_HOLIDAYS = [
    (1, 1, "new_year"),
    (2, 14, "valentine"),
    (3, 8, "womens_day"),
    (4, 30, "reunification"),
    (5, 1, "labor_day"),
    (9, 2, "national_day"),
    (10, 20, "vn_womens_day"),
    (11, 11, "dd_1111"),
    (12, 12, "dd_1212"),
    (12, 24, "christmas_eve"),
    (12, 25, "christmas"),
]


def nearest_tet_diff(dt: pd.Timestamp) -> int:
    candidates = []
    for offset in [-1, 0, 1]:
        year = dt.year + offset
        if year in TET_TS:
            diff = (dt - TET_TS[year]).days
            if abs(diff) <= 60:
                candidates.append(diff)
    return min(candidates, key=abs) if candidates else 999


def build_business_signals(data: dict[str, pd.DataFrame | None]) -> dict[str, pd.DataFrame | None]:
    """Aggregate operational tables into stable month-level signals.

    The v3 notebook inspected traffic, inventory, source mix, and order volume. Only the
    order-volume signal is kept as a model feature because the other signals were nearly flat.
    """

    promotions = data["promotions"]
    web_traffic = data["web_traffic"].copy()
    inventory = data["inventory"].copy()
    orders = data["orders"].copy()

    promo_schedule = []
    for _, row in promotions.iterrows():
        promo_schedule.append(
            {
                "name": row.get("promo_type", "unknown"),
                "start_month": row["start_date"].month,
                "start_day": row["start_date"].day,
                "duration": (row["end_date"] - row["start_date"]).days,
                "discount": row.get("discount_value", 0),
                "category": row.get("applicable_category", "all"),
            }
        )

    web_traffic["month"] = web_traffic["date"].dt.month
    if "conversion_rate" not in web_traffic.columns:
        web_traffic["sessions"] = web_traffic["sessions"].replace(0, np.nan)
        web_traffic["conversion_rate"] = 1 - web_traffic["bounce_rate"]
    monthly_conversion = (
        web_traffic.groupby("month")
        .agg(
            avg_sessions=("sessions", "mean"),
            avg_bounce_rate=("bounce_rate", "mean"),
            avg_conversion_rate=("conversion_rate", "mean"),
        )
        .reset_index()
    )

    inventory["month"] = inventory["snapshot_date"].dt.month
    stockout_by_month = (
        inventory.groupby("month")
        .agg(
            stockout_rate=("stockout_flag", "mean"),
            overstock_rate=("overstock_flag", "mean"),
            avg_stock=("stock_on_hand", "mean"),
        )
        .reset_index()
    )

    orders["month"] = orders["order_date"].dt.month
    source_mix_pct = None
    if "order_source" in orders.columns:
        source_mix = orders.groupby(["month", "order_source"]).size().unstack(fill_value=0)
        source_mix_pct = source_mix.div(source_mix.sum(axis=1), axis=0)

    orders["date"] = orders["order_date"].dt.date
    daily_orders = orders.groupby("date").size().reset_index(name="order_count")
    daily_orders["date"] = pd.to_datetime(daily_orders["date"])
    daily_orders["month"] = daily_orders["date"].dt.month
    monthly_order_pattern = daily_orders.groupby("month")["order_count"].mean().reset_index()

    return {
        "promo_schedule": promo_schedule,
        "monthly_conversion": monthly_conversion,
        "stockout_by_month": stockout_by_month,
        "source_mix_pct": source_mix_pct,
        "monthly_order_pattern": monthly_order_pattern,
    }


def build_features(
    dates: pd.DatetimeIndex | pd.Series,
    business_signals: dict[str, pd.DataFrame | None] | None = None,
    promo_schedule: list[tuple[str, int, int, int, int | None, bool | str]] | None = None,
) -> pd.DataFrame:
    if promo_schedule is None:
        promo_schedule = PROMO_SCHEDULE_DEFAULT

    df = pd.DataFrame({"Date": pd.to_datetime(dates)})
    d = df["Date"]

    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dow"] = d.dt.dayofweek
    df["doy"] = d.dt.dayofyear
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_weekday"] = (df["dow"] < 5).astype(int)
    df["days_in_month"] = d.dt.days_in_month
    df["days_to_eom"] = df["days_in_month"] - df["day"]
    df["days_from_som"] = df["day"] - 1
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)

    for k in [1, 2, 3]:
        df[f"is_last{k}"] = (df["days_to_eom"] <= k - 1).astype(int)
        df[f"is_first{k}"] = (df["days_from_som"] <= k - 1).astype(int)
    df["is_last_5d"] = (df["days_to_eom"] <= 4).astype(int)
    df["is_first_5d"] = (df["days_from_som"] <= 4).astype(int)
    df["is_last_10d"] = (df["days_to_eom"] <= 9).astype(int)
    df["is_mid_month"] = ((df["day"] >= 10) & (df["day"] <= 20)).astype(int)

    anchor = pd.Timestamp("2020-01-01")
    df["t_days"] = (d - anchor).dt.days
    df["t_years"] = df["t_days"] / 365.25
    df["regime_pre2019"] = (df["year"] <= 2018).astype(int)
    df["regime_2019"] = (df["year"] == 2019).astype(int)
    df["regime_post2019"] = (df["year"] >= 2020).astype(int)
    df["trend_post2019"] = np.where(df["year"] >= 2020, df["t_days"], 0)
    df["trend_pre2019"] = np.where(df["year"] < 2019, df["t_days"], 0)

    tau = 2 * np.pi
    for k in range(1, 6):
        df[f"sin_y{k}"] = np.sin(tau * k * df["doy"] / 365.25)
        df[f"cos_y{k}"] = np.cos(tau * k * df["doy"] / 365.25)
    for k in range(1, 3):
        df[f"sin_w{k}"] = np.sin(tau * k * df["dow"] / 7.0)
        df[f"cos_w{k}"] = np.cos(tau * k * df["dow"] / 7.0)
        df[f"sin_m{k}"] = np.sin(tau * k * (df["day"] - 1) / df["days_in_month"])
        df[f"cos_m{k}"] = np.cos(tau * k * (df["day"] - 1) / df["days_in_month"])

    diffs = np.array([nearest_tet_diff(dt) for dt in d])
    df["tet_days_diff"] = diffs
    df["tet_in_3"] = (np.abs(diffs) <= 3).astype(int)
    df["tet_in_7"] = (np.abs(diffs) <= 7).astype(int)
    df["tet_in_14"] = (np.abs(diffs) <= 14).astype(int)
    df["tet_before_7"] = ((diffs >= -7) & (diffs < 0)).astype(int)
    df["tet_after_7"] = ((diffs > 0) & (diffs <= 7)).astype(int)
    df["tet_after_14"] = ((diffs > 7) & (diffs <= 21)).astype(int)
    df["tet_on"] = (diffs == 0).astype(int)
    df["tet_proximity"] = np.exp(-0.5 * (diffs / 10) ** 2)

    for month, day, name in VN_FIXED_HOLIDAYS:
        df[f"hol_{name}"] = ((df["month"] == month) & (df["day"] == day)).astype(int)

    df["hol_black_friday"] = [int(_is_black_friday(dt)) for dt in d]
    df["hol_double_day"] = (
        ((df["month"] == 11) & (df["day"] == 11))
        | ((df["month"] == 12) & (df["day"] == 12))
    ).astype(int)

    df["is_odd_year"] = (df["year"] % 2).astype(int)
    df["is_even_year"] = 1 - df["is_odd_year"]
    for quarter in range(1, 5):
        df[f"is_q{quarter}"] = (df["quarter"] == quarter).astype(int)
    df["q3_odd_year"] = df["is_q3"] * df["is_odd_year"]
    df["q3_even_year"] = df["is_q3"] * df["is_even_year"]

    for month_name, month in [("jan", 1), ("mar", 3), ("jun", 6), ("aug", 8), ("nov", 11), ("dec", 12)]:
        df[f"{month_name}_odd"] = ((df["month"] == month) & df["is_odd_year"].astype(bool)).astype(int)
        df[f"{month_name}_even"] = ((df["month"] == month) & df["is_even_year"].astype(bool)).astype(int)
    for month_name, month in [("mar", 3), ("jun", 6), ("aug", 8), ("dec", 12)]:
        df[f"eom_{month_name}"] = df["is_last_5d"] * (df["month"] == month).astype(int)
    for quarter in [1, 2, 4]:
        df[f"q{quarter}_odd"] = df[f"is_q{quarter}"] * df["is_odd_year"]
        df[f"q{quarter}_even"] = df[f"is_q{quarter}"] * df["is_even_year"]

    all_years = range(df["year"].min() - 1, df["year"].max() + 2)
    for name, start_month, start_day, duration, discount, recur in promo_schedule:
        in_promo = np.zeros(len(df), dtype=int)
        since_arr = np.full(len(df), -1.0)
        until_arr = np.full(len(df), -1.0)
        disc_arr = np.zeros(len(df))
        for year in all_years:
            if recur == "odd" and year % 2 == 0:
                continue
            if recur == "even" and year % 2 != 0:
                continue
            try:
                start = pd.Timestamp(year=year, month=start_month, day=start_day)
            except ValueError:
                continue
            end = start + pd.Timedelta(days=duration)
            mask = (d >= start) & (d <= end)
            in_promo[mask] = 1
            since_arr[mask] = (d[mask] - start).dt.days
            until_arr[mask] = (end - d[mask]).dt.days
            disc_arr[mask] = discount or 0
        df[f"promo_{name}"] = in_promo
        df[f"promo_{name}_since"] = since_arr
        df[f"promo_{name}_until"] = until_arr
        df[f"promo_{name}_disc"] = disc_arr

    promo_cols = [
        col
        for col in df.columns
        if col.startswith("promo_") and not any(part in col for part in ["since", "until", "disc"])
    ]
    df["total_active_promos"] = df[promo_cols].sum(axis=1)
    df["has_any_promo"] = (df["total_active_promos"] > 0).astype(int)

    if business_signals is not None and business_signals.get("monthly_order_pattern") is not None:
        order_signal = business_signals["monthly_order_pattern"].set_index("month")
        mean_orders = order_signal["order_count"].mean()
        if mean_orders and np.isfinite(mean_orders):
            df["order_volume_signal"] = df["month"].map(order_signal["order_count"] / mean_orders).fillna(1.0)

    for month in range(1, 13):
        df[f"month_{month}"] = (df["month"] == month).astype(int)
    for quarter in range(1, 5):
        df[f"quarter_{quarter}"] = (df["quarter"] == quarter).astype(int)
    for dow in range(7):
        df[f"dow_{dow}"] = (df["dow"] == dow).astype(int)

    df["is_tet_season"] = df["month"].isin([1, 2]).astype(int)
    df["is_spring_peak"] = df["month"].isin([3, 4]).astype(int)
    df["is_summer_low"] = df["month"].isin([7, 8, 9]).astype(int)
    df["is_year_end_peak"] = df["month"].isin([11, 12]).astype(int)

    return df.drop(columns=["Date"])


def build_train_test_features(
    sales: pd.DataFrame,
    test: pd.DataFrame,
    business_signals: dict[str, pd.DataFrame | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    all_dates = pd.concat([sales["Date"], test["Date"]], ignore_index=True)
    features = build_features(pd.DatetimeIndex(all_dates), business_signals=business_signals)
    features["Date"] = all_dates.values
    n_train = len(sales)
    train_features = features.iloc[:n_train].reset_index(drop=True)
    test_features = features.iloc[n_train:].reset_index(drop=True)
    feature_cols = [col for col in features.columns if col != "Date"]
    return train_features, test_features, feature_cols


def compute_sample_weights(dates: pd.Series, scheme: str = "high_era") -> np.ndarray:
    years = dates.dt.year.values
    weights = np.ones(len(dates))
    if scheme == "high_era":
        weights = np.full(len(dates), 0.01)
        weights[(years >= 2014) & (years <= 2018)] = 1.0
    elif scheme == "blend_recent":
        weights = np.full(len(dates), 0.01)
        weights[(years >= 2014) & (years <= 2018)] = 1.0
        weights[(years >= 2020) & (years <= 2022)] = 0.8
    elif scheme == "post2019_focus":
        weights = np.full(len(dates), 0.01)
        weights[(years >= 2014) & (years <= 2018)] = 0.5
        weights[(years >= 2020) & (years <= 2022)] = 1.0
    return weights


def _is_black_friday(dt: pd.Timestamp) -> bool:
    if dt.month != 11:
        return False
    last = pd.Timestamp(year=dt.year, month=11, day=30)
    last_friday = last - pd.Timedelta(days=(last.dayofweek - 4) % 7)
    return dt == last_friday
