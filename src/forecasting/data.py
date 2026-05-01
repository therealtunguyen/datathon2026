from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecasting.config import DATA_DIR, SAMPLE_SUBMISSION, TRAIN_FILE


def load_sales() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_FILE, parse_dates=["Date"])
    df.columns = ["Date", "Revenue", "COGS"]
    df = df.sort_values("Date").reset_index(drop=True)
    validate_continuity(df.set_index("Date"))
    return df


def validate_continuity(df: pd.DataFrame) -> None:
    expected = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = expected.difference(df.index)
    if len(missing) > 0:
        raise ValueError(
            f"Gaps in sales series: {len(missing)} missing days, "
            f"first 5: {list(missing[:5])}"
        )


def _read_csv(name: str, data_dir: Path, **kwargs) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    return pd.read_csv(path, **kwargs)


def load_competition_data(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame | None]:
    """Load the files used by the v3 solution notebook."""

    root = Path(data_dir) if data_dir is not None else DATA_DIR
    sales = _read_csv("sales.csv", root, parse_dates=["Date"])
    sales.columns = ["Date", "Revenue", "COGS"]
    sales = sales.sort_values("Date").reset_index(drop=True)
    validate_continuity(sales.set_index("Date"))

    customers_path = root / "customers.csv"
    customers = (
        pd.read_csv(customers_path, parse_dates=["signup_date"])
        if customers_path.exists()
        else None
    )

    return {
        "sales": sales,
        "test": pd.read_csv(SAMPLE_SUBMISSION if data_dir is None else root / "sample_submission.csv", parse_dates=["Date"]),
        "promotions": _read_csv("promotions.csv", root, parse_dates=["start_date", "end_date"]),
        "web_traffic": _read_csv("web_traffic.csv", root, parse_dates=["date"]),
        "inventory": _read_csv("inventory.csv", root, parse_dates=["snapshot_date"]),
        "orders": _read_csv("orders.csv", root, parse_dates=["order_date"]),
        "customers": customers,
    }
