from __future__ import annotations

import json

from forecasting.config import CONFIG_DIR


def main() -> None:
    with (CONFIG_DIR / "cv_report.json").open() as f:
        report = json.load(f)

    print("| fold        |        MAE | n_val | best_iter |")
    print("|-------------|-----------:|------:|----------:|")
    for name, metrics in report["cv"].items():
        print(
            f"| {name:<11} | {metrics['mae']:>10.2f} | "
            f"{metrics['n_val']:>5} | {metrics['best_iter']:>9} |"
        )
    print()
    print("Submission sanity:")
    for key, value in report["submission"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
