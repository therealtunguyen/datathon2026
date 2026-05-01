from __future__ import annotations

import argparse
from pathlib import Path

from forecasting.pipeline import train_and_predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v3 Datathon forecasting pipeline.")
    parser.add_argument("--data-dir", default=None, help="Dataset directory. Defaults to DATATHON_DATA_DIR or dataset/.")
    parser.add_argument("--out", default=None, help="Submission path. Defaults to submission.csv.")
    args = parser.parse_args()

    submission, cv_results, report = train_and_predict(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        out_path=Path(args.out) if args.out else None,
    )
    print(f"wrote {args.out or 'submission.csv'} ({len(submission)} rows)")
    print(f"features: {report['feature_count']}")
    for fold, metrics in cv_results.items():
        print(f"{fold}: MAE={metrics['mae']:,.0f}, n={metrics['n_val']}, best_iter={metrics['best_iter']}")


if __name__ == "__main__":
    main()
