from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forecasting.config import ARTIFACT_DIR
from forecasting.ensemble import RUN_CONFIG, build_submission, candidate_tag, make_raw_predictions, predictions_frame_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Write calibrated v3 candidate submissions from trained artifacts.")
    parser.add_argument("--out-dir", default="artifacts/candidates")
    args = parser.parse_args()

    predictions = pd.read_parquet(ARTIFACT_DIR / "v3_raw_predictions.parquet")
    test = pd.read_parquet(ARTIFACT_DIR / "v3_test_dates.parquet")
    preds = predictions_frame_to_dict(predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for candidate in RUN_CONFIG["candidate_grid"]:
        config = dict(RUN_CONFIG)
        config.update(candidate)
        raw_rev, raw_cog = make_raw_predictions(preds, alpha=config["alpha"])
        submission = build_submission(test, raw_rev, raw_cog, cr=config["cr"], cc=config["cc"])
        path = out_dir / f"submission_v3_{candidate_tag(config)}.csv"
        submission.to_csv(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
