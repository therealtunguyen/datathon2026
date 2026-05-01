from __future__ import annotations

import argparse
from pathlib import Path

from forecasting.pipeline import build_submission_from_artifacts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()
    path = build_submission_from_artifacts(Path(args.out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
