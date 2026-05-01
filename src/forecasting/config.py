import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ["DATATHON_DATA_DIR"]) if "DATATHON_DATA_DIR" in os.environ else ROOT / "dataset"
ARTIFACT_DIR = ROOT / "artifacts"
CONFIG_DIR = ROOT / "configs"
SUBMISSION_PATH = ROOT / "submission.csv"

TRAIN_FILE = DATA_DIR / "sales.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-07-01"
HORIZON = 548

SEED = 1337

ARTIFACT_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)
