"""Lab 3 starter: dataset construction and split integrity."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset(data_path: Path = DATA_PATH) -> dict:
    """Load the Bayan feedback dataset, partitioned by its supplied split column.

    The split is already grouped by citizen_group_id upstream (see
    data/DATA_DICTIONARY.md: "The supplied split is 70/20/10 and deterministic.
    The final test split is frozen."), so this loads and filters rather than
    re-shuffling — re-splitting would risk breaking the frozen test contract.
    """
    df = pd.read_csv(data_path)
    return {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "validation": df[df["split"] == "validation"].reset_index(drop=True),
        "test": df[df["split"] == "test"].reset_index(drop=True),
    }
