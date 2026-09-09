"""Lab 4 starter: audit dialect mix and record the implication in NOTES.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

DATA_PATH = Path("data/raw/bayan_feedback.csv")


def main():
    df = pd.read_csv(DATA_PATH)
    ar = df[df["lang"] == "ar"]

    counts = ar["dialect_region"].value_counts()
    pct = ar["dialect_region"].value_counts(normalize=True) * 100

    print(f"Arabic rows: {len(ar)} / {len(df)} total")
    print("\nRegion distribution (Arabic only):")
    for region in counts.index:
        print(f"  {region:<10} {counts[region]:>6}  ({pct[region]:5.1f}%)")

    gulf_pct = pct.get("Gulf", 0.0)
    print(
        f"\nImplication: {gulf_pct:.1f}% of Arabic feedback is Gulf dialect, not MSA. "
        "Evaluating (or fine-tuning) only on MSA text would silently miss quality "
        "problems on the majority of real Arabic traffic."
    )
    return counts.to_dict()


if __name__ == "__main__":
    main()
