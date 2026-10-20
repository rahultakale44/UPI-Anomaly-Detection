from pathlib import Path
import pandas as pd

CANDIDATE_FILES = [
    Path("data/raw/PS_20174392719_152120.csv"),
    Path("data/raw/paysim.csv"),
    Path("data/raw/transactions.csv"),
]

def locate_csv() -> Path:
    for path in CANDIDATE_FILES:
        if path.exists():
            return path
    csvs = list(Path("data/raw").glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        "No CSV found in data/raw/. Download PaySim from the Kaggle link in README.md "
        "and place the extracted CSV in data/raw/."
    )

def main():
    path = locate_csv()
    print(f"Dataset: {path} ({path.stat().st_size / (1024**2):.1f} MiB)")
    df = pd.read_csv(path, nrows=100_000)
    print(f"Preview rows: {len(df):,}")
    print("Columns:", list(df.columns))
    print("\nMissing values (preview):")
    print(df.isna().sum().sort_values(ascending=False).head(10).to_string())
    if "isFraud" in df:
        print("\nLabel distribution (preview):")
        print(df["isFraud"].value_counts(dropna=False).to_string())
        print("NOTE: labels are for evaluation only; do not train the unsupervised detector on them.")
    print("\nPreview:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
