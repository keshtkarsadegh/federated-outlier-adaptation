"""Top-6 configuration selection per grid-search JSON, used across the results root."""

import json
import pandas as pd
from pathlib import Path

from federated_outlier_adaptation import config

def process_json_file(json_path: Path) -> pd.DataFrame | None:
    """
    Process one JSON file: pick best rows, compute weighted score,
    keep top 6 by score, sort ascending by local, save as CSV.
    """
    try:
        # Skip empty files
        if json_path.stat().st_size == 0:
            print(f"⚠️ Skipping empty file: {json_path}")
            return None

        with open(json_path, "r") as f:
            data = json.load(f)

        rows = []
        for method, tuples in data.items():
            if not tuples:
                continue
            last_five = tuples[-5:]
            best_row = max(last_five, key=lambda x: x[1])  # max by global
            rows.append([method, best_row[0], best_row[1]])

        if not rows:
            print(f"⚠️ No valid rows found in {json_path}")
            return None

        df = pd.DataFrame(rows, columns=["method", "clients", "global"])



        df["score"] = 4 * df["clients"] + 10 * (df["global"]) ** 2

        # Select top 6 by weighted score
        df = df.sort_values(by="score", ascending=False).head(6)

        # Sort the selected ones by local ascending
        df = df.sort_values(by="clients", ascending=True).reset_index(drop=True)

        # Save CSV with same name
        csv_path = json_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

        return df

    except Exception as e:
        print(f"❌ Failed to process {json_path}: {e}")
        return None


def top_6_configs(root_folder: str | Path | None = None):
    """
    Walk through root folder, process all JSON files,
    save corresponding CSVs next to them.

    ``root_folder`` defaults to the results root (``config.RESULTS_DIR``).
    """
    root = Path(root_folder) if root_folder else config.RESULTS_DIR
    json_files = list(root.rglob("*.json"))

    results = {}
    for jf in json_files:
        print(f"Processing {jf} ...")
        df = process_json_file(jf)
        if df is not None:
            results[str(jf)] = df

    return results


if __name__ == "__main__":
    top_6_configs()
