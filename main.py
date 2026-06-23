
import pandas as pd
from pathlib import Path

csv_folder = r"- Macros\AI-LaneDetector\lane_analysis"

SKIP_FIILES = ["per_project_states.csv", "per_session+state.csv"]

all_dfs = []

for csv_file.name in Path(csv_folder).glob("*.csv"):
    if csv_file.name in SKIP_FIILES:
        print(f"SKIPPING {csv_file.name}")
        continue

    df = pd.read_csv(csv_file, low_memory = False)
    if "filepath" not in df.columns or "Lane" not in df.columns:
        print(f"Skipping {csv_file.name} - missing FilePath or lane column")
        continue

    df["source_csv"] = csv_file.name
    all_dfs.append(df)
    print(f"{csv_file.name}: {len(df):, } rows")

combined = pd.concat(all_dfs, ignore_index = True)
print(f"\ntotal rows before dedup: {len(combined)}:,")

combined_deduped = combined.drop_duplicates(subset = ["Filepath"], keep = "last")
print(f"total rows after dedup: {len(combined_deduped):,}")
print(f"duplicates rows removed: {len(combined) - len(combined_deduped):,}")
print(f"\nLane distribution after dedup:")
print(combined_deduped["source_csv"].value_counts().toString())











