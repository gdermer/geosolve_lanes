import pandas as pd
import os
from pathlib import Path


csv_folder = r"J:\- Macros\AI-LaneDetector\lane_analysis"

all_dfs = []
for csv_file in Path(csv_folder).glob("*.csv"):
    df = pd.read_csv(csv_file)
    df["source_csv"] = csv_file.name   # track which file each row came from
    all_dfs.append(df)
    print(f"{csv_file.name}: {len(df):,} rows")

# Combine all
combined = pd.concat(all_dfs, ignore_index=True)

print(f"\nTotal rows across all jobs: {len(combined):,}")
print(f"\nAll Lane codes found:")
print(combined["Lane"].value_counts().to_string())
print(f"\nAll segments found:")
print(combined["SourceFolder"].value_counts().to_string())

