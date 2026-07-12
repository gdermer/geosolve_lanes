import pandas as pd

print("Loading new CSV...")
df = pd.read_csv(
    r"J:\- Macros\AI-LaneDetector\lane_analysis\coordinates_report_merged_fix_folder1.csv",
    low_memory=False
)
print(f"Total rows: {len(df):,}")

valid_lanes = ["1", "2", "3", "SK1"]
df = df[df["Lane"].astype(str).isin(valid_lanes)].copy()
df["Lane"] = df["Lane"].astype(str)
print(f"After filtering: {len(df):,}")

lane1 = df[df["Lane"] == "1"].sample(317714, random_state=42)
lane2 = df[df["Lane"] == "2"].sample(105904, random_state=42)
lane3 = df[df["Lane"] == "3"].sample(15501,  random_state=42)
sk1   = df[df["Lane"] == "SK1"].sample(5919,  random_state=42)

balanced = pd.concat([lane1, lane2, lane3, sk1]).sample(frac=1, random_state=42)
balanced = balanced.reset_index(drop=True)

print(f"\nBalanced dataset:")
print(f"Total: {len(balanced):,}")
print(balanced["Lane"].value_counts().to_string())

chunk_size = len(balanced) // 10
from pathlib import Path
output_dir = Path("Data/chunks_v2")
output_dir.mkdir(exist_ok=True)

for i in range(10):
    start = i * chunk_size
    end   = start + chunk_size if i < 9 else len(balanced)
    chunk = balanced.iloc[start:end].copy()
    
    output_path = output_dir / f"train_chunk_v2_{i+1:02d}.csv"
    chunk.to_csv(output_path, index=False)
    print(f"Chunk {i+1:02d}: {len(chunk):,} rows → {output_path}")

print("\nDone!")