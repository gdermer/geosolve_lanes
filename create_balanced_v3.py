import pandas as pd

print("Loading original CSV...")
df = pd.read_csv(
    r"J:\- Macros\AI-LaneDetector\lane_analysis\coordinates_report_merged_fix_folder1.csv",
    low_memory=False
)
print(f"Total rows: {len(df):,}")

df["Lane"] = df["Lane"].astype(str)

# Bucket A only — position-based lanes
# excludes turning lanes (TK1, TK2, TM1, TM2, TM3, SK2, SK3, BK4, SM1)
valid_lanes = ["1", "2", "3", "4", "SK1"]
df = df[df["Lane"].isin(valid_lanes)].copy()
print(f"\nAfter filtering to Bucket A (1,2,3,4,SK1): {len(df):,}")
print(df["Lane"].value_counts().to_string())

# check how many of each class actually exist
counts = df["Lane"].value_counts()
print(f"\nAvailable per class:")
for lane in valid_lanes:
    n = counts.get(lane, 0)
    print(f"  Lane {lane}: {n:,}")

# ---- Sampling strategy ----
# Use ALL examples of the rarest classes (3, 4, SK1)
# Cap Lane 1 and Lane 2 at a reasonable multiple to keep them
# dominant-but-not-overwhelming, matching our successful chunk_v2 approach

lane3_count = counts.get("3", 0)
lane4_count = counts.get("4", 0)
sk1_count   = counts.get("SK1", 0)

# take everything for the rarest classes
lane3 = df[df["Lane"] == "3"]
lane4 = df[df["Lane"] == "4"]
sk1   = df[df["Lane"] == "SK1"]

# Lane 2: keep same proportion as our successful chunks_v2 (10% of full)
lane2_target = min(counts.get("2", 0), 105904)
lane2 = df[df["Lane"] == "2"].sample(lane2_target, random_state=42)

# Lane 1: keep it dominant but not overwhelming (roughly 3x Lane 2 size)
lane1_target = min(counts.get("1", 0), lane2_target * 3)
lane1 = df[df["Lane"] == "1"].sample(lane1_target, random_state=42)

balanced = pd.concat([lane1, lane2, lane3, lane4, sk1]).sample(frac=1, random_state=42)
balanced = balanced.reset_index(drop=True)

print(f"\nBalanced dataset (Bucket A):")
print(f"Total: {len(balanced):,}")
print(balanced["Lane"].value_counts().to_string())

# ---- Split into chunks for nightly training ----
chunk_size = len(balanced) // 10
from pathlib import Path
output_dir = Path("Data/chunks_v3")
output_dir.mkdir(exist_ok=True)

for i in range(10):
    start = i * chunk_size
    end   = start + chunk_size if i < 9 else len(balanced)
    chunk = balanced.iloc[start:end].copy()

    output_path = output_dir / f"train_chunk_v3_{i+1:02d}.csv"
    chunk.to_csv(output_path, index=False)
    print(f"Chunk {i+1:02d}: {len(chunk):,} rows → {output_path}")
    print(f"  {chunk['Lane'].value_counts().to_string()}")

print("\nDone! All chunks saved to Data/chunks_v3/")