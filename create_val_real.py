import pandas as pd

print("Loading original CSV...")
df = pd.read_csv(
    r"J:\- Macros\AI-LaneDetector\lane_analysis\coordinates_report_merged_fix_folder1.csv",
    low_memory=False
)
print(f"Total rows: {len(df):,}")

df["Lane"] = df["Lane"].astype(str)
valid_lanes = ["1", "2", "3", "SK1"]
df = df[df["Lane"].isin(valid_lanes)].copy()

targets = {"1": 2400, "2": 90, "SK1": 5, "3": 5}

samples = []
for lane, n in targets.items():
    subset = df[df["Lane"] == lane]
    available = min(len(subset), n)
    samples.append(subset.sample(available, random_state=99))

val_real = pd.concat(samples).reset_index(drop=True)
print(f"\nReal distribution val set: {len(val_real):,}")
print(val_real["Lane"].value_counts().to_string())

val_real.to_csv("Data/val_real.csv", index=False)
print("saved Data/val_real.csv")