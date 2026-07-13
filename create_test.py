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

# extract segment name from filepath
df["Segment"] = df["Filepath"].apply(
    lambda x: str(x).split("Testing\\")[1].split("\\")[0] if "Testing\\" in str(x) else ""
)

test_segment = "250357 UHCC_25_LMD"
test_df = df[df["Segment"].str.contains(test_segment, na=False)].copy()
print(f"Test segment rows: {len(test_df):,}")
print(test_df["Lane"].value_counts().to_string())

test_df = test_df.drop(columns=["Segment"])
test_df.to_csv("Data/test.csv", index=False)
print("saved Data/test.csv")