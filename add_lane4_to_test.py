import pandas as pd

test_df = pd.read_csv("Data/test.csv")
test_df["Lane"] = test_df["Lane"].astype(str)

full_df = pd.read_csv(
    r"J:\- Macros\AI-LaneDetector\lane_analysis\coordinates_report_merged_fix_folder1.csv",
    low_memory=False
)
full_df["Lane"] = full_df["Lane"].astype(str)

lane4_test_sample = full_df[full_df["Lane"] == "4"].sample(500, random_state=777)

test_df_updated = pd.concat([test_df, lane4_test_sample]).reset_index(drop=True)
test_df_updated.to_csv("Data/test.csv", index=False)
print(f"Updated test.csv: {len(test_df_updated):,} rows")
print(test_df_updated["Lane"].value_counts())