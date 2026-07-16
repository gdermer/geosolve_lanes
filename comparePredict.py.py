import pandas as pd 


df = pd.read_csv("Data/ufld_compare_sample_with_AI_v3.csv", low_memory=False)

# only compare rows with valid ground truth in our 4 classes
valid = df[df["Lane"].astype(str).isin(["1", "2", "3", "SK1"])].copy()
valid["Lane"] = valid["Lane"].astype(str)
valid["Lane_AI"] = valid["Lane_AI"].astype(str)

print(f"Comparable rows: {len(valid):,}")

# our model accuracy
correct = (valid["Lane"] == valid["Lane_AI"]).sum()
total = len(valid)
print(f"\nOur model overall accuracy: {correct/total*100:.2f}%")

# per class
print("\nPer class accuracy (our model):")
for lane in ["1", "2", "3", "SK1"]:
    subset = valid[valid["Lane"] == lane]
    if len(subset) == 0:
        continue
    correct_lane = (subset["Lane_AI"] == lane).sum()
    print(f"  Lane {lane}: {correct_lane}/{len(subset)} = {correct_lane/len(subset)*100:.2f}%")

# compare with their vision (LaneUFLD column)
if "LaneUFLD" in valid.columns:
    valid_ufld = valid.dropna(subset=["LaneUFLD"]).copy()
    valid_ufld["LaneUFLD"] = valid_ufld["LaneUFLD"].astype(str)
    correct_ufld = (valid_ufld["Lane"] == valid_ufld["LaneUFLD"]).sum()
    print(f"\nTheir UFLD accuracy on same sample: {correct_ufld/len(valid_ufld)*100:.2f}%")
    print(f"(based on {len(valid_ufld):,} rows with UFLD predictions)")

    print("\nPer class accuracy (their UFLD):")
    for lane in ["1", "2", "3", "SK1"]:
        subset = valid_ufld[valid_ufld["Lane"] == lane]
        if len(subset) == 0:
            continue
        correct_lane = (subset["LaneUFLD"] == lane).sum()
        print(f"  Lane {lane}: {correct_lane}/{len(subset)} = {correct_lane/len(subset)*100:.2f}%")