import pandas as pd

df = pd.read_csv(r"J:\geosolve_training\ufld\ufld_compare.csv", low_memory=False)
print(f"Total rows: {len(df):,}")

samples = []
for lane_value in df["Lane"].unique():
    subset = df[df["Lane"] == lane_value]
    n = min(len(subset), max(1, int(5000 * len(subset) / len(df))))
    samples.append(subset.sample(n, random_state=42))

sample = pd.concat(samples).reset_index(drop=True)
print(f"Sample size: {len(sample):,}")
print(sample["Lane"].value_counts())

sample.to_csv("Data/ufld_compare_sample.csv", index=False)
print("saved Data/ufld_compare_sample.csv")