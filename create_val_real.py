import pandas as pd

print("Loading val.csv...")
df = pd.read_csv("Data/val.csv", low_memory=False)
print(f"Total rows: {len(df):,}")
print(df["Lane"].value_counts().to_string())

# stratified sample matching REAL distribution
targets = {"1": 2400, "2": 90, "SK1": 5, "3": 5}

samples = []
for lane, n in targets.items():
    subset = df[df["Lane"] == lane]
    available = min(len(subset), n)
    samples.append(subset.sample(available, random_state=42))

val_real = pd.concat(samples).reset_index(drop=True)
print(f"\nReal distribution val set: {len(val_real):,}")
print(val_real["Lane"].value_counts().to_string())

val_real.to_csv("Data/val_real.csv", index=False)
print("saved Data/val_real.csv")