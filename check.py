# paste in a new Python file or PyCharm console
import pandas as pd
import os

df = pd.read_csv("Data/train_small.csv")

# check size of first 10 images
total_size = 0
for filepath in df["Filepath"].head(10):
    filepath = str(filepath).replace("\\", "/")
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        total_size += size
        print(f"{size/1024/1024:.1f} MB — {filepath}")

avg_size = total_size / 10
estimate_50k = avg_size * 50000 / 1024 / 1024 / 1024
print(f"\nAverage image size: {avg_size/1024/1024:.1f} MB")
print(f"Estimated 50,000 images: {estimate_50k:.1f} GB")
print(f"Estimated 229,172 images: {avg_size*229172/1024/1024/1024:.1f} GB")