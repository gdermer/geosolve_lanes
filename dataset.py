# load images from the CSV file and prepares them fro the mode;

import cv2

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from config import (LANE_CLASSES, IMG_SIZE, TRAIN_CSV, VAL_CSV, TEST_CSV)

TRAIN_TRANSFORMS = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p = 0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit= 20, val_shift_limit=20, p=0.3),
    A.GaussNoise(var_limit= (10.0,50.0), p=0.3),
    A.RandomShadow(shadow_roi=(0,0.4, 1,1), p=0.2),
    A.Blur(blur_limit=3, p=0.2),
    A.Normalize(mean = [0.485, 0.465, 0.406], std= [0.229, 0.224, 0.225]),
    ToTensorV2()
    ])


VAL_TRANSFORMS = A.Compose([
    A.Normalize(mean= [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
])

class LaneDataSET(Dataset):
    """ loads road imgaes and their lanes from CSV file """

def __init__(self, csv_path, transform, path_prefix="" ):
    """ constructor
    csv_path = path to train train, csv, val.csv, test.csv """
    self.Transform = transform
    self.path_prefix = path_prefix

    print(f"[Dataset] Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)    # read the file into the dataframe
    df = df[df["Lane"].isin(LANE_CLASSES)].copy()   # keep only rows with valid labels

    # keep only rows with no ignore column
    if "Ignore" in df.columns:
        df = df[df["Ignore"] != 1.0].copy() #removes rows marked as ignore
    #reset rows indexes after dropping the ignored ones
    self.df = df.reset_index(drop = True)

    print(f"[Dataset] {len(self.df):,} valid images loaded (dropped ignore)")

    # print class distribution so see if imbalanced
    print(f"[Dataset] Lane distibution:")
    for lane, count in self.df["Lane"].value_counta().items():
        pct = count/ len(self.df)*100
        print(f" {lane:>4}: {count:>10,} ({pct:.1f%})")





