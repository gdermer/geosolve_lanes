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
    """ loads road imgaes and their lanes from CSV file "
