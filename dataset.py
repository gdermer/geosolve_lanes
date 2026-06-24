# load images from the CSV file and prepares them fro the mode;

import cv2

import numpy as np
import pandas as pd
import self
import torch
from pydantic.experimental.pipeline import transform
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

class LaneDataset(Dataset):
    #loads road imgaes and their lanes from CSV file

    def __init__(self, csv_path, transform, path_prefix="" ):
    # constructor
    #csv_path = path to train train, csv, val.csv, test.csv
        self.transform = transform
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
        for lane, count in self.df["Lane"].value_counts().items():
            pct = count/ len(self.df)*100
            print(f" {lane:>4}: {count:>10,} ({pct:.1f})")


    def __len__(self):
        """ return how many images are in the dataset
        pytorch calls it to know when to stop iterating
        """
        return len(self.df)


    def __getitem__(self, idx):
    #return one image and its lable as position idx"""

        row = self.df.iloc[idx]    # get row by position number

        filepath = row["Filepath"]  # load the image
        filepath = str(filepath).replace("\\", "/")
        if self.path_prefix: # if a prefi is set
            filepath = filepath.replace("J:/", self.path_prefix)

        # load image
        img = cv2.imread(filepath)
        # return numpy array in RGB format returns non if not file has been found

        if img is None:
            print(f"[Dataset] WARNING : could non load {filepath}") #return black image instead of crashing
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype = np.uint8)   # array oif zeros

        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # resize img to 224, 224 (width, height)


        # apply transforms
        transformed = self.transform(image = img)    # return dictionary
        img_tensor = transformed["image"]    # extract the image
        # img_tensor is now a pyTorch tensor shape (3,224,224)

    # get the lable
        lane_code = row["Lane"]       # the lane column
        label = LANE_CLASSES[lane_code]

        label_tensor= torch.tensor(label, dtype = torch.long)   # convert integer to pytorch tensor 64 bit integer

    # get gps fetures:

        gps_features = self._get_gps_features(row)   #

        return img_tensor, gps_features, label_tensor       # returns everything the model needs

    def _get_gps_features(self, row):
     # extract GPS and road context features froma  csv row, these are the extra numbers we add alongside the image
     #return a tensor of 5 numbers [lane_forward, is_oneway, road_type_code, bearing sin, bearing cos]



        # lanes forward : how many lanes go in the van;'s direction
        # from OSM preprocessing (0 if not yet added to the csv)
        lanes_forward = float(row.get("lanes forward", 1))
        is_oneway = float(row.get("oneway", 0))  # is this a one way road?
        road_type_map= {
            "motorway": 3.0, "primary": 2.0, "secondary": 1.0, "residential" : 0.0, "unknown": 0.0}        # encode road type as a number
        road_type_str = str(row.get("road_type", "unknown"))       # return road type and unknown as default
        road_type = road_type_map.get(road_type_str, 0.0)

        bearing = float(row.get("bearing", 0))
        if bearing > 360:
            bearing = 0.0          # in case of bearing 1000 unknown


        bearing_rad = np.radians(bearing) # converting the angle to sin/cos

        bearing_sin = float(np.sin(bearing_rad))    # getting the sin value
        bearing_cos = float(np.cos(bearing_rad))
        features = torch.tensor([lanes_forward, is_oneway, road_type, bearing_sin, bearing_cos], dtype = torch.float32)
        return features


    # dataloader factory function

def get_train_loader(batch_size, num_workers =0, path_prefix = ""):
   # creates the traiing data loader, shuffles data randomly each epoch
    dataset = LaneDataset(csv_path = TRAIN_CSV, transform= TRAIN_TRANSFORMS, path_prefix= path_prefix)
    loader = DataLoader(
    dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True if num_workers > 0 else False,
        drop_last = True,)

    print(f"[DataLoader] Train : {len(dataset):,} images, "
          f"{len(loader):,} batches per epoch")
    return loader

def get_val_loader(batch_size, num_workers = 0, path_prefix = ""):      # creates the validation data loader
    dataset = LaneDataset(
        csv_path = VAL_CSV, transform = VAL_TRANSFORMS, path_prefix=path_prefix)

    loader = DataLoader(
        dataset, batch_size = batch_size, shuffle = False, num_workers= num_workers, pin_memory=True if num_workers>0 else False, drop_last= False)

    print (f"[DataLoader] val: {len(dataset):, } images, "
           f"{len(loader):,} batches")
    return loader

def get_test_loader(batch_size, num_workers = 0, path_prefix = ""):      # creates the dataset loader, used at the end for final evaluation
    dataset = LaneDataset(csv_path= TEST_CSV, transform= VAL_TRANSFORMS, path_prefix=path_prefix)
    loader = DataLoader(dataset, batch_size= batch_size, shuffle=False, num_workers= num_workers, drop_last=False,
                        )
    print(f"[DataLoader] Test: {len(dataset):,} images, "
          f"{len(loader):,} batches")
    return loader


#### run this file directly to verify everything works:

if __name__ == "__main__":
    print("testing dataset.py...")
    print("=" *40)

    loader = get_train_loader(batch_size = 4, num_workers = 0)
    images, gps, labels = next(iter(loader))

    print(f"\nBatch shapes:")
    print(f"    images:      {images.shape}")
    print(f"    gps features: {gps.shape}")
    print(f"    labels:        {labels.shape}")
    print(f" f\nLabel values: {labels.tolist()}")
    print(f"\nImage tensor states:")
    print(f" min:        {images.min():.3f}")
    print(f" max:        {images.max():.3f}")
    print(f" mean: {images.mean():.3f}")

    print(f"\nGPS features (forst image):")
    features_names = ["lanes_forward", "is_oneway", "road_type", "bearing_sin", "bearing_cos"]

    for name, val in zip(features_names, gps[0].tolist()): # zip pairs two lists together
        print(f"{name} : {val:.3f}")
    print("\ndataset.py isn working correctly :)")











































