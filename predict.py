import torch
import pandas as pd
import numpy as np
import cv2
import argparse
import time
from pathlib import Path

from sympy import true
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import( IMG_SIZE, IDX_TO_LANE, CHECKPOINT_DIR,)
from model import load_trained_model


class PredictionDataset(Dataset):

    # loads images from a survey csv for prediction

    def __init__(self, df, path_prefix=""):
        # df: panadas dataFrame with filepath column
# path_prefix = optional path replacement

        self.df = df.reset_index(drop = True)
        self.path_prefix = path_prefix
        self.transform = A.Compose([A.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2(),])


    def __len__(self):
        return len(self.df)


    def __getitem__(self,idx):
        row = self.df.iloc[idx]

        filepath = str(row["FIlepath"]).replace("\\","/")
        if self.path_prefix:
            filepath = filepath.replace("J:/", self.path_prefix)

        img = cv2.imread(filepath)

        if img is None:


            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype = np.uint8)
            valid = False

        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            valid = True

        transformed = self.transform(image = img)
        img_tensor = transformed(["image"])


        # GPS features:

        bearing = float(row.get("Bearing", 0))
        if bearing>360:
            bearing = 0.0

        bearing_rad = np.radians(bearing)

        gps_features = torch.tensor([float(row.get("lanes_forward", 1)),
                                     float(row.get("oneway", 0)),
                                     0.0,
                                     float(np.sin(bearing_rad)),
                                     float(np.cos(bearing_rad)),
                                     ], dtype = torch.float32)

        return img_tensor, gps_features, idx, valid

    # main prediction function:

def predict( input_csv, output_csv, checkpoint_path = None, threshold = 0.90, batch_size = 32, path_prefix = "", device = None):
    # reads csv--> loads model --> prediccts lanes --> saves results
    start_time = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Predict] using device: {device}")


    # find checkpoint:

    if checkpoint_path is None:
        phase2 = CHECKPOINT_DIR / "best_phase2.pth"
        phase1 = CHECKPOINT_DIR / "best_phase1.pth"

        if phase2.exists():
            checkpoint_path= str(phase2)
            print(f"[predict] using phase 2 model")
        elif phase1.exists():
            checkpoint_path = str(phase1)
            print(f"[predict] using phase 1 model (phase 2 is not available")
        else:
            raise FileNotFoundError(f"NO trained model was found in {CHECKPOINT_DIR}\n"
                                    f"run train.py fist")



    # load model:
    print(f"[Predict] loading model..")
    model = load_trained_model(checkpoint_path, device = device)
    model.eval()

    #load csv
    print(f"[Predict loading: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"[Predict] {len(df):,} total rows")

    # handle ignore rows - skip them:
    if "Ignore" in df.columns:
        ignore_mask = df["Ignore"]==1.0
        print(f"[Predixted] {ignore_mask.sum():,} rows skipped (Ignore =1)")
    else:
        ignore_mask = pd.Series([False]*len(df))            # no ignore column predict everything

    predict_df = df[~ignore_mask].copy() # rows we will actually predict

    print(f"[predict] {len(predict_df):,} rows to predict")
    dataset = PredictionDataset(predict_df, path_prefix= path_prefix)
    loader = DataLoader(dataset, batch_size= batch_size, shuffle=False, num_workers=0, drop_last=False)

    print(f"[Predict] Predicting lanes..")
    print(f"[Predict] confidence threshold: {threshold:.0%}")

    all_predictions = {}        # mapping row index --> prediction result
    n_auto_coded = 0
    n_review = 0
    n_invalid = 0

    with torch.no_grad():
        for batch_idx, (imgage, gps, indices, valids) in enumerate(loader):
            images = images.to(device)
            gps = gps.to(device)

            # forward pass

            logits = model(images,gps)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_class = probs.max(dim=1)

            # preocess each image in the batch:
            for i in range(len(indices)):
                row_idx= indices[i].item()
                conf = confidence[i].item()
                cls_idx = predicted_class[i].item()
                is_valid = bool(valids[i].item())

                if not is_valid:
                    # image file couldnt be loaded
                    lane_code = "REVIEW"
                    needs_review = True
                    n_invalid +=1
                elif conf>= threshold:
                    lane_code = IDX_TO_LANE[cls_idx]
                    needs_review = False
                    n_auto_coded +=1
                else:
                    lane_code - IDX_TO_LANE[cls_idx]
                    needs_review = True
                    n_review+=1
                all_predictions[row_idx] = {
                    "lane": lane_code,
                    "confidence": round(conf,3),
                    "needs_review": needs_review,}

            if(batch_idx+1)%1000==0:
                processed = (batch_idx+1)* batch_size
                elapsed = time.time() - start_time
                pct = processed /len(predict_df)*100
                print(f"{processed:,}/{len(predict_df):,} "
                f"({pct:.1f}%) | "
                f"Elapsed: {elapsed:.0f}s")

    # write results back to csv
    print(f"\n[predict] writing results to :{output_csv}")


    # add three columns to the dataframe
    df["Lane_AI"] = ""
    df["AI_Confidence"] = 0.0
    df["Needs_review"] = False


    for row_idx, pred in all_predictions.items():
        orig_idx = predict_df.index[row_idx]

        df.at[orig_idx, "Lane_AI"] = pred["lane"]
        df.at[orig_idx, "AI_Confidence"] = pred["confidence"]
        df.at[orig_idx, "Needs_Review"] = pred["needs_review"]


    df.loc[ignore_mask, "Lane_AI"] = "IGNORED"
    df.loc[ignore_mask, "Needs_Review"] = False

    # save to disk
    output_path = Path(output_csv)
    output_path.parent.mkdir(parent= True, exist_ok = true)
    df.to_csv(output_csv, index = False)

    # print summary:
    elapsed = time.time() - start_time
    total = len(predict_df)

    print(f"\n{'='*50}")
    print(f"Prediction complete")
    print(f"{'='*50}")
    print(f"total rows: {len(df):,}")
    print(f"predicted: {total:,}")
    print(f"auto coded: {n_auto_coded:,}"
        f"({n_auto_coded/total*100:.1f}%")
    print(f"Needs review: {n_review} "
        f"({n_review/total*100:.1f}%)")
    print("invalid images: {n_invalid:,}")
    print(f"time elapsed: {elapsed:.0f}s"
            f"({elapsed/60:.1f} mins)")
    print(f"\nOutput saved to: {output_csv}")
    for lane, count in df["Lane_AI"].value_counts().items():
        pct= count/len(df)*100
        print(f" {lane:>8} {count:>10,} ({pct:.1f}%)")
        # ---- log embeddings to TensorBoard ----
        print(f"\n[TensorBoard] Computing prediction embeddings...")
        from torch.utils.tensorboard import SummaryWriter
        import torch.nn.functional as F

        writer = SummaryWriter(log_dir="runs/geosolve_predictions")

        # take 500 random samples for visualization
        sample_df = predict_df.sample(min(500, len(predict_df)), random_state=42).reset_index(drop=True)
        sample_dataset = PredictionDataset(sample_df, path_prefix=path_prefix)
        sample_loader = DataLoader(sample_dataset, batch_size=32, shuffle=False, num_workers=0)

        features_list = []
        labels_list = []
        images_list = []

        model.eval()
        with torch.no_grad():
            for imgs, gps, indices, valids in sample_loader:
                imgs = imgs.to(device)
                gps = gps.to(device)

                # extract backbone features
                image_features = model.backbone(imgs)
                features_list.append(image_features.cpu())

                # use predicted lane as label
                logits = model(imgs, gps)
                _, predicted = torch.softmax(logits, dim=1).max(dim=1)
                labels_list.append(predicted.cpu())

                # resize images to 32x32 for thumbnails
                small = F.interpolate(
                    imgs.cpu(), size=(32, 32),
                    mode="bilinear", align_corners=False
                )
                images_list.append(small)

        features = torch.cat(features_list)
    return df


# command line inteface

def get_args():
    parser = argparse.ArgumentParser(description = "geosolve lane detection prediction")
    parser.add_argument("--input", required=True, help= "Path to input survey csv")
    parser.add_argument("--output", required = True, help = "Path to save output csv with predictions")
    parser.add_argument("--checkpoint", default=None, help="Path to model pth. file (default- auto detect)")
    parser.add_argument("--threshold", type=float, default= 0.90, help= "Confidence threshold 0-1 (default0.90)")
    parser.add_argument("--batch_size", type=int, default = 32, help = "Images per batch (default:32)")
    parser.add_argument("--path_prefix", default = "", help=" Replace J:/ in paths withthis prefix")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    predict(input_csv = args.input, output_csv = args.output, checkpoint_path= args.checkpoint, threshold = args.threshold, batch_size = args.batch_size, path_prefix = args.path_prefix,)


































