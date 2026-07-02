# visualize_embeddings.py
# -----------------------
# Generates 3D embedding visualization for TensorBoard
# Run AFTER training is complete
# Much faster than full evaluation — just computes features

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path

from config import CHECKPOINT_DIR, IDX_TO_LANE, BATCH_SIZE
from dataset import get_val_loader, get_test_loader
from model import load_trained_model


def log_embeddings(model, loader, writer, device, tag, max_images=500):
    print(f"Computing embeddings: {tag}...")
    model.eval()

    features_list = []
    labels_list   = []
    images_list   = []
    count         = 0

    with torch.no_grad():
        for images, gps, labels in loader:
            images = images.to(device)
            gps    = gps.to(device)

            # extract features from EfficientNet backbone
            image_features = model.backbone(images)
            features_list.append(image_features.cpu())
            labels_list.append(labels.cpu())

            # small thumbnails for hover preview
            small = F.interpolate(
                images.cpu(), size=(32, 32),
                mode="bilinear", align_corners=False
            )
            images_list.append(small)

            count += images.shape[0]
            if count >= max_images:
                break

    features    = torch.cat(features_list)[:max_images]
    labels      = torch.cat(labels_list)[:max_images]
    thumbs      = torch.cat(images_list)[:max_images]
    label_names = [IDX_TO_LANE[l.item()] for l in labels]

    # normalize thumbnails to [0,1]
    thumbs = thumbs - thumbs.min()
    thumbs = thumbs / (thumbs.max() + 1e-8)

    writer.add_embedding(
        features,
        metadata  = label_names,
        label_img = thumbs,
        tag       = tag,
        global_step = 0
    )

    from collections import Counter
    counts = Counter(label_names)
    print(f"  Done! Classes: {dict(counts)}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load best model
    phase2_path = CHECKPOINT_DIR / "best_phase2.pth"
    phase1_path = CHECKPOINT_DIR / "best_phase1.pth"

    if phase2_path.exists():
        checkpoint_path = str(phase2_path)
        print("Using Phase 2 model")
    elif phase1_path.exists():
        checkpoint_path = str(phase1_path)
        print("Using Phase 1 model")
    else:
        print("ERROR: No trained model found!")
        exit(1)

    model = load_trained_model(checkpoint_path, device=device)

    writer = SummaryWriter(log_dir="runs/geosolve_embeddings")

    # val embeddings
    print("\nLoading val data...")
    val_loader = get_val_loader(batch_size=32, num_workers=0)
    log_embeddings(model, val_loader, writer, device,
                   tag="Val_embeddings", max_images=500)

    # test embeddings
    print("\nLoading test data...")
    test_loader = get_test_loader(batch_size=32, num_workers=0)
    log_embeddings(model, test_loader, writer, device,
                   tag="Test_embeddings", max_images=500)

    writer.close()
    print("\nDone!")
    print("View at: http://localhost:6006 → PROJECTOR tab")