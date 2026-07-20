# train.py
# --------
# The training loop — connects dataset.py and model.py
# Phase 1: frozen backbone (5 epochs)
# Phase 2: full fine-tuning (up to 30 epochs)
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from config import (
    BATCH_SIZE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    LR_PHASE1,
    LR_PHASE2,
    EARLY_STOP_PATIENCE,
    NUM_WORKERS,
    CHECKPOINT_DIR,
    LANE_CLASSES,
    N_CLASSES, IDX_TO_LANE,
)
from dataset import get_train_loader, get_val_loader
from model import get_model, freeze_backbone, unfreeze_backbone, count_parameters


# ================================================================
# PLOT TRAINING PROGRESS
# ================================================================

def plot_training_progress(
    train_losses, val_losses, val_accuracies, phase, save_path=None
):
    if save_path is None:
        save_path = CHECKPOINT_DIR / f"phase{phase}_progress.png"

    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"GeoSolve Lane Detection — Phase {phase} Training Progress",
        fontsize=14,
        fontweight="bold"
    )

    ax1.plot(epochs, train_losses, "b-o", label="Train loss", linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, "r--s", label="Val loss", linewidth=2, markersize=6)
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    if train_losses:
        ax1.annotate(
            f"{train_losses[-1]:.3f}",
            xy=(epochs[-1], train_losses[-1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="blue"
        )
    if val_losses:
        ax1.annotate(
            f"{val_losses[-1]:.3f}",
            xy=(epochs[-1], val_losses[-1]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=9,
            color="red"
        )

    ax2.plot(epochs, val_accuracies, "g-^", label="Val accuracy", linewidth=2, markersize=6)
    ax2.set_title("Validation accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    if val_accuracies:
        best_acc   = max(val_accuracies)
        best_epoch = val_accuracies.index(best_acc) + 1
        ax2.axhline(y=best_acc, color="green", linestyle=":", alpha=0.5, label=f"Best: {best_acc:.2f}%")
        ax2.annotate(
            f"Best: {best_acc:.2f}%\n(epoch {best_epoch})",
            xy=(best_epoch, best_acc),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
            color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=1)
        )
        ax2.legend()

    ax2.axhline(y=95.0, color="orange", linestyle="--", alpha=0.4, label="Target: 95%")
    ax2.text(
        0.02, 95.5, "target 95%",
        transform=ax2.get_xaxis_transform(),
        fontsize=8, color="orange", alpha=0.7
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved training progress → {save_path}")


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(model, loader, optimiser, criterion, device, epoch):
    """
    Trains the model for ONE complete pass through the training data.
    Processes all images in batches, updates weights after each batch.
    Returns average loss across all batches this epoch.
    """
    model.train()

    total_loss   = 0.0
    correct      = 0
    total_images = 0
    start_time   = time.time()

    for batch_idx, (images, gps, labels) in enumerate(loader):

        images = images.to(device)
        gps    = gps.to(device)
        labels = labels.to(device)

        # STEP 1: zero gradients
        optimiser.zero_grad()

        # STEP 2: forward pass
        logits = model(images, gps)

        loss = criterion(logits, labels)

        # skip NaN batches BEFORE backward - prevent gradient corruption
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN loss at batch {batch_idx + 1}, skipping")
            optimiser.zero_grad()
            continue

        # STEP 4: backward pass
        loss.backward()

        # prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # STEP 5: update weights
        optimiser.step()

        # track statistics
        total_loss   += loss.item()
        predicted     = logits.argmax(dim=1)
        correct      += (predicted == labels).sum().item()
        total_images += labels.size(0)

        # print progress every 1000 batches
        if (batch_idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch} | "
                  f"Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Elapsed: {elapsed:.0f}s")

    avg_loss       = total_loss / max(len(loader), 1)
    train_accuracy = correct / max(total_images, 1) * 100
    epoch_time     = time.time() - start_time

    print(f"  Epoch {epoch} complete | "
          f"Avg loss: {avg_loss:.4f} | "
          f"Train accuracy: {train_accuracy:.1f}% | "
          f"Time: {epoch_time:.0f}s")

    return avg_loss


# ================================================================
# EVALUATE ON VALIDATION SET
# ================================================================

def evaluate(model, loader, criterion, device):
    """
    Evaluates model on validation set after each epoch.
    No weight updates — just measuring accuracy.
    Returns (val_loss, val_accuracy).
    """
    model.eval()

    total_loss   = 0.0
    correct      = 0
    total_images = 0

    with torch.no_grad():
        for images, gps, labels in loader:
            images = images.to(device)
            gps    = gps.to(device)
            labels = labels.to(device)

            logits = model(images, gps)
            loss   = criterion(logits, labels)

            # skip NaN batches
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            predicted = logits.argmax(dim=1)
            total_loss   += loss.item()
            correct      += (predicted == labels).sum().item()
            total_images += labels.size(0)

    avg_loss     = total_loss / max(len(loader), 1)
    val_accuracy = correct / max(total_images, 1) * 100

    return avg_loss, val_accuracy


# ================================================================
# LOG EMBEDDINGS TO TENSORBOARD
# ================================================================

def log_embeddings(model, loader, writer, device, tag="embeddings", max_images=500):
    """
    Logs image embeddings to TensorBoard for 3D visualization.

    Shows how the model groups images in feature space.
    Similar images cluster together — good separation = model learned well.

    model:      trained model
    loader:     data loader (val or test)
    writer:     TensorBoard SummaryWriter
    device:     cpu or cuda
    tag:        name shown in TensorBoard
    max_images: how many images to visualize (keep low for performance)
    """
    print(f"[TensorBoard] Computing embeddings for {tag}...")
    model.eval()

    features_list = []
    labels_list = []
    images_list = []
    count = 0

    with torch.no_grad():
        for images, gps, labels in loader:
            images = images.to(device)
            gps = gps.to(device)

            # extract features BEFORE the final classifier
            # these are the 1280-dim image features from EfficientNet
            image_features = model.backbone(images)
            # shape: (batch_size, 1280)

            features_list.append(image_features.cpu())
            labels_list.append(labels.cpu())

            # store small version of images for the thumbnail preview
            # TensorBoard shows the actual image when you hover over a dot
            # resize to 32x32 to save memory
            small_images = torch.nn.functional.interpolate(
                images.cpu(), size=(32, 32), mode="bilinear", align_corners=False
            )
            images_list.append(small_images)

            count += images.shape[0]
            if count >= max_images:
                break

    # concatenate all batches
    features = torch.cat(features_list)[:max_images]
    labels = torch.cat(labels_list)[:max_images]
    imgs = torch.cat(images_list)[:max_images]

    # convert label indices to lane names for the legend
    label_names = [IDX_TO_LANE[l.item()] for l in labels]

    # normalize images to [0,1] for display
    # (they were normalized for training, need to reverse for display)
    imgs = imgs - imgs.min()
    imgs = imgs / imgs.max()

    # add to TensorBoard
    writer.add_embedding(
        features,
        metadata=label_names,
        label_img=imgs,
        tag=tag,
        global_step=0
    )

    # count per class
    from collections import Counter
    counts = Counter(label_names)
    print(f"[TensorBoard] Embeddings logged: {tag}")
    print(f"  Classes: {dict(counts)}")

# ================================================================
# MAIN TRAINING FUNCTION
# ================================================================

def train(resume_from=None):
    """
    Full training pipeline.
    Phase 1: backbone frozen, only classifier trains (5 epochs)
    Phase 2: full network fine-tuning (up to 30 epochs)
    """

    # setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    # load data
    print("[Train] Loading data...")
    train_loader = get_train_loader(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    val_loader = get_val_loader(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # build model
    print("[Train] Building model...")
    model = get_model(pretrained=True).to(device)

    # if resuming from previous chunk, load those weights
    # uses a PARTIAL load — any layer whose shape doesn't match
    # (e.g. the final classifier layer when N_CLASSES changes)
    # is skipped and left at its freshly-initialized value,
    # while everything else (backbone, gps_processor, etc.)
    # transfers over normally
    if resume_from is not None and Path(resume_from).exists():
        checkpoint     = torch.load(resume_from, map_location=device)
        old_state_dict = checkpoint["model"]
        new_state_dict = model.state_dict()

        transferred = []
        skipped     = []

        for key in old_state_dict:
            if key in new_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
                new_state_dict[key] = old_state_dict[key]
                transferred.append(key)
            else:
                skipped.append(key)

        model.load_state_dict(new_state_dict)

        print(f"[Train] Resumed from: {resume_from}")
        print(f"[Train] Transferred {len(transferred)} layers, "
              f"skipped {len(skipped)} layers (shape mismatch — reinitialized)")
        if skipped:
            print(f"[Train] Skipped layers: {skipped}")
    elif resume_from is not None:
        print(f"[Train] WARNING: checkpoint not found: {resume_from}")

    # setup TensorBoard writer
    writer = SummaryWriter(log_dir="runs/geosolve_training")
    print("[Train] TensorBoard logging to: runs/geosolve_training")
    print("[Train] Run: tensorboard --logdir=runs")

    # loss function
    # NOTE: class_weights below are currently computed but NOT applied
    # to the criterion — kept as CrossEntropyLoss() without weights
    # since weighted loss previously caused NaN training issues.
    # Left here in case you want to re-enable weighting later.
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # PHASE 1 — FROZEN BACKBONE
    # ============================================================

    print("\n" + "=" * 50)
    print("PHASE 1 — Frozen backbone")
    print("=" * 50)

    freeze_backbone(model)

    optimiser_phase1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1
    )

    best_val_accuracy_p1 = 0.0
    train_losses_p1      = []
    val_losses_p1        = []
    val_accuracies_p1    = []

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        print(f"\nPhase 1 | Epoch {epoch}/{EPOCHS_PHASE1}")

        train_loss = train_one_epoch(
            model, train_loader, optimiser_phase1,
            criterion, device, epoch
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        print(f"  Val loss: {val_loss:.4f} | "
              f"Val accuracy: {val_accuracy:.2f}%")

        train_losses_p1.append(train_loss)
        val_losses_p1.append(val_loss)
        val_accuracies_p1.append(val_accuracy)

        plot_training_progress(
            train_losses_p1, val_losses_p1, val_accuracies_p1, phase=1
        )

        if val_accuracy > best_val_accuracy_p1:
            best_val_accuracy_p1 = val_accuracy
            save_path = CHECKPOINT_DIR / "best_phase1.pth"
            torch.save({
                "model":   model.state_dict(),
                "epoch":   epoch,
                "val_acc": val_accuracy,
                "phase":   1,
            }, save_path)
            print(f"  ✓ Saved best Phase 1 model "
                  f"(val_acc={val_accuracy:.2f}%)")

        # log to TensorBoard
        writer.add_scalar("Phase1/Train_Loss", train_loss, epoch)
        writer.add_scalar("Phase1/Val_Loss", val_loss, epoch)
        writer.add_scalar("Phase1/Val_Accuracy", val_accuracy, epoch)

    print(f"\nPhase 1 complete. "
          f"Best val accuracy: {best_val_accuracy_p1:.2f}%")

    # ============================================================
    # PHASE 2 - FULL FINE-TUNING
    # ============================================================

    print("\n" + "=" * 50)
    print("PHASE 2 — Full fine-tuning")
    print("=" * 50)

    best_phase1_path = CHECKPOINT_DIR / "best_phase1.pth"
    checkpoint       = torch.load(best_phase1_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print("[Train] Loaded best Phase 1 weights")

    unfreeze_backbone(model)

    optimiser_phase2 = torch.optim.Adam(
        model.parameters(),
        lr=LR_PHASE2
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser_phase2,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_accuracy_p2 = 0.0
    no_improve_count     = 0
    train_losses_p2      = []
    val_losses_p2        = []
    val_accuracies_p2    = []

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        print(f"\nPhase 2 | Epoch {epoch}/{EPOCHS_PHASE2}")

        train_loss = train_one_epoch(
            model, train_loader, optimiser_phase2,
            criterion, device, epoch
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        print(f"  Val loss: {val_loss:.4f} | "
              f"Val accuracy: {val_accuracy:.2f}%")

        train_losses_p2.append(train_loss)
        val_losses_p2.append(val_loss)
        val_accuracies_p2.append(val_accuracy)

        plot_training_progress(
            train_losses_p2, val_losses_p2, val_accuracies_p2, phase=2
        )

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy_p2:
            best_val_accuracy_p2 = val_accuracy
            no_improve_count     = 0
            save_path = CHECKPOINT_DIR / "best_phase2.pth"
            torch.save({
                "model":   model.state_dict(),
                "epoch":   epoch,
                "val_acc": val_accuracy,
                "phase":   2,
            }, save_path)
            print(f"  ✓ Saved best Phase 2 model "
                  f"(val_acc={val_accuracy:.2f}%)")
        else:
            no_improve_count += 1
            print(f"  No improvement "
                  f"({no_improve_count}/{EARLY_STOP_PATIENCE})")

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping — "
                  f"no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

        # log to TensorBoard
        writer.add_scalar("Phase2/Train_Loss", train_loss, epoch)
        writer.add_scalar("Phase2/Val_Loss", val_loss, epoch)
        writer.add_scalar("Phase2/Val_Accuracy", val_accuracy, epoch)

    print(f"\nTraining complete!")
    print(f"Best Phase 1 accuracy: {best_val_accuracy_p1:.2f}%")
    print(f"Best Phase 2 accuracy: {best_val_accuracy_p2:.2f}%")

    # ---- log embeddings after Phase 1 ----
    print("\nLogging Phase 1 embeddings to TensorBoard...")
    # reload best phase 1 model first
    checkpoint = torch.load(CHECKPOINT_DIR / "best_phase1.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    log_embeddings(model, val_loader, writer, device,
                   tag="Phase1_embeddings", max_images=500)

    # ---- log embeddings after Phase 2 ----
    print("\nLogging Phase 2 embeddings to TensorBoard...")
    checkpoint = torch.load(CHECKPOINT_DIR / "best_phase2.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    log_embeddings(model, val_loader, writer, device,
                   tag="Phase2_embeddings", max_images=500)

    writer.close()
    print(f"Best model saved to:   {CHECKPOINT_DIR / 'best_phase2.pth'}")


# ================================================================
# QUICK TEST
# ================================================================

def quick_test():
    """Runs 2 batches to verify everything works before full training."""
    print("\nQUICK TEST — 2 batches only")
    print("=" * 40)

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = get_train_loader(batch_size=4, num_workers=0)
    val_loader   = get_val_loader(batch_size=4,   num_workers=0)
    model        = get_model(pretrained=False).to(device)
    criterion    = nn.CrossEntropyLoss()
    optimiser    = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for batch_idx, (images, gps, labels) in enumerate(train_loader):
        images = images.to(device)
        gps    = gps.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        logits = model(images, gps)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()
        print(f"  Train batch {batch_idx+1} | loss: {loss.item():.4f}")
        if batch_idx >= 1:
            break

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, gps, labels) in enumerate(val_loader):
            images    = images.to(device)
            gps       = gps.to(device)
            labels    = labels.to(device)
            logits    = model(images, gps)
            loss      = criterion(logits, labels)
            predicted = logits.argmax(dim=1)
            correct   = (predicted == labels).sum().item()
            print(f"  Val batch {batch_idx+1} | "
                  f"loss: {loss.item():.4f} | "
                  f"correct: {correct}/4")
            if batch_idx >= 1:
                break

    print("\nQuick test passed!")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("GeoSolve Lane Detection — Training")
    print("=" * 50)
    # quick_test()
    train(resume_from="checkpoints/best_phase2_chunks_v2_final.pth")