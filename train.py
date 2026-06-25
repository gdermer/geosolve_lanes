# train.py
# --------
# The training loop — connects dataset.py and model.py
# Phase 1: frozen backbone (5 epochs)
# Phase 2: full fine-tuning (up to 30 epochs)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path

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
    N_CLASSES,
)
from dataset import get_train_loader, get_val_loader
from model import get_model, freeze_backbone, unfreeze_backbone, count_parameters


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
    # switch to training mode
    # enables dropout, calculates fresh batch norm stats

    total_loss   = 0.0
    correct      = 0
    total_images = 0
    start_time   = time.time()

    for batch_idx, (images, gps, labels) in enumerate(loader):

        # move data to device (CPU or GPU)
        images = images.to(device)
        gps    = gps.to(device)
        labels = labels.to(device)

        # STEP 1: zero gradients
        optimiser.zero_grad()
        # must do before every forward pass
        # PyTorch accumulates gradients by default
        # if we don't zero them, old gradients add to new ones

        # STEP 2: forward pass
        logits = model(images, gps)
        # run all images through the model
        # logits shape: (batch_size, 4)

        # STEP 3: calculate loss
        loss = criterion(logits, labels)
        # how wrong was the model on average for this batch?
        # one number — higher = more wrong

        # STEP 4: backward pass
        loss.backward()
        # PyTorch calculates gradients for ALL weights
        # which weights caused the mistake and by how much?

        # STEP 5: update weights
        optimiser.step()
        # nudge all weights in direction that reduces loss
        # weight = weight - (learning_rate × gradient)

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

    avg_loss       = total_loss / len(loader)
    train_accuracy = correct / total_images * 100
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
    # switch to evaluation mode
    # disables dropout for consistent results

    total_loss   = 0.0
    correct      = 0
    total_images = 0

    with torch.no_grad():
        # no gradient tracking needed
        # saves memory and speeds up evaluation

        for images, gps, labels in loader:
            images = images.to(device)
            gps    = gps.to(device)
            labels = labels.to(device)

            logits    = model(images, gps)
            loss      = criterion(logits, labels)
            predicted = logits.argmax(dim=1)

            total_loss   += loss.item()
            correct      += (predicted == labels).sum().item()
            total_images += labels.size(0)

    avg_loss     = total_loss / len(loader)
    val_accuracy = correct / total_images * 100

    return avg_loss, val_accuracy


# ================================================================
# MAIN TRAINING FUNCTION
# ================================================================

def train():
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

    # loss function with class weights
    # handles imbalance: Lane1=95.8%, SK1=0.2%
    class_counts  = [2193795, 82741, 10709, 4480]
    # updated to match our actual train split counts
    total_count   = sum(class_counts)
    class_weights = torch.tensor(
        [total_count / (N_CLASSES * count) for count in class_counts],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)


    # ============================================================
    # PHASE 1 — FROZEN BACKBONE
    # ============================================================
    # Only classifier + GPS processor train
    # Backbone stays frozen at pretrained ImageNet weights

    print("\n" + "=" * 50)
    print("PHASE 1 — Frozen backbone")
    print("=" * 50)

    freeze_backbone(model)
    # sets requires_grad=False on backbone weights

    optimiser_phase1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1
    )
    # only passes TRAINABLE parameters to optimiser
    # frozen backbone excluded

    best_val_accuracy_p1 = 0.0

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        print(f"\nPhase 1 | Epoch {epoch}/{EPOCHS_PHASE1}")

        train_one_epoch(
            model, train_loader, optimiser_phase1,
            criterion, device, epoch
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        print(f"  Val loss: {val_loss:.4f} | "
              f"Val accuracy: {val_accuracy:.2f}%")

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

    # Phase 1 loop ends here
    print(f"\nPhase 1 complete. "
          f"Best val accuracy: {best_val_accuracy_p1:.2f}%")


    # ============================================================
    # PHASE 2 — FULL FINE-TUNING
    # ============================================================
    # Entire network trains with small learning rate
    # Backbone gently adapts to NZ road images

    print("\n" + "=" * 50)
    print("PHASE 2 — Full fine-tuning")
    print("=" * 50)

    # load best Phase 1 weights
    best_phase1_path = CHECKPOINT_DIR / "best_phase1.pth"
    checkpoint = torch.load(best_phase1_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print("[Train] Loaded best Phase 1 weights")

    unfreeze_backbone(model)
    # sets requires_grad=True on ALL weights

    optimiser_phase2 = torch.optim.Adam(
        model.parameters(),
        lr=LR_PHASE2
        # much smaller LR to protect pretrained weights
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser_phase2,
        mode="max",      # we want accuracy to go UP
        factor=0.5,      # halve LR when plateauing
        patience=3,      # wait 3 epochs before reducing

    )

    best_val_accuracy_p2 = 0.0
    no_improve_count     = 0

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        print(f"\nPhase 2 | Epoch {epoch}/{EPOCHS_PHASE2}")

        train_one_epoch(
            model, train_loader, optimiser_phase2,
            criterion, device, epoch
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        print(f"  Val loss: {val_loss:.4f} | "
              f"Val accuracy: {val_accuracy:.2f}%")

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

    # Phase 2 loop ends here
    print(f"\nTraining complete!")
    print(f"Best Phase 1 accuracy: {best_val_accuracy_p1:.2f}%")
    print(f"Best Phase 2 accuracy: {best_val_accuracy_p2:.2f}%")
    print(f"Best model saved to: "
          f"{CHECKPOINT_DIR / 'best_phase2.pth'}")


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
    # quick_test()  # uncomment to test before training
    train()