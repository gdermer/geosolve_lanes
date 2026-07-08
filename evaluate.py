import torch
import numpy as np
import matplotlib
from torch.utils.tensorboard import SummaryWriter

from train import log_embeddings

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import (N_CLASSES, LANE_CLASSES,  IDX_TO_LANE, CHECKPOINT_DIR, BATCH_SIZE, NUM_WORKERS)
from dataset import get_test_loader
from model import load_trained_model


def compute_confusion_matrix(all_labels, all_predictions, n_classes):


    matrix = np.zeros((n_classes, n_classes), dtype = int)
    for label, pred in zip(all_labels, all_predictions):
        matrix[label][pred] +=1

    return matrix

def plot_confusion_matrix(matrix, class_names, save_path):
    # saved visual confusion matrix as a PNG file
    fig, ax = plt.subplots(figsize = (7,8))
    row_sums = matrix.sum(axis = 1, keepdims = True)
    matrix_pct = matrix/ row_sums*100
    im = ax.imshow(matrix_pct, interpolation="nearest", cmap = "Blues")
    plt.colorbar(im, ax = ax, label="% of actual class")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(class_names, rotation = 45, ha = "right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted lanes")
    ax.set_ylabel("actual lane")
    ax.set_title("confusion matrix - test set \n(% of each actual class)")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = "white" if matrix_pct[i,j]> 50 else "black"
            ax.text(j,i,
                    f"{matrix_pct[i,j]:.1f}%\n({matrix[i,j]:,})",
                    ha="center",
                    va= "center",
                    color = color,
                    fontsize=9,)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches = "tight")
    plt.close(fig)
    print(f"[evaluate] confusion matrix saved --> {save_path}")

    from sklearn.metrics import (
        f1_score, precision_score, recall_score, classification_report
    )

    # ---- Per class and aggregate metrics ----
    print(f"\n{'=' * 50}")
    print(f"SKLEARN METRICS")
    print(f"{'=' * 50}")

    # full classification report
    print(f"\nClassification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=3
    ))

    # macro F1 (unweighted - shows if balanced across classes)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    macro_precision = precision_score(all_labels, all_predictions, average='macro')
    macro_recall = recall_score(all_labels, all_predictions, average='macro')

    # weighted F1 (weighted by class frequency - comparable to accuracy)
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
    weighted_precision = precision_score(all_labels, all_predictions, average='weighted')
    weighted_recall = recall_score(all_labels, all_predictions, average='weighted')

    print(f"Macro metrics (unweighted average across classes):")
    print(f"  Macro F1:        {macro_f1:.3f}")
    print(f"  Macro Precision: {macro_precision:.3f}")
    print(f"  Macro Recall:    {macro_recall:.3f}")

    print(f"\nWeighted metrics (weighted by class frequency):")
    print(f"  Weighted F1:        {weighted_f1:.3f}")
    print(f"  Weighted Precision: {weighted_precision:.3f}")
    print(f"  Weighted Recall:    {weighted_recall:.3f}")

    print(f"\nComparison:")
    print(f"  Overall accuracy:  {overall_accuracy:.3f}%")
    print(f"  Weighted F1:       {weighted_f1:.3f}  <- comparable to accuracy")
    print(f"  Macro F1:          {macro_f1:.3f}  <- shows class balance")
    print(f"  Previous team F1:  0.640            <- their best result")

    # save to results file
    with open(results_path, 'a') as f:
        f.write(f"\nSKLEARN METRICS\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Macro F1:        {macro_f1:.3f}\n")
        f.write(f"Macro Precision: {macro_precision:.3f}\n")
        f.write(f"Macro Recall:    {macro_recall:.3f}\n")
        f.write(f"Weighted F1:     {weighted_f1:.3f}\n")
        f.write(f"Weighted Precision: {weighted_precision:.3f}\n")
        f.write(f"Weighted Recall:    {weighted_recall:.3f}\n")

def evaluate_model(checkpoint_path, device = "cpu"):
    # loads train model and evaluates on test set
    print("f\n{'='*50}")
    print(f" evaluation- test set")
    print(f"{'='*50}")
    print(f" loading model from : {checkpoint_path}")
    model = load_trained_model(checkpoint_path, device = device)
    model.eval()
    print(f"\nLoading test data..")
    test_loader = get_test_loader(batch_size = BATCH_SIZE, num_workers= NUM_WORKERS)
    print(f"Running evaluation..")
    all_labels = []
    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for batch_idx, (images, gps, labels) in enumerate(test_loader):
            images = images.to(device)
            gps = gps.to(device)
            labels = labels.to(device)
            logits = model(images, gps)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

            if (batch_idx+1)%500==0:
                print(f" processed {(batch_idx+1)* BATCH_SIZE:,} images..")

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)


    # calculate matrix
    n_classes = N_CLASSES
    class_names = [IDX_TO_LANE[i] for i in range(n_classes)]
    overall_accuracy = (all_labels == all_predictions).mean()*100
    print(f"\n{'='*50}")
    print(f"results:")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"total test images: {len(all_labels):,}")
    print(f"\n Per class accuracy:")
    print(f"{'Lane':>6} {'correct':>10}, {'Total':>10} {'Accuracy':>10} {'Abgconf':>10}")
    print('-'*50)
    for class_idx in range(n_classes):
        class_mask = all_labels == class_idx
        if class_mask.sum() ==0:
            print(f" {class_names[class_idx]:>4}: no test samples")
            continue
        class_correct = (all_predictions[class_mask] == class_idx).sum()
        class_total = class_mask.sum()
        class_accuracy = class_correct/ class_total *100
        class_confidence = all_confidences[class_mask].mean()*100
        print(f" {class_names[class_idx]:>4}: "
              f"{class_correct:>10,} / "
              f"{class_total:>8,}   "
              f"{class_accuracy:>8.2f}% "
              f"{class_confidence:8.2f}%")


    # confidence analysis:
    print(f" confodence analysis:")
    thresholds = [0.70, 0.80, 0.90, 0.95]
    for thresh in thresholds:
        auto_coded = (all_confidences>= thresh).mean()*100
        print(f" >= {thresh:.0%} confidence: "
              f"{auto_coded:.1f}% auto coded,"
              f"{100-auto_coded:.1f}% needs review")


    matrix = compute_confusion_matrix(all_labels, all_predictions, n_classes)

    print(f"\n confusion matrix (counts): ")
    print(f"{'':>8}", end = "")
    for name in class_names:
        print(f"n{name:>8}", end = "")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:>8}", end="")
        for j in range(n_classes):
            print(f"{matrix[i][j]:>8,}", end="")
        print()

    plot_path = CHECKPOINT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(matrix, class_names, plot_path)

    results_path = CHECKPOINT_DIR / "evaluation_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Geosolve lane detection- evaluaitonl results\n")
        f.write(f"{'='*50}\n")
        f.write(f"model: {checkpoint_path}\n")
        f.write(f"Overall acuuracy: {overall_accuracy:.2f}%\n")
        f.write(f"per class accuracy:\n")
        for class_idx in range(n_classes):
            class_mask = all_labels==class_idx
            if class_mask.sum() ==0:
                continue
            class_correct = (all_predictions[class_mask]== class_idx).sum()
            class_total = class_mask.sum()
            class_accuracy = class_correct / class_total*100
            f.write(f" lane {class_names[class_idx]}: "
                    f"{class_accuracy:.2f}% "
                    f"{class_correct:,}/{class_total:,}\n")
    print(f"\n[evaluate] results saved --> {results_path}")
    print(f"\n evaluation complete")
    # ---- log embeddings to TensorBoard ----
    writer = SummaryWriter(log_dir="runs/geosolve_evaluation")
    log_embeddings(model, test_loader, writer, device,
                   tag="Test_embeddings", max_images=500)
    writer.close()
    print("[TensorBoard] Evaluation embeddings logged")
    print("View at: tensorboard --logdir=runs")
    return overall_accuracy


# main
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    phase2_path =CHECKPOINT_DIR / "best_phase2.pth"
    phase1_path = CHECKPOINT_DIR / "best_phase1.pth"
    if phase2_path.exists():
        checkpoint_path = phase2_path
        print(f"using phase 2 model")
    elif phase1_path.exists():
        checkpoint_path = phase1_path
        print(f" phase 2 has not found using phase 1 model")
    else:
        print(f" ERROR: no trained model found in {CHECKPOINT_DIR}")
        print(f"Run train.py first!")
        exit(1)
    evaluate_model(str(checkpoint_path), device= device)





















