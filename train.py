import torch
import torch.nn as nn
from pydantic_core.core_schema import model_field
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path
from config import(
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
from dataset import get_train_loader, get_test_loader, get_val_loader
from model import get_model, freeze_backbone, unfreeze_backbone, count_parameters

def train_one_epoch(model, loader, optimiser, criterion, device, epoch):     # train the model for one complete pass, processes all images in batches, updates weights after each batch
    # return average loss across all batches this epoch
    model.train()       # switch mode to training mode
    total_loss = 0.0        # accomulate lossa cross all batches
    correct = 0     # counts correct predictions
    total_images = 0   # count total images processed
    start_time = time.time()


    for batch_idx, (images,gps, labels) in enumerate(loader):

        images = images.to(device)
        gps = gps.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()           # step 1: zero the gradiants must zero them so they will noa ccumulate before every forward pass
        logits = model(images,gps)       # step 2: forward pass, run all images trought the model
        loss = criterion(logits, labels)
        loss.backward()     # step 3: calculate loss, return one number how wrong was the modelon average?
        loss.backwards()     # step 4: pytorch calculates gradients for all weights simultaniusly, which weights contributed to the mistae and by how much
        optimiser.step()   # step 5: update weights, nudge all weights in the direction that reduces loss
        total_loss+= loss.item()
        predicted = logits.argmax(dim=1)        # highest score per image
        correct += (predicted == labels).sum().item()       # count all of the true predictions
        total_images+= labels.size(0)       # number if images in the batch



        if (batch_idx +1) %1000 ==0:        # every 1000 batches
            elapsed = time.time() - start_time
            print(f" Epoch {epoch} | "
                  f" batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Elapsed: {elapsed:0.f}s"
                  )

    avg_loss = total_loss/len(loader)       # total_loss divided by number of batches = average loss per batch
    train_accuracy = correct / total_images *100        # precentages of images calculated correctly
    epoch_time = time.time() - start_time
    print(f" Epoch {epoch} complete | "
          f"avg loss : {avg_loss:.4f} | "
          f" train accuracy {train_accuracy:1.f} % | "
          f"time: {epoch_time:.0f} s")

    return avg_loss

# evaluating on validation set
def evaluate(model, loader, criterion, device):     # called after each each training epoch to measure real progress, returns (val_loss,  val_accuracy)
    model.eval()        # switch to evaluation mode
    total_loss = 0.0
    correct = 0
    total_images = 0
    with torch.no_grad():       # no gradients tracking needed
        for images, gps, labels in loader:
            images = images.to(device)
            gps = gps.to(device)
            labels = labels.to(device)
            logits = model(images, gps)     #forward pass only
            loss = criterion(logits, labels)
            total_loss+= loss.item()
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total_images += labels.size(0)


    avg_loss = total_loss/ len(loader)
    val_accuracy = correct/ total_images *100
    return avg_loss, val_accuracy

def train():            # full training pipeline, phase 1: backbone prozen, only classifier trains, phase 2: full network fine tuning, saves best model to checkpoints/best_phase1/2
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] using device: {device}")
    print("[Train] Loading data..")
    train_loader = get_train_loader( batch_size = BATCH_SIZE, num_workers= NUM_WORKERS)         # create training data loader from dataset, shuffles data, applies augmentations
    val_loader = get_val_loader(batch_size = BATCH_SIZE, num_workers= NUM_WORKERS)      # creates validation loader, no shuffle, no augmentations

    # building model
    print("[Train] Building model..")
    model = get_model(pretrained = True)        # download ImageNet weighs
    model = model.to(device) #moves model to cpu or gpu

    # loss function
    class_counts = [2278852, 84589, 10721, 4577] # approximate count of lanes
    total_count = sum(class_counts)
    class_weights = torch.tensor([total_count/ (N_CLASSES* count ) for count in class_counts ], dtype = torch.float32).to(device)       # calculate weights for each class, classes with fewer examples get higher weights so the modelpays more attention to rare classes
    criterion = nn.CrossEntropyLoss(weight = class_weights)     # loss function used during training


    # phase 1 : frozen backbone (warm up the classifier), goal: training only the final classifier +gps, backbone forzen (pretrained weights unchanged)
    print("\n"+ "="*50)
    print("Phase 1 - frozen backbone")
    print("="*50)
    freeze_backbone(model)
    optimiser_phase1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = LR_PHASE1)
    best_val_accuracy_p1 = 0.0
    for epoch in range(1, EPOCHS_PHASE1+1):
        print(f"\nPhase1 | epoch {epoch}/{EPOCHS_PHASE1}")
        train_loss = train_one_epoch(model, train_loader, optimiser_phase1, criterion, device, epoch)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(f" Val loss: {val_loss:,4f} | "
              f"val accuracy: {val_accuracy:.2f} %")
        if val_accuracy > best_val_accuracy_p1:
            best_val_accuracy_p1 = val_accuracy
            save_path = CHECKPOINT_DIR / "best_pahse1.pth"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_accuracy,
                "phase": 1,
            }, save_path
            )

            print(f" saved best phase model 1" 
                  f"(cal_acc= {val_accuracy:.2f}%")
        print(f"\nPhase 1 complete. best val accuracy: "
              f"{best_val_accuracy_p1:.2f}")



        # phase 2: full fine tuning (whoole network train), goal: gently fine tune the entire network, backbone now unfreeze and slowly adapts to NZ roads
        print("\n" +"="*50)
        print("phase 2 - full fine tuning")
        print("="*50)


        # load best phase 1 weights before starting phase 2
        best_phase1_path = CHECKPOINT_DIR / "best_phase1.pth"
        checkpoint = torch.load(best_phase1_path, map_location= device)
        model.load_state_dict(checkpoint["model"])
        print(f"[Train] Loaded best phase 1 weights")

        # start phase 2 from the best phase 1 model

        optimiser_phase2 = torch.optim.Adam(model.parameters(), lr = LR_PHASE2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser_phase2, mode="max", factor= 0.5, patience = 3, verbose = True)

        best_val_accuracy_p2 = 0.0
        no_improve_count = 0 # count how many epoch without improvment
        for epoch in range(1, EPOCHS_PHASE2+1):
            print(f"\nphase 2 | epoch {epoch}/{EPOCHS_PHASE2}")
            train_loss = train_one_epoch(model, train_loader, optimiser_phase2, criterion, device, epoch)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            print(f" val loss: {val_loss:.4f} | "
                  f" val accuracy: {val_accuracy:.2f} %")
            scheduler.step(val_accuracy)        # update learning rate schedualer
            # pass current val_accuracy to schedualer if accuracy hasnt improved for 3 epochs, scedualer automatically reduce LR by 50%

            # check if best model
            if val_accuracy> best_val_accuracy_p2:
                best_val_accuracy_p2 = val_accuracy
                no_improve_count = 0

                save_path = CHECKPOINT_DIR / "best_phase2.pth"
                torch.save({ "model": model.state_dict(), "epoch": epoch, "val_acc": val_accuracy, "phase": 2}, save_path)
                print(f" saved best phase 2 model "
                      f" val_acc = {val_accuracy:.2f}")
            else:
                no_improve_count+=1         # if no improvement this epoch
                print(f" No improvement"
                      f"({no_improve_count}/{EARLY_STOP_PATIENCE})")

            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"\n Early stopping triggered - "
                      f" no improvement for {EARLY_STOP_PATIENCE} epochs")
                print(f"best phase 2 val accuracy: "
                      f"{best_val_accuracy_p2:.2f}%")
                break

            print(f"\n TRaining complete!")
            print(f"best phase 1 accuracy: {best_val_accuracy_p1:.2f}%")
            print(f" best phase 2 accuracy: {best_val_accuracy_p2:.2f}")
            print(f" best model saved tp: {CHECKPOINT_DIR / 'best_phase2.pth'}")





def quick_test():
    print(f"QUICK TEST - 2 btaches only")
    print("="*40)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    train_loader = get_train_loader(batch_size = 4, num_workers=0)
    val_loader = get_val_loader(batch_size =4, num_workers=0)

    model = get_model(pretrained = False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = 1e-3)
    model.train()
    for batch_idx, (images,gps, labels) in enumerate(train_loader):
        images = images.to(device)
        gps = gps.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        logits = model(images,gps)
        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        print(f" batch {batch_idx+1} | loss: {loss.item():.4f}")
        if batch_idx>=1:
            break


    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f" val loss: {val_loss:.4f} | val acc: {val_acc:.2f} %")
    print("\n quick test passed - training loop work correctly! ")



if __name__ == "__main__":
    print("GEOSOLVE lane detection- training")
    print("="*50)
    quick_test()
    # train()

























































