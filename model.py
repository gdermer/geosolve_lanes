import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


from config import N_CLASSES, MODEL_NAME

class GeoSolveLaneModel(nn.Module):
    def __init__(self, n_classes = N_CLASSES, pretrained = True):
        super().__init__()
        self.n_classes = n_classes

        self.backbone = timm.create_model(
        MODEL_NAME, pretrained = pretrained, num_classes = 0, global_pool = "avg",)

        self.backbone_features = self.backbone.num_features
        print(f"[Model] Backbone: {MODEL_NAME}")
        print(f"[Model] backbone output features: {self.backbone_features}")

        # gps processor

        self.gps_processor = nn.Sequential( nn.Linear(5,32),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.2),
                                            nn.Linear(32,64),
                                            nn.ReLU()
                                            )
        print(f"[Model] GPS processor: 5--> 64 features")

        # final classifier:
        combined_features = self.backbone_features +64

        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, n_classes) #produce the logits
            ,)

        print(f"[Model] Classifier: {combined_features} -> {n_classes} classes")
        print(f"[Model] Output classes: {n_classes}")


    def forward(self, image, gps_features):   # describe what happens when data flows through the model pytorch calls it automatically when you call model(image, gps)

        image_features = self.backbone(image)
        gps_out = self.gps_processor(gps_features)
        combined = torch.cat([image_features, gps_out], dim=1)
        logits = self.classifier(combined)

        return logits

    def predict(self,images, gps_features, threshold = 0.90):        # return lane code + confidence below 90% confident flagg the image as review for hman
        self.eval()    # switch model to evaluation mode
        with torch.no_grad():
            logits = self.forward(images, gps_features)
            probabilities = F.softmax(logits, dim=1)
            confidence , predicted_class = probabilities.max(dim=1)    # return max probability and max prediction
        results = []
        for conf, cls_idx in zip(confidence, predicted_class):
            conf = conf.item()
            cls_idx = cls_idx.item()

            from config import IDX_TO_LANE
            lane_code = IDX_TO_LANE[cls_idx]
            needs_review = conf< threshold

            if needs_review:
                lane_code = "REVIEW"

            results.append({
                "lane": lane_code,
                "confidence": round(conf, 3),
                "needs_review": needs_review,
            })
        return results      # return the list of dictionaries - one per image in the batch

def get_model(pretrained = True):
    model = GeoSolveLaneModel(n_classes = N_CLASSES, pretrained= pretrained)
    return model


def load_trained_model(checkpoint_path, device = "cpu"):        # loads a trained model form a checkpoint file
    model = get_model(pretrained=False)     # create model with random weights first
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)            # moves model to cpu
    model.eval()   # switch to evaluation mode
    print(f"[Model] Loaded trained model from {checkpoint_path}")
    return model

def count_parameters(model):        # counts how many weitghs the model has
    total = sum(p.numel() for p in model.parameters())      #p.numel = number of wieghts in one weight tensor ,  model.parameter = generator of all weight tensors in the model
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"[Model] total parameters: {total:,}")
    print(f"[Model] trainable parameters: {trainable:,}")
    return total, trainable

def freeze_backbone(model):         # freeze efficientNet backbone weights used at the start of phase 1 training
    for param in model.backbone.parameters():       # fro all the backnbone weigths tensor
        param.requires_grad = False
    print("[Model] Backbone FROZEN - only classifier trains")
    count_parameters(model)
    return model    # print show many weights are now trainable

def unfreeze_backbone(model):       # unfreeze backbone for phase 2
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("[Model] backbone UNFROZEN- full fine tuining active")
    count_parameters(model)
    return model


# main
if __name__ == "__main__":
    print("Testing model.py..")
    print("="*40)

    model  = get_model(pretrained=False)        # build model
    count_parameters(model)
    fake_images = torch.randn(4,3,224,224)
    fake_gps = torch.randn(4,5)
    print(f"\nInput shapes:")
    print(f"images: {fake_images.shape}")
    print(f"GPS  {fake_gps.shape}")

    model.eval() # run forward pass
    with torch.no_grad():
        output = model(fake_images, fake_gps)       # automatically calss forward

    print(f" output shape: {output.shape}")

    from config import IDX_TO_LANE
    print(f"\n sources for first image:")
    for idx, score in enumerate(output[0]):
        lane = IDX_TO_LANE[idx]
        print(f" {lane:.4}: {score.item():.4f}")

    print(f"\n predicted class: {output[0].argmax().item()}")
    print("\n --- freeze/ unfreeze test--")
    freeze_backbone(model)
    unfreeze_backbone(model)

    print("\n predicte test --")
    results = model.predict(fake_images, fake_gps, threshold = 0.90)
    for i, r in enumerate(results):         # enumerates gives index i and vaslue r together
        print(f" Image {i+1}: {r}")

    print("\nmodel.py works correctly :)")





































