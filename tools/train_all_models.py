#!/usr/bin/env python3
"""Train custom CNN from scratch, MobileNetV2 (ResNet transfer), and DINOv2 probe.
Exports ONNX models and evaluation metrics to web/models/ and data/.
"""
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CROPS_DIR = DATA_DIR / "crops"
MODELS_DIR = ROOT / "web" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Dataset definition
class CatDogDataset(Dataset):
    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = ROOT / entry["file"]
        image = Image.open(img_path).convert("RGB")
        label = 0 if entry["label"] == "cat" else 1
        if self.transform:
            image = self.transform(image)
        return image, label

manifest = json.loads((DATA_DIR / "manifest.json").read_text())
train_entries = [e for e in manifest if e["split"] == "train"]
val_entries = [e for e in manifest if e["split"] == "val"]
holdout_entries = [e for e in manifest if e["split"] == "holdout"]

print(f"Dataset: Train={len(train_entries)}, Val={len(val_entries)}, Holdout={len(holdout_entries)}")

# Common transforms
cnn_transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
cnn_transform_val = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

pretrained_transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
pretrained_transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# MODEL 1: Custom CNN (from scratch on 64x64)
# ==========================================
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv block 1: 64x64 -> 32x32
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv block 2: 32x32 -> 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv block 3: 16x16 -> 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        # Classifier
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu_fc(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_and_eval(name, model, train_loader, val_loader, holdout_loader, epochs=15, lr=1e-3, weight_decay=1e-4):
    print(f"\n--- Training {name} ---")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                pred = out.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    hold_acc = evaluate(holdout_loader)
    print(f"[{name}] Final Acc -> Train: {train_acc*100:.1f}%, Val: {val_acc*100:.1f}%, Holdout: {hold_acc*100:.1f}%")
    return {
        "train": round(train_acc * 100, 2),
        "val": round(val_acc * 100, 2),
        "holdout": round(hold_acc * 100, 2)
    }

metrics = {}

# Train Custom CNN
train_loader_cnn = DataLoader(CatDogDataset(train_entries, cnn_transform_train), batch_size=32, shuffle=True)
val_loader_cnn = DataLoader(CatDogDataset(val_entries, cnn_transform_val), batch_size=32, shuffle=False)
holdout_loader_cnn = DataLoader(CatDogDataset(holdout_entries, cnn_transform_val), batch_size=32, shuffle=False)

cnn = CustomCNN()
metrics["custom_cnn"] = train_and_eval("Custom CNN", cnn, train_loader_cnn, val_loader_cnn, holdout_loader_cnn, epochs=25, lr=1e-3)

# Export Custom CNN to ONNX
cnn.eval().cpu()
dummy_cnn = torch.randn(1, 3, 64, 64)
cnn_onnx_path = MODELS_DIR / "custom_cnn.onnx"
torch.onnx.export(
    cnn,
    dummy_cnn,
    str(cnn_onnx_path),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=14,
)
print(f"Exported Custom CNN -> {cnn_onnx_path} ({cnn_onnx_path.stat().st_size / 1024:.1f} KB)")

# ==========================================
# MODEL 2: Transfer Learning (MobileNetV2 / ResNet-style)
# ==========================================
# We use MobileNetV2 pretrained on ImageNet: lightweight, ultra-fast in browser ONNX Web (< 9 MB).
train_loader_pt = DataLoader(CatDogDataset(train_entries, pretrained_transform_train), batch_size=32, shuffle=True)
val_loader_pt = DataLoader(CatDogDataset(val_entries, pretrained_transform_val), batch_size=32, shuffle=False)
holdout_loader_pt = DataLoader(CatDogDataset(holdout_entries, pretrained_transform_val), batch_size=32, shuffle=False)

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in mobilenet.features.parameters():
    param.requires_grad = False
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)

metrics["transfer_mobilenet"] = train_and_eval("Transfer MobileNetV2", mobilenet, train_loader_pt, val_loader_pt, holdout_loader_pt, epochs=12, lr=1e-3)

# Export Transfer MobileNet to ONNX
mobilenet.eval().cpu()
dummy_pt = torch.randn(1, 3, 224, 224)
pt_onnx_path = MODELS_DIR / "transfer_mobilenet.onnx"
torch.onnx.export(
    mobilenet,
    dummy_pt,
    str(pt_onnx_path),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=14,
)
print(f"Exported Transfer MobileNet -> {pt_onnx_path} ({pt_onnx_path.stat().st_size / (1024*1024):.1f} MB)")

# ==========================================
# MODEL 3: Vision Foundation Probe (DINOv2 linear probe)
# ==========================================
# We load DINOv2-small (dinov2_vits14), extract CLS token embeddings (384-dim),
# train a linear probe layer, and export both the linear probe weights and pre-extracted embeddings.
print("\n--- Training Foundation Model Linear Probe (DINOv2 ViT-S/14) ---")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
dinov2.eval()

def extract_dinov2_features(entries):
    feats, labels = [], []
    with torch.no_grad():
        for entry in entries:
            img = Image.open(ROOT / entry["file"]).convert("RGB")
            tensor = pretrained_transform_val(img).unsqueeze(0).to(device)
            out = dinov2(tensor) # 384-dim cls embedding
            feats.append(out.cpu().squeeze(0))
            labels.append(0 if entry["label"] == "cat" else 1)
    return torch.stack(feats), torch.tensor(labels)

print("Extracting DINOv2 features for all dataset splits...")
train_dino_x, train_dino_y = extract_dinov2_features(train_entries)
val_dino_x, val_dino_y = extract_dinov2_features(val_entries)
holdout_dino_x, holdout_dino_y = extract_dinov2_features(holdout_entries)

# Linear Probe: Linear(384, 2)
probe = nn.Linear(384, 2).to(device)
probe_criterion = nn.CrossEntropyLoss()
probe_opt = optim.AdamW(probe.parameters(), lr=5e-3, weight_decay=1e-3)

for epoch in range(40):
    probe.train()
    probe_opt.zero_grad()
    preds = probe(train_dino_x.to(device))
    loss = probe_criterion(preds, train_dino_y.to(device))
    loss.backward()
    probe_opt.step()

probe.eval()
with torch.no_grad():
    train_acc = (probe(train_dino_x.to(device)).argmax(dim=1).cpu() == train_dino_y).float().mean().item()
    val_acc = (probe(val_dino_x.to(device)).argmax(dim=1).cpu() == val_dino_y).float().mean().item()
    hold_acc = (probe(holdout_dino_x.to(device)).argmax(dim=1).cpu() == holdout_dino_y).float().mean().item()

print(f"[DINOv2 Linear Probe] Final Acc -> Train: {train_acc*100:.1f}%, Val: {val_acc*100:.1f}%, Holdout: {hold_acc*100:.1f}%")
metrics["dinov2_probe"] = {
    "train": round(train_acc * 100, 2),
    "val": round(val_acc * 100, 2),
    "holdout": round(hold_acc * 100, 2)
}

# Export DINOv2 probe weights & holdout embeddings cache for instant 0ms evaluation in browser
probe_cpu = probe.cpu()
probe_data = {
    "weights": probe_cpu.weight.tolist(),
    "bias": probe_cpu.bias.tolist(),
    "feature_dim": 384,
}
(DATA_DIR / "dinov2_probe_model.json").write_text(json.dumps(probe_data, indent=2))
(MODELS_DIR / "dinov2_probe_model.json").write_text(json.dumps(probe_data, indent=2))

# Also export cached holdout embeddings map so the web UI can predict instantly on any holdout image without downloading 80MB ViT!
holdout_embeddings = {}
for entry, feat in zip(holdout_entries, holdout_dino_x):
    holdout_embeddings[entry["id"]] = feat.tolist()

(MODELS_DIR / "dinov2_holdout_embeddings.json").write_text(json.dumps(holdout_embeddings))
print(f"Exported DINOv2 probe model and {len(holdout_embeddings)} cached holdout embeddings.")

# Export consolidated model metrics
(MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
(DATA_DIR / "all_model_metrics.json").write_text(json.dumps(metrics, indent=2))
print("\nAll models trained and exported successfully!")
