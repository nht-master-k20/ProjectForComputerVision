import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
# [UPDATE] Added roc_auc_score
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             classification_report, roc_auc_score)
from scripts.ISICDataset import ISICDataset


# --- 1. CUSTOM LOSS: BINARY FOCAL LOSS ---
class BinaryFocalLoss(nn.Module):
    """
    Loss function specialized for Binary Classification.
    Combines BCEWithLogitsLoss and Focal mechanism to focus on hard samples.
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        # Alpha balancing: if target=1 use alpha, else use (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# --- 2. HELPERS & LOGGING ---
def start_mlflow_run(run_name):
    return mlflow.start_run(run_name=run_name)


def log_training_params(image_size, batch_size, epochs, train_len, val_len, test_len, device, lr):
    params = {
        "version": "v5_Binary_pAUC_Optimized",
        "model": "tf_efficientnet_b3.ns_jft_in1k",
        "architecture": "Binary (1 Output Node)",
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "AdamW",
        "lr": lr,
        "train_size": train_len,
        "val_size": val_len,
        "test_size": test_len,
        "loss_function": "BinaryFocalLoss",
        "sampler": "WeightedRandomSampler (50/50 Balance)",
        "metric_focus": "pAUC (max_fpr=0.2)",  # [UPDATE] Log focus
        "training_type": "GPU" if torch.cuda.is_available() else "CPU"
    }
    mlflow.log_params(params)


def log_metrics(prefix, metrics, step=None):
    mlflow.log_metrics({k: v for k, v in metrics.items()}, step=step)


def find_best_threshold(y_true, y_probs):
    """Find threshold that maximizes F1-Score (for reporting purposes)"""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.01, 0.90, 0.01):
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def calculate_metrics(y_true, y_probs, threshold=0.5, prefix="val"):
    """
    [UPDATE] Added AUC and pAUC calculation
    """
    y_pred = (y_probs >= threshold).astype(int)

    # 1. Calculate AUC (Full)
    try:
        auc_score = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_score = 0.0

    # 2. Calculate pAUC (Partial AUC, max_fpr=0.2)
    # Important for medical diagnosis to keep False Positives low
    try:
        pauc_score = roc_auc_score(y_true, y_probs, max_fpr=0.2)
    except ValueError:
        pauc_score = 0.0

    return {
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        f"{prefix}_f1_malignant": f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_recall_malignant": recall_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_precision_malignant": precision_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_auc": auc_score,  # [NEW]
        f"{prefix}_pauc_0.2": pauc_score,  # [NEW]
        f"{prefix}_threshold": threshold
    }


def initialize_bias(model, device):
    """Initialize bias to output low probability (prior=0.01) to stabilize training start."""
    prior = 0.01
    bias_value = -np.log((1 - prior) / prior)

    if hasattr(model, 'classifier'):
        layer = model.classifier
    elif hasattr(model, 'fc'):
        layer = model.fc
    else:
        return model

    if isinstance(layer, nn.Linear):
        with torch.no_grad():
            layer.bias.data.fill_(bias_value)
            print(f"🔧 Bias Initialized to {bias_value:.4f}")

    model.to(device)
    return model


# --- 3. TRAINING LOOPS ---
def train_one_epoch(model, loader, optimizer, criterion, scaler, gradient_clip=1.0):
    model.train()
    total_loss, count = 0.0, 0

    for imgs, labels in loader:
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        labels = labels.float().unsqueeze(1)  # Shape (N, 1)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.cuda(non_blocking=True)
            labels_orig = labels.cuda(non_blocking=True)
            labels_float = labels_orig.float().unsqueeze(1)

            outputs = model(imgs)
            loss = criterion(outputs, labels_float)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels_orig.cpu().numpy())

    return total_loss / max(1, len(loader)), np.array(all_labels), np.array(all_probs)


# --- 4. MAIN EXECUTION ---
def train(mode='processed', image_size=300, batch_size=32, epochs=10, base_lr=1e-3):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device} | Version: v5 (pAUC Optimized)")

    # A. MLflow Config
    os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "dapif865faf65e4f29f9f213de9b6f2ffa3c"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3_v5_pAUC")

    # B. Load Data
    CSV_DIR = 'dataset_splits'
    prefix = "processed" if mode == 'processed' else "raw"
    paths = {
        'train': os.path.join(CSV_DIR, f'{prefix}_train.csv'),
        'val': os.path.join(CSV_DIR, f'{prefix}_val.csv'),
        'test': os.path.join(CSV_DIR, f'{prefix}_test.csv')
    }

    train_df = pd.read_csv(paths['train'])
    val_df = pd.read_csv(paths['val'])
    test_df = pd.read_csv(paths['test'])
    print(f"📊 Stats: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # C. Sampler (Weighted 50/50)
    y_train = train_df['malignant'].values.astype(int)
    class_counts = np.bincount(y_train)
    sample_weights = 1. / class_counts[y_train]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)
    print(f"⚖️ WeightedRandomSampler Activated")

    # D. DataLoaders
    train_loader = DataLoader(ISICDataset(train_df, img_size=image_size), batch_size=batch_size, sampler=sampler,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # E. Model Setup
    print("🏗️ Creating Model: EfficientNet-B3 (Binary)")
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=1)
    model = initialize_bias(model, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    criterion = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # F. Training Loop
    mlflow_run = start_mlflow_run(f"EffB3_pAUC_v5")
    log_training_params(image_size, batch_size, epochs, len(train_df), len(val_df), len(test_df), device, base_lr)

    best_pauc = -1  # [UPDATE] Track pAUC instead of F1
    model_path = f"checkpoints/best_effb3_{mode}_v5_pauc.pth"
    os.makedirs("checkpoints", exist_ok=True)

    print(f"🚀 Training {epochs} epochs (Optimizing for pAUC)...")

    try:
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            # 1. Train
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)

            # 2. Val
            val_loss, val_labels, val_probs = validate(model, val_loader, criterion)

            # 3. Calculate Metrics (Including pAUC)
            optimal_threshold, best_val_f1 = find_best_threshold(val_labels, val_probs)
            metrics = calculate_metrics(val_labels, val_probs, threshold=optimal_threshold, prefix="val")

            # Extract current pAUC
            current_pauc = metrics['val_pauc_0.2']

            # 4. Log
            log_metrics("val", {**metrics, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}, step=epoch)

            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {train_loss:.4f} | "
                  f"pAUC(0.2): {current_pauc:.4f} | AUC: {metrics['val_auc']:.4f} | "
                  f"F1 Mal: {best_val_f1:.4f}")

            # 5. Save Best Model based on pAUC
            if current_pauc > best_pauc:
                best_pauc = current_pauc
                best_thresh_val = optimal_threshold
                torch.save({
                    'state_dict': model.state_dict(),
                    'threshold': best_thresh_val,
                    'epoch': epoch,
                    'pauc_score': best_pauc,  # Save pAUC
                    'f1_score': best_val_f1
                }, model_path)
                print(f"   🔥 Saved New Best (pAUC: {best_pauc:.4f})")

        # G. Final Test
        print(f"\n🧪 Testing Best Model (Best pAUC)...")
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt['state_dict'])
            loaded_thresh = ckpt.get('threshold', 0.5)
            best_epoch = ckpt.get('epoch', 'Unknown')
            best_metric = ckpt.get('pauc_score', 0.0)
            print(f"   📂 Loaded Checkpoint (Epoch {best_epoch} | pAUC={best_metric:.4f})")
        else:
            loaded_thresh = 0.5
            print("   ⚠️ Warning: Checkpoint not found.")

        test_loss, test_labels, test_probs = validate(model, test_loader, criterion)
        # Recalculate best threshold for Test set specifically for report (optional) or use Loaded
        test_metrics = calculate_metrics(test_labels, test_probs, threshold=loaded_thresh, prefix="test")

        print("\n" + "=" * 40)
        print(f"FINAL REPORT (Model Selected by pAUC)")
        print(f"pAUC (0.2): {test_metrics['test_pauc_0.2']:.4f}")
        print(f"AUC Full  : {test_metrics['test_auc']:.4f}")
        print(classification_report(test_labels, (test_probs >= loaded_thresh).astype(int),
                                    target_names=['Benign', 'Malignant'], digits=4))
        print("=" * 40)

        log_metrics("test", {**test_metrics, "test_loss": test_loss})

    finally:
        mlflow.end_run()
        print("🔚 Done v5.")