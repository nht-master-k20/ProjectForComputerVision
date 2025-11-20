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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, \
    confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scripts.ISICDataset import ISICDataset

# --- 1. CONFIGURATION ---
os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi987a9e46da628dbdb4a22949054afa24"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3_v3")


# --- 2. HELPERS ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma, self.weight, self.reduction = gamma, weight, reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def start_mlflow_run(run_name): return mlflow.start_run(run_name=run_name)


def log_training_params(mode, image_size, batch_size, epochs, early_stop_patience,
                        train_size, val_size, test_size, device, lr, weight_decay, class_weights=None):
    params = {
        "version": "v3_Bias_Threshold", "model": "tf_efficientnet_b3.ns_jft_in1k", "mode": mode,
        "image_size": image_size, "batch_size": batch_size, "epochs": epochs,
        "optimizer": "AdamW", "lr": lr, "device": str(device),
        "loss_function": "FocalLoss", "sampler": "WeightedRandomSampler", "technique": "Bias+Threshold"
    }
    if class_weights is not None:
        params.update({"cw_benign": float(class_weights[0]), "cw_malignant": float(class_weights[1])})
    mlflow.log_params(params)


def log_epoch_metrics(epoch, train_loss, val_loss, val_f1, current_lr, detailed_metrics=None):
    metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_f1_macro": val_f1, "lr": current_lr}
    if detailed_metrics: metrics.update(detailed_metrics)
    mlflow.log_metrics(metrics, step=epoch)


def find_best_threshold(y_true, y_probs):
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.01, 0.90, 0.01):
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0)
        if f1 > best_f1: best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def calculate_metrics(y_true, y_probs, threshold=0.5, prefix="val"):
    y_pred = (y_probs >= threshold).astype(int)
    return {
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        f"{prefix}_f1_malignant": f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_recall_malignant": recall_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_threshold": threshold
    }


def initialize_bias(model, device):
    prior = 0.01
    bias_value = -np.log((1 - prior) / prior)
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        with torch.no_grad(): model.classifier.bias.data.fill_(bias_value)
        print(f"🔧 Bias Initialized: {bias_value:.4f}")
    model.to(device)
    return model


# --- 3. LOOPS ---
def train_one_epoch(model, loader, optimizer, criterion, scaler, gradient_clip=1.0):
    model.train()
    total_loss, count = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item();
        count += 1
    return total_loss / max(1, count)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # v3: Get PROBABILITIES
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / max(1, len(loader)), np.array(all_labels), np.array(all_probs)


# --- 4. MAIN TRAIN ---
def train(mode='processed', image_size=300, batch_size=32, epochs=10, base_lr=1e-3):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device} | Version: v3 (Bias + Dynamic Threshold)")

    CSV_DIR = 'dataset_splits'
    prefix = "processed" if mode == 'processed' else "raw"

    train_path = os.path.join(CSV_DIR, f'{prefix}_train.csv')
    val_path = os.path.join(CSV_DIR, f'{prefix}_val.csv')
    test_path = os.path.join(CSV_DIR, f'{prefix}_test.csv')

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p): raise FileNotFoundError(f"❌ Missing: {p}")

    train_df, val_df, test_df = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)
    print(f"📊 Stats: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Sampler
    y_train = train_df['malignant'].values.astype(int)
    unique_classes = np.unique(y_train)
    sampler = None
    class_weights_tensor = None
    class_weights = None

    if len(unique_classes) < 2:
        print("⚠️ Only 1 class. Defaulting.")
        class_weights_tensor = None
    else:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        print(f"⚖️ Class Weights: {class_weights}")
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in unique_classes])
        weight_per_class = 1. / class_sample_count
        samples_weight = np.array([weight_per_class[t] for t in y_train])
        sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weight), len(samples_weight), replacement=True)
        print("✅ WeightedRandomSampler activated.")

    # Loaders
    train_loader = DataLoader(ISICDataset(train_df, img_size=image_size), batch_size=batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # Model (Bias Init)
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2)
    model = initialize_bias(model, device)  # v3 specific

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Run
    start_mlflow_run(f"EffB3_{mode}_v3")
    log_training_params(mode, image_size, batch_size, epochs, 5, len(train_df), len(val_df), len(test_df), device,
                        base_lr, 0.01, class_weights)

    best_f1, patience, best_thresh_val = -1, 0, 0.5
    model_path = f"checkpoints/best_effb3_{mode}_v3.pth"
    os.makedirs("checkpoints", exist_ok=True)

    try:
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            print(f"🚀 Epoch [{epoch + 1}/{epochs}] | LR: {lr:.6f}")
            scheduler.step()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_labels, val_probs = validate(model, val_loader, criterion)

            # v3: Dynamic Threshold
            optimal_threshold, best_val_f1_mal = find_best_threshold(val_labels, val_probs)
            metrics = calculate_metrics(val_labels, val_probs, threshold=optimal_threshold, prefix="val")

            log_epoch_metrics(epoch, train_loss, val_loss, metrics['val_f1_macro'], lr, metrics)
            print(
                f"✅ Train: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Thresh: {optimal_threshold:.2f} | F1 Mal: {best_val_f1_mal:.4f}")

            if best_val_f1_mal > best_f1 * 1.005:
                best_f1, best_thresh_val, patience = best_val_f1_mal, optimal_threshold, 0
                torch.save({'state_dict': model.state_dict(), 'threshold': best_thresh_val}, model_path)
                print(f"💾 Saved with thresh {best_thresh_val:.2f}: {model_path}")
            else:
                patience += 1
                if patience >= 5: print("🛑 Early Stopping"); break

        print(f"\n🧪 Testing Best Model (Thresh={best_thresh_val:.2f})...")
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        loaded_thresh = ckpt.get('threshold', 0.5)

        test_loss, test_labels, test_probs = validate(model, test_loader, criterion)
        test_metrics = calculate_metrics(test_labels, test_probs, threshold=loaded_thresh, prefix="test")

        print(classification_report(test_labels, (test_probs >= loaded_thresh).astype(int),
                                    target_names=['Benign', 'Malignant'], digits=4))
        mlflow.log_metrics({**test_metrics, "test_loss": test_loss})

    finally:
        mlflow.end_run()