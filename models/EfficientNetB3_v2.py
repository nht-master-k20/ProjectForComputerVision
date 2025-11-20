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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from scripts.ISICDataset import ISICDataset

# --- 1. CONFIGURATION ---
os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi987a9e46da628dbdb4a22949054afa24"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3_v2")


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
        "version": "v2_Focal_Sampler", "model": "tf_efficientnet_b3.ns_jft_in1k", "mode": mode,
        "image_size": image_size, "batch_size": batch_size, "epochs": epochs,
        "optimizer": "AdamW", "lr": lr, "device": str(device),
        "loss_function": "FocalLoss", "sampler": "WeightedRandomSampler"
    }
    if class_weights is not None:
        params.update({"cw_benign": float(class_weights[0]), "cw_malignant": float(class_weights[1])})
    mlflow.log_params(params)


def log_epoch_metrics(epoch, train_loss, val_loss, val_f1, current_lr, detailed_metrics=None):
    metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_f1_macro": val_f1, "lr": current_lr}
    if detailed_metrics: metrics.update(detailed_metrics)
    mlflow.log_metrics(metrics, step=epoch)


def calculate_metrics(y_true, y_pred, prefix="val"):
    return {
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        f"{prefix}_f1_malignant": f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_recall_malignant": recall_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred)
    }


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


def validate(model, loader, criterion, show_report=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    if show_report:
        print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'], digits=4,
                                    zero_division=0))
    return total_loss / max(1, len(loader)), np.array(all_labels), np.array(all_preds)


# --- 4. MAIN TRAIN ---
def train(mode='processed', image_size=300, batch_size=32, epochs=10, base_lr=1e-3):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device} | Version: v2 (Focal + Sampler)")

    # A. Load Data
    CSV_DIR = 'dataset_splits'
    prefix = "processed" if mode == 'processed' else "raw"
    train_path = os.path.join(CSV_DIR, f'{prefix}_train.csv')
    val_path = os.path.join(CSV_DIR, f'{prefix}_val.csv')
    test_path = os.path.join(CSV_DIR, f'{prefix}_test.csv')
    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p): raise FileNotFoundError(f"❌ Missing: {p}")

    train_df, val_df, test_df = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)
    print(f"📊 Stats: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # B. Sampler & Weights
    y_train = train_df['malignant'].values.astype(int)
    unique_classes = np.unique(y_train)
    sampler = None
    class_weights_tensor = None
    class_weights = None

    if len(unique_classes) < 2:
        print("⚠️ Only 1 class. Using Default Weights.")
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

    # C. DataLoaders
    train_loader = DataLoader(ISICDataset(train_df, img_size=image_size), batch_size=batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # D. Model
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # E. Run
    start_mlflow_run(f"EffB3_{mode}_v2")
    log_training_params(mode, image_size, batch_size, epochs, 5, len(train_df), len(val_df), len(test_df), device,
                        base_lr, 0.01, class_weights)

    best_f1, patience = -1, 0
    model_path = f"checkpoints/best_effb3_{mode}_v2.pth"
    os.makedirs("checkpoints", exist_ok=True)

    try:
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            print(f"🚀 Epoch [{epoch + 1}/{epochs}] | LR: {lr:.6f}")
            scheduler.step()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_labels, val_preds = validate(model, val_loader, criterion)

            metrics = calculate_metrics(val_labels, val_preds, prefix="val")
            log_epoch_metrics(epoch, train_loss, val_loss, metrics['val_f1_macro'], lr, metrics)
            print(f"✅ Train: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1 Mal: {metrics['val_f1_malignant']:.4f}")

            # v2: Early stopping on Malignant F1
            if metrics['val_f1_malignant'] > best_f1 * 1.005:
                best_f1, patience = metrics['val_f1_malignant'], 0
                torch.save(model.state_dict(), model_path)
                print(f"💾 Saved: {model_path}")
            else:
                patience += 1
                if patience >= 5: print("🛑 Early Stopping"); break

        print(f"\n🧪 Testing Best Model...")
        model.load_state_dict(torch.load(model_path))
        test_loss, test_labels, test_preds = validate(model, test_loader, criterion, show_report=True)
        test_metrics = calculate_metrics(test_labels, test_preds, prefix="test")
        mlflow.log_metrics({**test_metrics, "test_loss": test_loss})

    finally:
        mlflow.end_run()