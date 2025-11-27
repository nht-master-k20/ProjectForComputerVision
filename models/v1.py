# Metric: Đánh giá bằng pAUC (max_fpr=0.01) thay vì F1.
# Data: Sử dụng Online Augmentation (is_train=True cho tập train).
# Sampler: KHÔNG dùng WeightedRandomSampler (để Shuffle=True mặc định) -> Đây là Baseline thuần.
# Loss: CrossEntropyLoss (Có thể dùng class weights nhẹ để model hội tụ, nhưng không dùng Focal Loss).
# Reproducibility: Thêm hàm seed_everything để đảm bảo chuỗi Augmentation giống nhau cho các Version sau.

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from scripts.ISICDataset2 import ISICDataset


# --- 0. SEED CONTROL ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 1. LOGGING (CLEANED) ---
def log_training_params(version, batch_size, epochs, lr, class_weights=None):
    """Chỉ log những tham số quan trọng thay đổi giữa các version"""
    params = {
        "version": version,
        "model": "EfficientNet-B3",
        "loss_function": "CrossEntropyLoss",
        "sampler": "OFF (Shuffle=True)",
        "augmentation": "Online (Albumentations)",
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
    }
    if class_weights is not None:
        params["class_weights"] = f"Benign:{class_weights[0]:.2f} | Malignant:{class_weights[1]:.2f}"
    mlflow.log_params(params)


def calculate_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    try:
        pauc = roc_auc_score(y_true, y_probs, max_fpr=0.01)
        auc = roc_auc_score(y_true, y_probs)
    except:
        pauc, auc = 0.0, 0.0

    return {
        "pauc_0.01": pauc,
        "auc": auc,
        "f1_malignant": f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred)
    }


# --- 2. TRAINING CORE ---
def train_one_epoch(model, loader, optimizer, criterion, scaler):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Hardcode clip=1.0 cho gọn
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
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Lấy xác suất lớp 1 (Ác tính)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / max(1, len(loader)), np.array(all_labels), np.array(all_probs)


# --- 3. MAIN (SIMPLIFIED) ---
def train(img_size=300, batch_size=32, epochs=10, base_lr=1e-3):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running V1 Baseline on {device}...")

    # MLflow Setup
    mlflow.set_tracking_uri("databricks")
    # Thay đổi path experiment của bạn nếu cần
    mlflow.set_experiment("/SkinDisease_Experiment")

    # Paths (Mặc định dùng dataset_splits)
    CSV_DIR = 'dataset_splits'
    train_df = pd.read_csv(f'{CSV_DIR}/processed_train.csv')
    val_df = pd.read_csv(f'{CSV_DIR}/processed_val.csv')
    test_df = pd.read_csv(f'{CSV_DIR}/processed_test.csv')
    print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Loaders
    train_loader = DataLoader(ISICDataset(train_df, img_size, is_train=True),
                              batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, img_size, is_train=False),
                            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size, is_train=False),
                             batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Model Setup
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    # Class Weights
    y_train = train_df['malignant'].values
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to(device))

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Start Run
    with mlflow.start_run(run_name="V1_Baseline"):
        log_training_params("V1_Baseline", batch_size, epochs, base_lr, cw)

        best_pauc = -1
        model_path = "checkpoints/best_v1.pth"
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            # Train & Val
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_labels, val_probs = validate(model, val_loader, criterion)

            # Metrics
            metrics = calculate_metrics(val_labels, val_probs)
            current_pauc = metrics['pauc_0.01']

            # Log Metrics (Thêm prefix 'val_' tự động khi log)
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()}, step=epoch)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            print(f"Epoch [{epoch + 1}/{epochs}] | pAUC: {current_pauc:.4f} | AUC: {metrics['auc']:.4f}")

            if current_pauc > best_pauc:
                best_pauc = current_pauc
                torch.save(model.state_dict(), model_path)
                print(f"  🔥 Saved Best Model (pAUC: {best_pauc:.4f})")

        # Test
        print("\n🧪 Testing Best Model...")
        model.load_state_dict(torch.load(model_path))
        test_loss, test_labels, test_probs = validate(model, test_loader, criterion)
        test_metrics = calculate_metrics(test_labels, test_probs)

        print(f"🏆 FINAL TEST pAUC: {test_metrics['pauc_0.01']:.4f}")
        print(classification_report(test_labels, (test_probs >= 0.5).astype(int), target_names=['Benign', 'Malignant']))

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == '__main__':
    train()