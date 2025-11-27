# Chiến thuật: Sử dụng Cross Entropy Loss kết hợp Weighted Random Sampler (Để batch luôn cân bằng 50/50).
# Dữ liệu: Bật Online Augmentation (is_train=True).
# Metric: Tối ưu hóa theo pAUC (0.01).
# Lưu ý quan trọng: Khi đã dùng WeightedRandomSampler, ta không cần truyền weight vào hàm Loss nữa (vì dữ liệu vào model đã được cân bằng rồi).

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score
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


# --- 1. LOGGING ---
def log_training_params(version, batch_size, epochs, lr):
    """Log các tham số đặc trưng của V2"""
    params = {
        "version": version,
        "model": "EfficientNet-B3",
        "loss_function": "CrossEntropyLoss",  # V2 dùng CE bình thường
        "sampler": "WeightedRandomSampler (Balance 50/50)",  # Điểm nhấn của V2
        "augmentation": "Online (Albumentations)",
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "metric_target": "pAUC (0.01)"
    }
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
        "recall_malignant": recall_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            # Lấy xác suất lớp 1
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / max(1, len(loader)), np.array(all_labels), np.array(all_probs)


# --- 3. MAIN V2 ---
def train(image_size=300, batch_size=32, epochs=10, base_lr=1e-3):
    seed_everything(42)  # Cố định hạt giống
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running V2 (Sampler + CE) on {device}...")

    # MLflow Setup
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/SkinDisease_Experiment")  # Dùng chung Experiment với V1 để dễ so sánh

    # Paths
    CSV_DIR = 'dataset_splits'
    train_df = pd.read_csv(f'{CSV_DIR}/processed_train.csv')
    val_df = pd.read_csv(f'{CSV_DIR}/processed_val.csv')
    test_df = pd.read_csv(f'{CSV_DIR}/processed_test.csv')
    print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # --- CẤU HÌNH SAMPLER (ĐIỂM KHÁC BIỆT CHÍNH CỦA V2) ---
    print("⚖️ Configuring WeightedRandomSampler...")
    y_train = train_df['malignant'].values.astype(int)
    class_counts = np.bincount(y_train)
    # Trọng số nghịch đảo với số lượng mẫu (Lớp ít -> Trọng số cao)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    # --------------------------------------------------------

    # Loaders
    # LƯU Ý: Khi dùng Sampler thì shuffle phải là False
    train_loader = DataLoader(
        ISICDataset(train_df, image_size, is_train=True),  # Online Augmentation ON
        batch_size=batch_size,
        sampler=sampler,  # Sampler ON
        shuffle=False,  # Bắt buộc False khi có Sampler
        num_workers=8, pin_memory=True
    )

    val_loader = DataLoader(ISICDataset(val_df, image_size, is_train=False),
                            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, image_size, is_train=False),
                             batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    # Loss V2: CrossEntropyLoss KHÔNG weight (Vì Sampler đã cân bằng batch rồi)
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Run MLflow
    with mlflow.start_run(run_name="V2_Sampler_CE"):
        log_training_params("V2_Sampler_CE", batch_size, epochs, base_lr)

        best_pauc = -1
        model_path = "checkpoints/best_v2.pth"
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

            # Log
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()}, step=epoch)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            print(f"Epoch [{epoch + 1}/{epochs}] | pAUC: {current_pauc:.4f} | AUC: {metrics['auc']:.4f}")

            # Save Best by pAUC
            if current_pauc > best_pauc:
                best_pauc = current_pauc
                torch.save(model.state_dict(), model_path)
                print(f"  🔥 Saved Best Model (pAUC: {best_pauc:.4f})")

        # Final Test
        print("\n🧪 Testing Best Model V2...")
        model.load_state_dict(torch.load(model_path))
        test_loss, test_labels, test_probs = validate(model, test_loader, criterion)
        test_metrics = calculate_metrics(test_labels, test_probs)

        print(f"🏆 FINAL TEST V2 pAUC: {test_metrics['pauc_0.01']:.4f}")
        print(classification_report(test_labels, (test_probs >= 0.5).astype(int), target_names=['Benign', 'Malignant']))

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == '__main__':
    train()