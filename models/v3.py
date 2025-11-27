# Kiến trúc: Chuyển sang num_classes=1 (Binary Mode). Đây là cấu hình chuyên nghiệp cho bài toán 2 lớp, giúp việc tính toán Loss và Bias Init chính xác hơn so với để 2 output nodes.
# Loss Function: Sử dụng Binary Focal Loss.
# Vì đã dùng Sampler (cân bằng số lượng 50/50), Focal Loss ở đây đóng vai trò tập trung vào "Hard Examples" (những ca khó phân biệt) thay vì cân bằng dữ liệu.
# Bias Initialization: Khởi tạo bias lớp cuối cùng để output ban đầu của model có xác suất ~1% (prior probability). Điều này giúp Loss không bị "nổ" (explosion) ở những epoch đầu, giúp model hội tụ mượt hơn.

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


# --- 1. CUSTOM COMPONENTS (V3 SPECIALS) ---

class BinaryFocalLoss(nn.Module):
    """
    Loss chuyên dụng cho bài toán 1 Output Node.
    Kết hợp BCEWithLogitsLoss và cơ chế Focal (alpha, gamma).
    """

    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: Logits (chưa qua sigmoid)
        # targets: 0 hoặc 1 (float)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt là xác suất dự đoán đúng

        # Alpha balancing: Vì đã dùng Sampler cân bằng 50/50, ta để alpha=0.5 (cân bằng)
        # hoặc có thể bỏ qua term alpha_t. Ở đây giữ lại để loss linh hoạt.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def initialize_bias(model, device):
    """
    Khởi tạo Bias lớp cuối sao cho xác suất output ban đầu thấp (prior=0.01).
    Tránh Loss quá lớn ở epoch đầu tiên.
    """
    prior = 0.01
    # Công thức nghịch đảo Sigmoid: b = -log((1-p)/p)
    bias_value = -np.log((1 - prior) / prior)

    # Tìm lớp classifier cuối cùng của EfficientNet
    if hasattr(model, 'classifier'):
        layer = model.classifier
    elif hasattr(model, 'fc'):
        layer = model.fc
    else:
        return model  # Không tìm thấy thì bỏ qua

    # Chỉ khởi tạo nếu là Linear Layer
    if isinstance(layer, nn.Linear):
        with torch.no_grad():
            layer.bias.data.fill_(bias_value)
            print(f"🔧 Bias Initialized to {bias_value:.4f} (Prior prob ~ {prior * 100}%)")

    model.to(device)
    return model


# --- 2. LOGGING ---
def log_training_params(version, batch_size, epochs, lr):
    params = {
        "version": version,
        "model": "EfficientNet-B3 (Binary Node)",
        "loss_function": "BinaryFocalLoss (Gamma=2.0)",
        "sampler": "WeightedRandomSampler",
        "init_strategy": "Bias Initialization (p=0.01)",
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


# --- 3. TRAINING CORE (UPDATED FOR BINARY) ---
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, count = 0.0, 0

    for imgs, labels in loader:
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # [V3 UPDATE] Chuyển Labels sang Float và shape (N, 1) để khớp BCEWithLogits
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(imgs)  # Output shape (N, 1)
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
            imgs = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            labels_float = labels.float().unsqueeze(1)  # Chuẩn bị cho Loss

            outputs = model(imgs)
            loss = criterion(outputs, labels_float)
            total_loss += loss.item()

            # [V3 UPDATE] Dùng Sigmoid (vì num_classes=1)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / max(1, len(loader)), np.array(all_labels), np.array(all_probs)


# --- 4. MAIN V3 ---
def train(image_size=300, batch_size=32, epochs=10, base_lr=1e-3):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running V3 (Advanced Arch) on {device}...")

    # MLflow Setup
    os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "dapif865faf65e4f29f9f213de9b6f2ffa3c"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/v3")

    # Paths
    CSV_DIR = 'dataset_splits'
    train_df = pd.read_csv(f'{CSV_DIR}/processed_train.csv')
    val_df = pd.read_csv(f'{CSV_DIR}/processed_val.csv')
    test_df = pd.read_csv(f'{CSV_DIR}/processed_test.csv')
    print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # --- SAMPLER SETUP (Vẫn giữ từ V2) ---
    print("⚖️ Configuring Sampler...")
    y_train = train_df['malignant'].values.astype(int)
    class_counts = np.bincount(y_train)
    sample_weights = 1. / class_counts[y_train]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)

    # Loaders
    train_loader = DataLoader(
        ISICDataset(train_df, image_size, is_train=True),  # Online Aug ON
        batch_size=batch_size, sampler=sampler, shuffle=False,  # Sampler ON
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(ISICDataset(val_df, image_size, is_train=False),
                            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, image_size, is_train=False),
                             batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # --- MODEL SETUP V3 ---
    # [V3 UPDATE] num_classes=1 (Binary Mode)
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=1)

    # [V3 UPDATE] Bias Initialization
    model = initialize_bias(model, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    # [V3 UPDATE] Binary Focal Loss
    criterion = BinaryFocalLoss(alpha=0.5, gamma=2.0)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Run
    with mlflow.start_run(run_name="V3_Advanced"):
        log_training_params("V3_Advanced", batch_size, epochs, base_lr)

        best_pauc = -1
        model_path = "checkpoints/best_v3.pth"
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_labels, val_probs = validate(model, val_loader, criterion)

            metrics = calculate_metrics(val_labels, val_probs)
            current_pauc = metrics['pauc_0.01']

            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()}, step=epoch)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            print(
                f"Epoch [{epoch + 1}/{epochs}] | pAUC: {current_pauc:.4f} | AUC: {metrics['auc']:.4f} | Loss: {val_loss:.4f}")

            if current_pauc > best_pauc:
                best_pauc = current_pauc
                torch.save(model.state_dict(), model_path)
                print(f"  🔥 Saved Best Model (pAUC: {best_pauc:.4f})")

        # Final Test
        print("\n🧪 Testing Best Model V3 (Single View)...")
        model.load_state_dict(torch.load(model_path))
        test_loss, test_labels, test_probs = validate(model, test_loader, criterion)
        test_metrics = calculate_metrics(test_labels, test_probs)

        print(f"🏆 FINAL TEST V3 pAUC: {test_metrics['pauc_0.01']:.4f}")
        print(classification_report(test_labels, (test_probs >= 0.5).astype(int), target_names=['Benign', 'Malignant']))

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == '__main__':
    train()