# Kiến trúc: Chuyển sang num_classes=1 (Binary Mode). Đây là cấu hình chuyên nghiệp cho bài toán 2 lớp, giúp việc tính toán Loss và Bias Init chính xác hơn so với để 2 output nodes.
# Loss Function: Sử dụng Binary Focal Loss.
# Vì đã dùng Sampler (cân bằng số lượng 50/50), Focal Loss ở đây đóng vai trò tập trung vào "Hard Examples" (những ca khó phân biệt) thay vì cân bằng dữ liệu.
# Bias Initialization: Khởi tạo bias lớp cuối cùng để output ban đầu của model có xác suất ~1% (prior probability). Điều này giúp Loss không bị "nổ" (explosion) ở những epoch đầu, giúp model hội tụ mượt hơn.

import sys
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
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score

# Import Dataset
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


# --- 1. COMPONENTS ---
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # targets phải ở cùng device với inputs
        targets = targets.to(inputs.device)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def initialize_bias(model, device):
    prior = 0.01
    bias_value = -np.log((1 - prior) / prior)
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        with torch.no_grad():
            model.classifier.bias.data.fill_(bias_value)
            print(f"🔧 Bias Initialized to {bias_value:.4f}")
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        with torch.no_grad():
            model.fc.bias.data.fill_(bias_value)
            print(f"🔧 Bias Initialized to {bias_value:.4f}")

    model.to(device)
    return model


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


def log_training_params(version, batch_size, epochs, lr):
    params = {
        "version": version,
        "loss": "BinaryFocalLoss",
        "sampler": "WeightedRandomSampler",
        "metric": "pAUC (0.01)"
    }
    mlflow.log_params(params)


# --- 2. TRAIN STEP ---
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, count = 0.0, 0

    for imgs, labels in loader:
        # [QUAN TRỌNG] Gán lại biến sau khi gọi .cuda()
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # Chuyển label sang float (N, 1) - Lúc này labels đã ở trên GPU nên labels_float cũng sẽ ở GPU
        labels_float = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels_float)

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
            # [FIX LỖI DEVICE CPU/CUDA Ở ĐÂY]
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)  # <-- Phải gán lại vào labels

            # labels đang ở GPU -> labels_float sẽ ở GPU
            labels_float = labels.float().unsqueeze(1)

            outputs = model(imgs)  # outputs ở GPU
            loss = criterion(outputs, labels_float)  # Cả 2 đều ở GPU -> OK
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / max(1, len(loader)), np.array(all_labels), np.array(all_probs)


# --- 3. MAIN TRAIN ---
def train(image_size=300, batch_size=32, epochs=10, base_lr=1e-3):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running V3 (Advanced) on {device}...")

    # MLflow Setup
    os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "dapif865faf65e4f29f9f213de9b6f2ffa3c"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/v2")

    CSV_DIR = 'dataset_splits'
    train_df = pd.read_csv(f'{CSV_DIR}/processed_train.csv')
    val_df = pd.read_csv(f'{CSV_DIR}/processed_val.csv')
    test_df = pd.read_csv(f'{CSV_DIR}/processed_test.csv')
    print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Sampler Setup
    print("⚖️ Configuring Sampler...")
    y_train = train_df['malignant'].values.astype(int)
    class_counts = np.bincount(y_train)
    sample_weights = 1. / class_counts[y_train]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)

    # Loaders
    train_loader = DataLoader(ISICDataset(train_df, image_size, is_train=True),
                              batch_size=batch_size, sampler=sampler, shuffle=False,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, image_size, is_train=False),
                            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, image_size, is_train=False),
                             batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=1)
    model = initialize_bias(model, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    criterion = BinaryFocalLoss(alpha=0.5, gamma=2.0)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Run
    with mlflow.start_run(run_name="V3_Advanced"):
        log_training_params("V3_Advanced", batch_size, epochs, base_lr)

        best_pauc = -1
        ckpt_dir = os.path.join(parent_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        model_path = os.path.join(ckpt_dir, "best_v3.pth")

        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']

            # Train step
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)

            # [FIX CẢNH BÁO SCHEDULER] Gọi step() sau khi train xong epoch
            scheduler.step()

            # Validation step
            val_loss, val_labels, val_probs = validate(model, val_loader, criterion)

            # Metrics
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

        # Test
        print("\n🧪 Testing Best Model V3...")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            test_loss, test_labels, test_probs = validate(model, test_loader, criterion)
            test_metrics = calculate_metrics(test_labels, test_probs)

            print(f"🏆 FINAL TEST V3 pAUC: {test_metrics['pauc_0.01']:.4f}")
            print(classification_report(test_labels, (test_probs >= 0.5).astype(int),
                                        target_names=['Benign', 'Malignant']))
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        else:
            print("⚠️ Warning: Model checkpoint not found.")