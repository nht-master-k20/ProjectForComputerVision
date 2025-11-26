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
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from scripts.ISICDataset import ISICDataset


# --- 1. CUSTOM LOSS: BINARY FOCAL LOSS ---
class BinaryFocalLoss(nn.Module):
    """
    Loss function cho bài toán 1 đầu ra (Binary).
    Kết hợp BCEWithLogitsLoss và cơ chế Focal để tập trung vào ca khó.
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # Flatten để đảm bảo shape khớp nhau: (N, 1) -> (N,)
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Tính BCE loss (Logits -> Sigmoid -> Log)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)  # pt là xác suất mô hình dự đoán đúng

        # Áp dụng trọng số Alpha (Cân bằng giữa Lành và Ác)
        # Nếu target=1 (Ác), nhân với alpha. Nếu target=0 (Lành), nhân với (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Công thức Focal: alpha * (1-pt)^gamma * loss
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


def log_training_params(
    image_size,
    batch_size,
    epochs,
    train_len,
    val_len,
    test_len,
    device,
    lr,
    weight_decay,
    class_weights,
):
    params = {
        "version": "v5_Binary_pAUC",
        "model": "tf_efficientnet_b3.ns_jft_in1k",
        "architecture": "Binary (1 Output Node)",
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "early_stopping": "False (Full Epochs)",
        "optimizer": "AdamW",
        "lr": lr,
        "weight_decay": weight_decay,
        "train_size": train_len,
        "val_size": val_len,
        "test_size": test_len,
        "device": str(device),
        "loss_function": "BinaryFocalLoss",
        "sampler": "WeightedRandomSampler (50/50 Balance)",
        "technique": "BiasInit + pAUC model selection",
        "training_type": "GPU" if torch.cuda.is_available() else "CPU",
        "selection_metric": "val_pAUC_0.1",
    }
    if class_weights is not None:
        params.update(
            {
                "cw_benign": float(class_weights[0]),
                "cw_malignant": float(class_weights[1]),
            }
        )
    mlflow.log_params(params)


def log_metrics(prefix, metrics, step=None):
    # prefix hiện không dùng, giữ để không phá API cũ
    mlflow.log_metrics({k: v for k, v in metrics.items()}, step=step)


def find_best_threshold(y_true, y_probs):
    """Tìm ngưỡng tối ưu F1 cho class Malignant (label=1)."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.01, 0.90, 0.01):
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(
            y_true,
            y_pred,
            labels=[1],
            average='binary',
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def calculate_metrics(y_true, y_probs, threshold=0.5, prefix="val"):
    y_pred = (y_probs >= threshold).astype(int)
    return {
        f"{prefix}_f1_macro": f1_score(
            y_true, y_pred, average='macro', zero_division=0
        ),
        f"{prefix}_f1_malignant": f1_score(
            y_true,
            y_pred,
            labels=[1],
            average='binary',
            zero_division=0,
        ),
        f"{prefix}_recall_malignant": recall_score(
            y_true,
            y_pred,
            labels=[1],
            average='binary',
            zero_division=0,
        ),
        f"{prefix}_precision_malignant": precision_score(
            y_true,
            y_pred,
            labels=[1],
            average='binary',
            zero_division=0,
        ),
        f"{prefix}_threshold": threshold,
    }


def calculate_pauc(y_true, y_probs, max_fpr=0.1, prefix="val"):
    """
    Tính partial AUC trong vùng FPR ∈ [0, max_fpr].
    Sử dụng roc_auc_score với tham số max_fpr (sklearn sẽ scale về [0,1]).
    """
    try:
        pauc = roc_auc_score(y_true, y_probs, max_fpr=max_fpr)
    except ValueError:
        # Trường hợp hiếm: chỉ có 1 class trong y_true
        pauc = float('nan')
    return {f"{prefix}_pAUC_{max_fpr}": float(pauc)}


def initialize_bias(model, device):
    """
    Khởi tạo Bias cho 1 nơ-ron đầu ra.
    Mục đích: Ép xác suất dự đoán ban đầu về gần 1% (prior=0.01)
    để tránh Loss quá lớn lúc khởi động.
    """
    prior = 0.01
    bias_value = -np.log((1 - prior) / prior)

    # Tìm lớp classifier cuối cùng của EfficientNet
    if hasattr(model, 'classifier'):
        layer = model.classifier
    elif hasattr(model, 'fc'):
        layer = model.fc
    else:
        model.to(device)
        return model

    # Chỉ khởi tạo nếu là lớp Linear
    if isinstance(layer, nn.Linear):
        with torch.no_grad():
            layer.bias.data.fill_(bias_value)
            print(
                f"Bias Initialized to {bias_value:.4f} "
                f"(Setting initial prob ~ {prior * 100}%)"
            )

    model.to(device)
    return model


# --- 3. TRAINING LOOPS ---
def train_one_epoch(model, loader, optimizer, criterion, scaler, gradient_clip=1.0):
    model.train()
    total_loss, count = 0.0, 0

    for imgs, labels in loader:
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # Chuyển labels sang float và shape (N, 1) để khớp với BinaryFocalLoss
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(imgs)  # Logits (1 giá trị thực)
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

            # Chuẩn bị label cho Loss calculation
            labels_float = labels_orig.float().unsqueeze(1)

            outputs = model(imgs)
            loss = criterion(outputs, labels_float)
            total_loss += loss.item()

            # Dùng Sigmoid cho Binary
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels_orig.cpu().numpy())

    return total_loss / max(1, len(loader)), np.array(all_labels), np.array(all_probs)


# --- 4. MAIN EXECUTION (v5 - pAUC based) ---
def train(mode='processed', image_size=300, batch_size=32, epochs=10, base_lr=1e-3):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Version: v5 (Binary + pAUC)")

    # A. MLflow Config
    # Lưu ý: token nên được set qua env bên ngoài trong thực tế.
    os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "dapif865faf65e4f29f9f213de9b6f2ffa3c"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(
        "/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3_v5"
    )

    # B. Load Data
    CSV_DIR = 'dataset_splits'
    prefix = "processed" if mode == 'processed' else "raw"
    paths = {
        'train': os.path.join(CSV_DIR, f'{prefix}_train.csv'),
        'val': os.path.join(CSV_DIR, f'{prefix}_val.csv'),
        'test': os.path.join(CSV_DIR, f'{prefix}_test.csv'),
    }
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    train_df = pd.read_csv(paths['train'])
    val_df = pd.read_csv(paths['val'])
    test_df = pd.read_csv(paths['test'])
    print(f"Stats: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # C. Sampler (Weighted 50/50)
    y_train = train_df['malignant'].values.astype(int)
    class_counts = np.bincount(y_train)

    if len(class_counts) < 2:
        print("Cảnh báo: Chỉ có 1 class. Sampler bị tắt.")
        sampler = None
    else:
        sample_weights = 1.0 / class_counts[y_train]
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(sample_weights),
            len(sample_weights),
            replacement=True,
        )
        print("WeightedRandomSampler Activated (Ép cân bằng 50/50)")

    # D. DataLoaders
    train_loader = DataLoader(
        ISICDataset(train_df, img_size=image_size),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ISICDataset(val_df, img_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = DataLoader(
        ISICDataset(test_df, img_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # E. Model Setup (Binary: num_classes=1)
    print("Creating Model: EfficientNet-B3 (num_classes=1)")
    model = timm.create_model(
        "tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=1
    )
    model = initialize_bias(model, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    criterion = BinaryFocalLoss(gamma=2.0, alpha=0.25)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # F. Training Loop
    mlflow_run = start_mlflow_run(f"EffB3_{mode}_v5")
    log_training_params(
        image_size,
        batch_size,
        epochs,
        len(train_df),
        len(val_df),
        len(test_df),
        device,
        base_lr,
        0.01,
        None,
    )

    best_pauc = -1.0
    best_epoch = -1
    best_thresh_val = 0.5
    model_path = f"checkpoints/best_effb3_{mode}_v5.pth"
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Bắt đầu Training {epochs} epochs (Model selection bằng val_pAUC_0.1)...")

    try:
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            # 1. Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler
            )

            # 2. Val
            val_loss, val_labels, val_probs = validate(model, val_loader, criterion)

            # 3. Tìm threshold tốt nhất theo F1 (để dùng cho classification report)
            optimal_threshold, best_val_f1_mal = find_best_threshold(
                val_labels, val_probs
            )
            val_cls_metrics = calculate_metrics(
                val_labels, val_probs, threshold=optimal_threshold, prefix="val"
            )

            # 3b. Tính pAUC trong vùng FPR <= 0.1
            val_pauc_metrics = calculate_pauc(
                val_labels, val_probs, max_fpr=0.1, prefix="val"
            )
            current_val_pauc = val_pauc_metrics["val_pAUC_0.1"]

            # 4. Log
            log_metrics(
                "val",
                {
                    **val_cls_metrics,
                    **val_pauc_metrics,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                },
                step=epoch,
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"F1 Mal: {best_val_f1_mal:.4f} (Thresh: {optimal_threshold:.2f}) | "
                f"val_pAUC_0.1: {current_val_pauc:.4f}"
            )

            # 5. Save Best: CHỌN THEO pAUC
            if current_val_pauc > best_pauc:
                best_pauc = current_val_pauc
                best_thresh_val = optimal_threshold
                best_epoch = epoch
                torch.save(
                    {
                        'state_dict': model.state_dict(),
                        'threshold': best_thresh_val,
                        'epoch': best_epoch,
                        'val_pAUC_0.1': best_pauc,
                        'val_f1_malignant': best_val_f1_mal,
                    },
                    model_path,
                )
                print(
                    f"Saved New Best Model "
                    f"(val_pAUC_0.1: {best_pauc:.4f}, F1_mal: {best_val_f1_mal:.4f})"
                )

        # G. Final Test
        print("\nTesting Best Model (Chọn theo val_pAUC_0.1 cao nhất)...")
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            loaded_thresh = ckpt.get('threshold', 0.5)
            best_epoch = ckpt.get('epoch', 'Unknown')
            best_pauc_loaded = ckpt.get('val_pAUC_0.1', float('nan'))
            print(
                f"Loaded Checkpoint (Epoch {best_epoch} | "
                f"val_pAUC_0.1={best_pauc_loaded:.4f} | Thresh={loaded_thresh:.2f})"
            )
        else:
            loaded_thresh = 0.5
            print(
                "Warning: Không tìm thấy file checkpoint. "
                "Dùng model cuối cùng và threshold 0.5."
            )

        test_loss, test_labels, test_probs = validate(model, test_loader, criterion)
        test_metrics = calculate_metrics(
            test_labels, test_probs, threshold=loaded_thresh, prefix="test"
        )
        test_pauc_metrics = calculate_pauc(
            test_labels, test_probs, max_fpr=0.1, prefix="test"
        )

        print("\n" + "=" * 30)
        print(f"FINAL TEST REPORT (Threshold: {loaded_thresh:.2f})")
        print(
            classification_report(
                test_labels,
                (test_probs >= loaded_thresh).astype(int),
                target_names=['Benign', 'Malignant'],
                digits=4,
            )
        )
        print(
            f"Test pAUC (FPR ≤ 0.1): "
            f"{test_pauc_metrics['test_pAUC_0.1']:.4f}"
        )
        print("=" * 30)

        log_metrics(
            "test",
            {
                **test_metrics,
                **test_pauc_metrics,
                "test_loss": test_loss,
            },
        )

    finally:
        mlflow.end_run()
        print("Done v5 (Binary + pAUC).")
