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

# --- 1. MLflow Configuration ---
os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi987a9e46da628dbdb4a22949054afa24"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3_v4")


# --- 2. Helper Functions ---
def start_mlflow_run(run_name):
    return mlflow.start_run(run_name=run_name)


def log_training_params(mode, image_size, batch_size, epochs, early_stop_patience,
                        train_size, val_size, test_size, device, lr, weight_decay,
                        class_weights=None, lr_strategy="warmup_cosine", gradient_clip=1.0):
    params = {
        "model": "tf_efficientnet_b3.ns_jft_in1k",
        "mode": mode,
        "image_size": image_size,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "early_stop_patience": early_stop_patience,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "device": str(device),
        "loss_function": "FocalLoss",
        "sampler": "WeightedRandomSampler",
        "training_type": "GPU" if torch.cuda.is_available() else "CPU"
    }
    if class_weights is not None:
        params["class_weight_benign"] = float(class_weights[0])
        if len(class_weights) > 1:
            params["class_weight_malignant"] = float(class_weights[1])
    mlflow.log_params(params)


def log_epoch_metrics(epoch, train_loss, val_loss, val_f1, current_lr, detailed_metrics=None):
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_f1_macro": val_f1,
        "learning_rate": current_lr,
    }
    if detailed_metrics:
        metrics.update(detailed_metrics)
    mlflow.log_metrics(metrics, step=epoch)


def log_test_metrics(test_loss, test_f1, detailed_metrics=None):
    metrics = {"test_loss": test_loss, "test_f1_macro": test_f1}
    if detailed_metrics:
        metrics.update(detailed_metrics)
    mlflow.log_metrics(metrics)


# --- 3. Custom Loss: Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# --- 4. Training & Validation Loops ---
def train_one_epoch(model, loader, optimizer, criterion, scaler, gradient_clip=1.0):
    model.train()
    total_loss = 0.0
    count = 0

    for imgs, labels in loader:
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

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


def validate(model, loader, criterion, show_report=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))

    if show_report:
        print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'], digits=4,
                                    zero_division=0))

    return avg_loss, np.array(all_labels), np.array(all_preds)


def calculate_detailed_metrics(y_true, y_pred, prefix="val"):
    f1_benign = f1_score(y_true, y_pred, labels=[0], average='binary', zero_division=0)
    f1_malignant = f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0)

    return {
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        f"{prefix}_f1_benign": f1_benign,
        f"{prefix}_f1_malignant": f1_malignant,
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision_malignant": precision_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_recall_malignant": recall_score(y_true, y_pred, labels=[1], average='binary', zero_division=0)
    }


# --- 5. Main Training Function ---
def train(mode='clean', image_size=300, batch_size=32, epochs=10, base_lr=1e-3, warmup_epochs=2):
    """
    V4: Focal Loss + WeightedRandomSampler + AdamW.
    Chạy trên toàn bộ dữ liệu (Full Data).
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    CSV_DIR = 'dataset_splits'

    # --- A. Path Selection ---
    if mode == 'raw':
        print("📢 MODE: RAW (Imbalanced)")
        train_path = os.path.join(CSV_DIR, 'raw_train.csv')
        val_path = os.path.join(CSV_DIR, 'raw_val.csv')
        test_path = os.path.join(CSV_DIR, 'raw_test.csv')
    elif mode == 'clean':
        print("📢 MODE: CLEAN (Hair Removed)")
        train_path = os.path.join(CSV_DIR, 'clean_train.csv')
        val_path = os.path.join(CSV_DIR, 'clean_val.csv')
        test_path = os.path.join(CSV_DIR, 'clean_test.csv')
    else:
        raise ValueError(f"❌ Mode '{mode}' not supported in V4 (Use 'raw' or 'clean').")

    for name, path in [('Train', train_path), ('Val', val_path), ('Test', test_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing {name}: {path}")

    # --- B. Load Full Data ---
    print(f"📂 Loading FULL Dataset...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"📊 Data Summary:\n  - Train: {len(train_df)}\n  - Val:   {len(val_df)}\n  - Test:  {len(test_df)}")

    # --- C. Weighted Sampler Setup (The V4 Magic) ---
    y_train = train_df['malignant'].values.astype(int)
    unique_classes = np.unique(y_train)

    sampler = None
    if len(unique_classes) < 2:
        print("⚠️ Warning: Single class in training data. Defaulting weights.")
        class_weights = np.array([1.0, 1.0])
        class_weights_tensor = None
    else:
        # 1. Class Weights (for Focal Loss)
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        print(f"⚖️ Class Weights: {class_weights}")
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        # 2. Sampler (for DataLoader balancing)
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in unique_classes])
        weight_per_class = 1. / class_sample_count
        samples_weight = np.array([weight_per_class[t] for t in y_train])
        samples_weight = torch.FloatTensor(samples_weight)

        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
        print("✅ WeightedRandomSampler activated.")

    # --- D. DataLoaders ---
    train_loader = DataLoader(
        ISICDataset(train_df, img_size=image_size),
        batch_size=batch_size,
        sampler=sampler,  # Use Sampler
        shuffle=(sampler is None),  # No shuffle if sampler is used
        num_workers=8, pin_memory=True
    )

    val_loader = DataLoader(ISICDataset(val_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # --- E. Model & Optimizer ---
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    # Focal Loss
    criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
    scaler = GradScaler()
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    # --- F. MLflow ---
    run_name = f"EfficientNetB3_{mode}_v4_Full_Focal"
    mlflow_run = start_mlflow_run(run_name)
    log_training_params(mode, image_size, batch_size, epochs, early_stop_patience=5,
                        train_size=len(train_df), val_size=len(val_df), test_size=len(test_df),
                        device=device, lr=base_lr, weight_decay=0.01,
                        class_weights=class_weights)

    best_f1 = -1
    patience_counter = 0
    delta = 0.005
    gradient_clip = 1.0

    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"best_effb3_{mode}_v4_final.pth")

    try:
        for epoch in range(epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"🚀 Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f}")

            if epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for g in optimizer.param_groups:
                    g['lr'] = warmup_lr
            else:
                cosine_scheduler.step(epoch - warmup_epochs)

            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, gradient_clip)

            # Validate
            val_loss, val_labels, val_preds = validate(model, val_loader, criterion)

            # Metrics
            val_metrics = calculate_detailed_metrics(val_labels, val_preds, prefix="val")
            val_f1 = val_metrics["val_f1_macro"]

            # Logging
            log_epoch_metrics(epoch, train_loss, val_loss, val_f1, current_lr, detailed_metrics=val_metrics)

            print(f"✅ Epoch [{epoch + 1}] | Train: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"F1 Macro: {val_f1:.4f} | F1 Mal: {val_metrics['val_f1_malignant']:.4f} | Best: {best_f1:.4f}")

            # Early Stopping
            if val_f1 > best_f1 * (1 + delta):
                best_f1 = val_f1
                patience_counter = 0
                torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'best_f1': best_f1}, model_path)
                print(f"💾 Model saved: {model_path}")
            else:
                patience_counter += 1

            if patience_counter >= 5:
                print("🛑 Early stopping triggered")
                mlflow.log_param("actual_epochs", epoch + 1)
                break

        # Test Evaluation
        print(f"\n🧪 Testing Best Model...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_labels, test_preds = validate(model, test_loader, criterion, show_report=True)
        test_metrics = calculate_detailed_metrics(test_labels, test_preds, prefix="test")
        log_test_metrics(test_loss, test_metrics["test_f1_macro"], detailed_metrics=test_metrics)

    except Exception as e:
        print(f"❌ Error: {e}")
        mlflow.log_param("error", str(e))
        raise e
    finally:
        mlflow.end_run()
        print("🔚 MLflow run closed.")
