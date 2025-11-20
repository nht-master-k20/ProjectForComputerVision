from torch.utils.data import DataLoader
import timm
import torch.nn as nn
import torch
from torch.amp import GradScaler, autocast
import pandas as pd
from scripts.ISICDataset import ISICDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split

# --- 1. MLflow Configuration ---
os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi987a9e46da628dbdb4a22949054afa24"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3_v3")


# --- 2. Helper Functions ---
def start_mlflow_run(run_name):
    return mlflow.start_run(run_name=run_name)


def log_training_params(mode, image_size, batch_size, epochs, early_stop_patience,
                        train_size, val_size, test_size, device, lr=1e-3, weight_decay=0.01,
                        class_weights=None, lr_strategy="warmup_cosine", gradient_clip=1.0):
    params = {
        "model": "tf_efficientnet_b3.ns_jft_in1k",
        "mode": mode,
        "image_size": image_size,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "lr_strategy": lr_strategy,
        "weight_decay": weight_decay,
        "gradient_clip_max_norm": gradient_clip,
        "scheduler": "CosineAnnealingLR",
        "scheduler_T_max": epochs,
        "scheduler_eta_min": 1e-6,
        "early_stop_patience": early_stop_patience,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "device": str(device),
        "num_workers": 8,
        "pin_memory": True,
        "mixed_precision": True,
        "metrics_average": "macro",
        "use_class_weights": class_weights is not None,
        "training_type": "GPU" if torch.cuda.is_available() else "CPU"
    }
    if class_weights is not None:
        params["class_weight_benign"] = float(class_weights[0])
        if len(class_weights) > 1:
            params["class_weight_malignant"] = float(class_weights[1])
    mlflow.log_params(params)


def log_epoch_metrics(epoch, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1, current_lr):
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "learning_rate": current_lr,
    }, step=epoch)


def log_test_metrics(test_loss, test_acc, test_precision, test_recall, test_f1, best_val_f1):
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "best_val_f1": best_val_f1,
    })


# --- 3. Training & Validation Loops ---
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
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0.0
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0.0

    if show_report:
        print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'], digits=4,
                                    zero_division=0))

    return avg_loss, accuracy, macro_precision, macro_recall, macro_f1


# --- 4. Main Training Function ---
def train(mode='clean', image_size=300, batch_size=32, epochs=10, base_lr=1e-3, warmup_epochs=2):
    """
    Hàm train chính hỗ trợ 2 chế độ: 'raw' và 'clean'.
    Chạy trên TOÀN BỘ dữ liệu.
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    CSV_DIR = 'dataset_splits'

    # --- A. Chọn đường dẫn dữ liệu ---
    if mode == 'raw':
        print("📢 Chế độ: RAW (Dữ liệu gốc, chưa xử lý)")
        train_path = os.path.join(CSV_DIR, 'raw_train.csv')
        val_path = os.path.join(CSV_DIR, 'raw_val.csv')
        test_path = os.path.join(CSV_DIR, 'raw_test.csv')

    elif mode == 'clean':
        print("📢 Chế độ: CLEAN (Dữ liệu sạch - đã xóa lông)")
        train_path = os.path.join(CSV_DIR, 'clean_train.csv')
        val_path = os.path.join(CSV_DIR, 'clean_val.csv')
        test_path = os.path.join(CSV_DIR, 'clean_test.csv')

    else:
        raise ValueError(f"❌ Mode không hợp lệ: '{mode}'. Chỉ hỗ trợ: 'raw', 'clean'.")

    # Kiểm tra file tồn tại
    for name, path in [('Train', train_path), ('Val', val_path), ('Test', test_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ Không tìm thấy file {name}: {path}. Hãy chạy ReadData.run(mode='{mode}', clean=True/False) trước.")

    print(f"📂 Loading Data from:\n - Train: {train_path}\n - Val:   {val_path}\n - Test:  {test_path}")

    # --- B. Load dữ liệu (FULL DATASET) ---
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"📊 Data Summary:\n  - Train: {len(train_df)}\n  - Val:   {len(val_df)}\n  - Test:  {len(test_df)}")

    # --- D. DataLoaders ---
    train_loader = DataLoader(ISICDataset(train_df, img_size=image_size), batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # --- E. Model & Optimizer ---
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    # --- F. Tính Class Weights (Giữ lại check an toàn) ---
    y_train = train_df['malignant'].values
    unique_classes = np.unique(y_train)

    if len(unique_classes) < 2:
        print(f"⚠️ CẢNH BÁO: Dữ liệu train chỉ có 1 lớp ({unique_classes}). Gán trọng số mặc định [1.0, 1.0].")
        class_weights = np.array([1.0, 1.0])
    else:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)

    print(f"⚖️ Class Weights: {class_weights}")
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = GradScaler()
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    # --- G. MLflow & Directories ---
    run_name = f"EfficientNetB3_{mode}_v3"
    mlflow_run = start_mlflow_run(run_name)
    log_training_params(
        mode, image_size, batch_size, epochs, early_stop_patience=5,
        train_size=len(train_df), val_size=len(val_df), test_size=len(test_df),
        device=device, lr=base_lr, class_weights=class_weights
    )

    best_f1 = -1
    patience_counter = 0
    delta = 0.005
    gradient_clip = 1.0

    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"best_efficientnet_b3_{mode}_v3.pth")

    try:
        # --- H. Training Loop ---
        for epoch in range(epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"🚀 Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f}")

            if epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for g in optimizer.param_groups:
                    g['lr'] = warmup_lr
            else:
                cosine_scheduler.step(epoch - warmup_epochs)

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, gradient_clip)
            val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion,
                                                                            show_report=False)

            log_epoch_metrics(epoch, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1, current_lr)
            print(
                f"✅ Epoch [{epoch + 1}] | Train: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Best F1: {best_f1:.4f}")

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

        # --- I. Test Evaluation ---
        print(f"\n🧪 Evaluating on Test Set...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_precision, test_recall, test_f1 = validate(model, test_loader, criterion,
                                                                             show_report=True)
        log_test_metrics(test_loss, test_acc, test_precision, test_recall, test_f1, best_f1)

    except Exception as e:
        print("❌ ERROR:", str(e))
        mlflow.log_param("run_failed", True)
        mlflow.log_param("error_message", str(e))
        raise e

    finally:
        print("🔚 MLflow run closed.")
        mlflow.end_run()
