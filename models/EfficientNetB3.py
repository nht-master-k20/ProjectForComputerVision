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
import torch.distributed as dist

# MLflow setup
os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi987a9e46da628dbdb4a22949054afa24"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3_v3")


def start_mlflow_run(run_name, mode, image_size):
    run_name = run_name or f"EfficientNetB3_{mode}_{image_size}"
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
    }
    if class_weights is not None:
        params["class_weight_benign"] = float(class_weights[0])
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


def log_model_artifact(model_path):
    mlflow.log_artifact(model_path)


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def gather_lists_across_ranks(local_list):
    if not dist.is_available() or not dist.is_initialized():
        return local_list
    gathered = [None for _ in range(dist.get_world_size())]
    try:
        dist.all_gather_object(gathered, local_list)
    except Exception:
        return local_list
    out = []
    for part in gathered:
        if part:
            out.extend(part)
    return out


def train_one_epoch(model, loader, optimizer, criterion, scaler, gradient_clip=1.0, ddp=False):
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
    if ddp and dist.is_initialized():
        t_loss = torch.tensor(total_loss, device='cuda')
        t_count = torch.tensor(count, device='cuda')
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
        total_loss = t_loss.item()
        count = t_count.item()
    return total_loss / max(1, count)


def validate(model, loader, criterion, show_report=False, ddp=False):
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
    if ddp and dist.is_initialized():
        gathered_preds = gather_lists_across_ranks(all_preds)
        gathered_labels = gather_lists_across_ranks(all_labels)
        local_count = len(all_labels)
        t_loss = torch.tensor(total_loss, device='cuda')
        t_count = torch.tensor(local_count, device='cuda')
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
        total_loss = t_loss.item()
        total_count = t_count.item()
        avg_loss = total_loss / max(1, total_count)
        accuracy = 0.0;
        macro_precision = 0.0;
        macro_recall = 0.0;
        macro_f1 = 0.0
        if is_main_process():
            accuracy = accuracy_score(gathered_labels, gathered_preds)
            macro_precision = precision_score(gathered_labels, gathered_preds, average='macro', zero_division=0)
            macro_recall = recall_score(gathered_labels, gathered_preds, average='macro', zero_division=0)
            macro_f1 = f1_score(gathered_labels, gathered_preds, average='macro', zero_division=0)
            if show_report:
                print(classification_report(gathered_labels, gathered_preds,
                                            target_names=['Benign (0)', 'Malignant (1)'], digits=4, zero_division=0))
    else:
        avg_loss = total_loss / max(1, len(loader))
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        macro_precision = precision_score(all_labels, all_preds, average='macro',
                                          zero_division=0) if all_labels else 0.0
        macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0.0
        if show_report:
            print(classification_report(all_labels, all_preds,
                                        target_names=['Benign (0)', 'Malignant (1)'], digits=4, zero_division=0))
    return avg_loss, accuracy, macro_precision, macro_recall, macro_f1


def train(mode='augment', image_size=300, batch_size=32, epochs=10, base_lr=1e-3, warmup_epochs=2):
    """
    Hàm train chính với 3 chế độ bắt buộc:
    - mode='raw': Dữ liệu gốc, chưa xử lý.
    - mode='clean': Dữ liệu đã xóa lông (nhưng chưa cân bằng/augment).
    - mode='augment': Dữ liệu đã xóa lông VÀ cân bằng (sinh thêm ảnh).
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CẤU HÌNH ĐƯỜNG DẪN CSV ---
    # Tất cả file CSV đều nằm trong thư mục này (do class ReadData sinh ra)
    CSV_DIR = 'dataset_splits'

    if mode == 'raw':
        print("📢 Chế độ: RAW (Dữ liệu gốc)")
        train_path = os.path.join(CSV_DIR, 'raw_train.csv')
        val_path = os.path.join(CSV_DIR, 'raw_val.csv')
        test_path = os.path.join(CSV_DIR, 'raw_test.csv')

    elif mode == 'clean':
        print("📢 Chế độ: CLEAN (Dữ liệu sạch - xóa lông)")
        train_path = os.path.join(CSV_DIR, 'clean_train.csv')
        val_path = os.path.join(CSV_DIR, 'clean_val.csv')
        test_path = os.path.join(CSV_DIR, 'clean_test.csv')

    else:
        raise ValueError(f"❌ Mode không hợp lệ: '{mode}'. Vui lòng chọn: 'raw', 'clean', hoặc 'augment'.")

    # Kiểm tra file tồn tại
    for name, path in [('Train', train_path), ('Val', val_path), ('Test', test_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ Không tìm thấy file {name}: {path}.\n👉 Hãy chạy ReadData.run(mode='{mode if mode != 'clean' else 'raw'}', clean=True) trước.")

    print(f"📂 Loading Data from:\n - Train: {train_path}\n - Val:   {val_path}\n - Test:  {test_path}")

    # Load CSV
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"\n⚠️ WARNING: Đang chạy chế độ DEBUG với {500} mẫu mỗi tập!")
    train_df = train_df.head(500)

    # DataLoaders
    train_loader = DataLoader(ISICDataset(train_df, img_size=image_size), batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # Model
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    # Class Weights: Tính toán tự động dựa trên dữ liệu Train load được
    # - Raw/Clean: Thường mất cân bằng -> Trọng số sẽ cao cho lớp Ác tính.
    # - Augment: Đã cân bằng -> Trọng số xấp xỉ 1:1.
    y_train = train_df['malignant'].values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"⚖️ Class Weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = GradScaler()
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )

    # MLflow Run
    if is_main_process():
        run_name = f"EfficientNetB3_{mode}"
        mlflow_run = start_mlflow_run(run_name, mode, image_size)
        log_training_params(
            mode, image_size, batch_size, epochs, early_stop_patience=5,
            train_size=len(train_df), val_size=len(val_df), test_size=len(test_df),
            device=device, lr=base_lr, class_weights=class_weights
        )
    else:
        mlflow_run = None

    best_f1 = -1
    patience_counter = 0
    delta = 0.005
    gradient_clip = 1.0

    # Checkpoint directory
    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"best_efficientnet_b3_{mode}.pth")

    try:
        # Training Loop
        for epoch in range(epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"🚀 Starting Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f}")

            # Warmup
            if epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for g in optimizer.param_groups:
                    g['lr'] = warmup_lr
            else:
                cosine_scheduler.step(epoch - warmup_epochs)

            # Train Step
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, gradient_clip
            )

            # Validation Step
            val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
                model, val_loader, criterion, show_report=False
            )

            if is_main_process():
                log_epoch_metrics(
                    epoch, train_loss, val_loss, val_acc,
                    val_precision, val_recall, val_f1, current_lr
                )

                print(
                    f"✅ Epoch [{epoch + 1}/{epochs}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val F1: {val_f1:.4f} | "
                    f"Best F1: {best_f1:.4f} | "
                    f"Patience: {patience_counter}/5"
                )

                # Early Stopping & Save Best Model
                if val_f1 > best_f1 * (1 + delta):
                    best_f1 = val_f1
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_f1': best_f1
                    }, model_path)
                    print(f"💾 Model saved: {model_path}")
                else:
                    patience_counter += 1

                if patience_counter >= 5:
                    print("🛑 Early stopping triggered")
                    mlflow.log_param("actual_epochs", epoch + 1)
                    break

        # Final Evaluation on Test Set
        if is_main_process():
            print(f"\n🧪 Evaluating on Test Set (Mode: {mode})...")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            test_loss, test_acc, test_precision, test_recall, test_f1 = validate(
                model, test_loader, criterion, show_report=True
            )

            log_test_metrics(
                test_loss, test_acc, test_precision, test_recall, test_f1, best_f1
            )

    except Exception as e:
        if is_main_process():
            print("❌ ERROR:", str(e))
            mlflow.log_param("run_failed", True)
            mlflow.log_param("error_message", str(e))
        raise e

    finally:
        if is_main_process():
            print("🔚 MLflow run closed.")
            mlflow.end_run()
