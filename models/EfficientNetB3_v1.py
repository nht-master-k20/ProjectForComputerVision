import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from scripts.ISICDataset import ISICDataset


# --- 1. HELPER FUNCTIONS ---
def start_mlflow_run(run_name):
    return mlflow.start_run(run_name=run_name)


def log_training_params(image_size, batch_size, epochs, train_len, val_len, test_len, device, lr, weight_decay,
                        class_weights):
    params = {
        "version": "v1_Baseline",
        "model": "tf_efficientnet_b3.ns_jft_in1k",
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "AdamW",
        "lr": lr,
        "weight_decay": weight_decay,  # Đã thêm
        "train_size": train_len,  # Đã thêm
        "val_size": val_len,  # Đã thêm
        "test_size": test_len,  # Đã thêm
        "device": str(device),
        "loss_function": "CrossEntropyLoss",
        "sampler": "None (Shuffle=True)",
        "training_type": "GPU" if torch.cuda.is_available() else "CPU"
    }
    if class_weights is not None:
        params["cw_benign"] = float(class_weights[0])
        if len(class_weights) > 1: params["cw_malignant"] = float(class_weights[1])
    mlflow.log_params(params)


def log_metrics(prefix, metrics, step=None):
    log_data = {k: v for k, v in metrics.items()}
    mlflow.log_metrics(log_data, step=step)


def calculate_metrics(y_true, y_pred, prefix="val"):
    return {
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        f"{prefix}_f1_malignant": f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_recall_malignant": recall_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_precision_malignant": precision_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred)
    }


# --- 2. TRAINING LOOPS ---
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


# --- 3. MAIN FUNCTION ---
def train(mode='processed', image_size=300, batch_size=32, epochs=10, base_lr=1e-3):
    # A. Setup
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device} | Version: v1 (Baseline)")

    # MLflow Setup
    os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "dapif865faf65e4f29f9f213de9b6f2ffa3c"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3_v1")

    # B. Paths
    CSV_DIR = 'dataset_splits'
    prefix = "processed" if mode == 'processed' else "raw"

    train_path = os.path.join(CSV_DIR, f'{prefix}_train.csv')
    val_path = os.path.join(CSV_DIR, f'{prefix}_val.csv')
    test_path = os.path.join(CSV_DIR, f'{prefix}_test.csv')

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p): raise FileNotFoundError(f"❌ Missing: {p}")

    print(f"📂 Loading Data ({mode})...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    print(f"📊 Stats: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # C. DataLoaders
    train_loader = DataLoader(ISICDataset(train_df, img_size=image_size), batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # D. Model & Optim
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    y_train = train_df['malignant'].values
    if len(np.unique(y_train)) < 2:
        cw = np.array([1.0, 1.0])
    else:
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"⚖️ Class Weights: {cw}")

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to(device))
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # E. Loop
    mlflow_run = start_mlflow_run(f"EffB3_{mode}_v1")
    log_training_params(image_size, batch_size, epochs, len(train_df), len(val_df), len(test_df), device, base_lr, 0.01,
                        cw)

    best_f1, patience = -1, 0
    model_path = f"checkpoints/best_effb3_{mode}_v1.pth"
    os.makedirs("checkpoints", exist_ok=True)

    try:
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            print(f"🚀 Epoch [{epoch + 1}/{epochs}] | LR: {lr:.6f}")
            scheduler.step()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_labels, val_preds = validate(model, val_loader, criterion)

            metrics = calculate_metrics(val_labels, val_preds, prefix="val")
            log_metrics("val", {**metrics, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}, step=epoch)

            print(f"✅ Train: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1 Macro: {metrics['val_f1_macro']:.4f}")

            if metrics['val_f1_macro'] > best_f1 * 1.005:
                best_f1, patience = metrics['val_f1_macro'], 0
                torch.save(model.state_dict(), model_path)
                print(f"💾 Saved: {model_path}")
            else:
                patience += 1
                if patience >= 5: print("🛑 Early Stopping"); break

        print(f"\n🧪 Testing Best Model...")
        model.load_state_dict(torch.load(model_path))
        test_loss, test_labels, test_preds = validate(model, test_loader, criterion, show_report=True)
        test_metrics = calculate_metrics(test_labels, test_preds, prefix="test")
        log_metrics("test", {**test_metrics, "test_loss": test_loss})

    finally:
        mlflow.end_run()