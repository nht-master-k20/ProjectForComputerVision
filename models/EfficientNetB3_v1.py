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

# --- 1. CONFIGURATION ---
os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi987a9e46da628dbdb4a22949054afa24"  # LƯU Ý: Nên dùng biến môi trường
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3_v1")


# --- 2. HELPER FUNCTIONS ---
def start_mlflow_run(run_name):
    return mlflow.start_run(run_name=run_name)


def log_training_params(mode, image_size, batch_size, epochs, early_stop_patience,
                        train_size, val_size, test_size, device, lr, weight_decay,
                        class_weights=None):
    params = {
        "version": "v1_Baseline",
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
        "loss_function": "CrossEntropyLoss",
        "sampler": "None (Shuffle=True)",
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


def calculate_metrics(y_true, y_pred, prefix="val"):
    return {
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        f"{prefix}_f1_malignant": f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_recall_malignant": recall_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_precision_malignant": precision_score(y_true, y_pred, labels=[1], average='binary', zero_division=0),
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred)
    }


# --- 3. LOOPS ---
def train_one_epoch(model, loader, optimizer, criterion, scaler, gradient_clip=1.0):
    model.train()
    total_loss = 0.0
    count = 0
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
        total_loss += loss.item()
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
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(1, len(loader))
    if show_report:
        print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'], digits=4,
                                    zero_division=0))
    return avg_loss, np.array(all_labels), np.array(all_preds)


# --- 4. MAIN TRAIN ---
def train(mode='clean', image_size=300, batch_size=32, epochs=10, base_lr=1e-3, warmup_epochs=2):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device} | Version: v1 (Baseline)")

    # A. Load Data
    CSV_DIR = 'dataset_splits'
    paths = {
        'raw': ('raw_train.csv', 'raw_val.csv', 'raw_test.csv'),
        'clean': ('clean_train.csv', 'clean_val.csv', 'clean_test.csv')
    }
    if mode not in paths: raise ValueError(f"❌ Invalid mode: {mode}")

    train_path, val_path, test_path = [os.path.join(CSV_DIR, p) for p in paths[mode]]
    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p): raise FileNotFoundError(f"❌ Missing: {p}")

    print(f"📂 Loading Data...")
    train_df, val_df, test_df = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)
    print(f"📊 Stats: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # B. DataLoaders (Shuffle=True for v1)
    train_loader = DataLoader(ISICDataset(train_df, img_size=image_size), batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size=image_size), batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # C. Model & Weights
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    y_train = train_df['malignant'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"⚖️ Class Weights: {class_weights}")

    # v1 uses CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    # D. Training Loop
    mlflow_run = start_mlflow_run(f"EffB3_{mode}_v1")
    log_training_params(mode, image_size, batch_size, epochs, 5, len(train_df), len(val_df), len(test_df), device,
                        base_lr, 0.01, class_weights)

    best_f1, patience = -1, 0
    model_path = f"checkpoints/best_effb3_{mode}_v1.pth"
    os.makedirs("checkpoints", exist_ok=True)

    try:
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            print(f"🚀 Epoch [{epoch + 1}/{epochs}] | LR: {lr:.6f}")

            if epoch < warmup_epochs:  # Warmup
                for g in optimizer.param_groups: g['lr'] = base_lr * (epoch + 1) / warmup_epochs
            else:
                scheduler.step(epoch - warmup_epochs)

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_labels, val_preds = validate(model, val_loader, criterion)

            metrics = calculate_metrics(val_labels, val_preds, prefix="val")
            log_epoch_metrics(epoch, train_loss, val_loss, metrics['val_f1_macro'], lr, metrics)

            print(
                f"✅ Train: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1 Macro: {metrics['val_f1_macro']:.4f} | Best: {best_f1:.4f}")

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
        log_test_metrics(test_loss, test_metrics['test_f1_macro'], test_metrics)

    finally:
        mlflow.end_run()
        print("🔚 Done v1.")
