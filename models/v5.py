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
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score

# --- 1. FIX PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scripts.ISICDataset2 import ISICDataset


# --- 0. SEED ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 1. HELPERS ---
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


# --- 2. PREDICT FUNCTIONS (Hỗ trợ 2 classes) ---
def get_probs_from_model(model, loader, device='cuda'):
    """Chạy 1 lần (Single View) để tìm Threshold trên Val"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            # V2 dùng 2 output nodes -> Dùng Softmax
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Lấy cột 1 (Malignant)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


def predict_tta(model, loader, tta_steps=5, device='cuda'):
    """Chạy TTA trên Test"""
    model.eval()
    num_samples = len(loader.dataset)
    accumulated_probs = np.zeros(num_samples)
    final_labels = None

    print(f"🔄 Starting TTA ({tta_steps} views)...")
    with torch.no_grad():
        for i in range(tta_steps):
            print(f"   ► View {i + 1}/{tta_steps}")
            step_probs = []
            step_labels = []

            for imgs, labels in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)

                # Softmax cho V2
                probs = torch.softmax(outputs, dim=1)[:, 1]

                step_probs.extend(probs.cpu().numpy())
                step_labels.extend(labels.numpy())

            accumulated_probs += np.array(step_probs)
            if final_labels is None: final_labels = np.array(step_labels)

    avg_probs = accumulated_probs / tta_steps
    return final_labels, avg_probs


# --- 3. MAIN V5 ---
def run_v5(image_size=300, batch_size=32, tta_steps=5):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running V5 (Best Model V2 + Tuning + TTA) on {device}...")

    # MLflow Setup
    os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "dapif865faf65e4f29f9f213de9b6f2ffa3c"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/v5")

    # Load Data
    CSV_DIR = os.path.join(parent_dir, 'dataset_splits')
    val_df = pd.read_csv(f'{CSV_DIR}/processed_val.csv')
    test_df = pd.read_csv(f'{CSV_DIR}/processed_test.csv')
    print(f"📊 Data: Val={len(val_df)} | Test={len(test_df)}")

    # Loaders
    # Val Loader (Single view, no augment) -> Để tìm threshold
    val_loader = DataLoader(ISICDataset(val_df, image_size, is_train=False),
                            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Test Loader (TTA view, Augment ON) -> Để dự đoán cuối cùng
    tta_loader = DataLoader(ISICDataset(test_df, image_size, is_train=True),
                            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # --- LOAD MODEL V2 ---
    ckpt_dir = os.path.join(parent_dir, 'checkpoints')
    model_path = os.path.join(ckpt_dir, "best_v2.pth")  # Load V2!

    if not os.path.exists(model_path):
        print(f"❌ Error: Không tìm thấy {model_path}. Hãy train V2 trước!")
        return

    print("🏗️ Loading V2 Model (EfficientNet-B3, num_classes=2)...")
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    with mlflow.start_run(run_name="V5_Final_Combo"):
        mlflow.log_param("version", "V5_Combo")
        mlflow.log_param("base_model", "V2_Sampler")

        # --- BƯỚC 1: TÌM THRESHOLD TRÊN VAL ---
        print("\n🔎 Step 1: Finding Best Threshold on Validation...")
        val_labels, val_probs = get_probs_from_model(model, val_loader, device)

        best_thresh = 0.5
        best_f1 = 0.0
        # Quét ngưỡng
        for thr in np.arange(0.01, 0.95, 0.01):
            preds = (val_probs >= thr).astype(int)
            # Ưu tiên F1 score của lớp Malignant
            score = f1_score(val_labels, preds, labels=[1], average='binary', zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thr

        print(f"✅ Found Threshold: {best_thresh:.3f} (Val F1: {best_f1:.4f})")
        mlflow.log_param("tuned_threshold", best_thresh)

        # --- BƯỚC 2: CHẠY TTA TRÊN TEST ---
        print(f"\n🚀 Step 2: Running TTA on Test ({tta_steps} steps)...")
        test_labels, test_probs = predict_tta(model, tta_loader, tta_steps, device)

        # --- BƯỚC 3: ĐÁNH GIÁ ---
        metrics = calculate_metrics(test_labels, test_probs, threshold=best_thresh)

        print("\n" + "=" * 40)
        print(f"🏆 FINAL RESULT V5 (Model V2 + TTA + Thresh {best_thresh:.2f})")
        print(f"pAUC (0.01): {metrics['pauc_0.01']:.4f}")
        print(f"Recall Mal : {metrics['recall_malignant']:.4f}")
        print(f"F1 Mal     : {metrics['f1_malignant']:.4f}")
        print("=" * 40)

        print(classification_report(test_labels, (test_probs >= best_thresh).astype(int),
                                    target_names=['Benign', 'Malignant']))

        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})