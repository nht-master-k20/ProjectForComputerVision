import sys
import os
import random
import numpy as np
import pandas as pd
import torch
import timm
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score

# --- 1. FIX PATH IMPORT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Dataset (Kiểm tra kỹ tên file ISICDataset hay ISICDataset2 trong folder scripts)
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


def log_inference_params(tta_steps):
    params = {
        "version": "v4_Inference_TTA",
        "base_model": "V3_Advanced (Loaded Checkpoint)",
        "inference_strategy": f"Test Time Augmentation (Steps={tta_steps})",
        "metric_target": "pAUC (0.01)"
    }
    mlflow.log_params(params)


# --- 2. TTA CORE FUNCTION ---
def predict_tta(model, loader, tta_steps=5, device='cuda'):
    """
    Chạy dự đoán nhiều lần trên cùng 1 tập dữ liệu (với Augmentation khác nhau)
    và lấy trung bình cộng.
    """
    model.eval()

    # Số lượng mẫu trong tập test
    num_samples = len(loader.dataset)

    # Mảng cộng dồn xác suất (Khởi tạo bằng 0)
    accumulated_probs = np.zeros(num_samples)
    final_labels = None

    print(f"🔄 Starting TTA ({tta_steps} views per image)...")

    with torch.no_grad():
        for i in range(tta_steps):
            print(f"   ► View {i + 1}/{tta_steps}")
            step_probs = []
            step_labels = []

            for imgs, labels in loader:
                imgs = imgs.to(device)

                # Forward
                outputs = model(imgs)  # Logits

                # Sigmoid để ra xác suất (0-1)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                step_probs.extend(probs)
                step_labels.extend(labels.cpu().numpy())

            # Cộng dồn xác suất của lần chạy này vào tổng
            accumulated_probs += np.array(step_probs)

            # Lưu labels (Chỉ cần lưu 1 lần vì thứ tự không đổi)
            if final_labels is None:
                final_labels = np.array(step_labels)

    # Chia trung bình
    avg_probs = accumulated_probs / tta_steps
    return final_labels, avg_probs


# --- 3. MAIN FUNCTION ---
# Lưu ý: Hàm này tên là run_tta để main.py gọi đúng
def run_tta(image_size=300, batch_size=32, tta_steps=5):
    # Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running V4 (Inference TTA) on {device}...")

    # MLflow
    os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "dapif865faf65e4f29f9f213de9b6f2ffa3c"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/v4")

    # Load Data (Chỉ cần tập Test)
    CSV_DIR = os.path.join(parent_dir, 'dataset_splits')
    test_path = f'{CSV_DIR}/processed_test.csv'

    if not os.path.exists(test_path):
        print(f"❌ Error: Không tìm thấy {test_path}")
        return

    test_df = pd.read_csv(test_path)
    print(f"📊 Test Data: {len(test_df)} samples")

    # --- DATALOADER CHO TTA ---
    # QUAN TRỌNG 1: is_train=True -> Để BẬT Augmentation (Xoay, Lật...)
    # QUAN TRỌNG 2: shuffle=False -> Để thứ tự ảnh giữ nguyên qua các vòng lặp, cộng dồn đúng index
    tta_loader = DataLoader(
        ISICDataset(test_df, image_size, is_train=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8, pin_memory=True
    )

    # --- LOAD MODEL V3 ---
    # Đường dẫn file checkpoint V3
    ckpt_dir = os.path.join(parent_dir, 'checkpoints')
    model_path = os.path.join(ckpt_dir, "best_v3.pth")

    if not os.path.exists(model_path):
        print(f"❌ Error: Không tìm thấy model V3 tại {model_path}!")
        print("👉 Hãy chạy 'python main.py v3' để train xong V3 trước.")
        return

    print("🏗️ Loading Model V3 structure (EfficientNet-B3 Binary)...")
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=False, num_classes=1)

    print(f"📂 Loading Weights from {model_path}...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.to(device)

    # --- RUN INFERENCE ---
    with mlflow.start_run(run_name="V4_TTA_Inference"):
        log_inference_params(tta_steps)

        # Chạy TTA
        labels, avg_probs = predict_tta(model, tta_loader, tta_steps=tta_steps, device=device)

        # Tính metrics
        metrics = calculate_metrics(labels, avg_probs)

        print("\n" + "=" * 40)
        print(f"🏆 FINAL RESULT (V4 - TTA {tta_steps}x)")
        print(f"pAUC (0.01): {metrics['pauc_0.01']:.4f}")
        print(f"AUC Full   : {metrics['auc']:.4f}")
        print(f"F1 Mal     : {metrics['f1_malignant']:.4f}")
        print(f"Recall Mal : {metrics['recall_malignant']:.4f}")
        print("=" * 40)

        print(classification_report(labels, (avg_probs >= 0.5).astype(int), target_names=['Benign', 'Malignant']))

        # Log metrics
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # Lưu kết quả dự đoán ra file CSV để nộp bài hoặc phân tích
        output_file = "v4_tta_predictions.csv"
        result_df = pd.DataFrame({'label': labels, 'prob_tta': avg_probs})
        result_df.to_csv(output_file, index=False)
        print(f"💾 Saved predictions to {output_file}")