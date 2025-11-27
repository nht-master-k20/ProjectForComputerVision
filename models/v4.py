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
    Dự đoán TTA: Chạy lặp dataset tta_steps lần, mỗi lần với một augmentation ngẫu nhiên khác nhau.
    Sau đó lấy trung bình cộng xác suất.
    """
    model.eval()

    # Chuẩn bị mảng để cộng dồn xác suất
    num_samples = len(loader.dataset)
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
                outputs = model(imgs)  # Logits
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()  # Sigmoid -> Probs

                step_probs.extend(probs)
                step_labels.extend(labels.cpu().numpy())

            # Cộng dồn vào kết quả tổng
            accumulated_probs += np.array(step_probs)

            # Lưu labels ở vòng lặp đầu tiên (các vòng sau label vẫn thế)
            if final_labels is None:
                final_labels = np.array(step_labels)

    # Lấy trung bình
    avg_probs = accumulated_probs / tta_steps
    return final_labels, avg_probs


# --- 3. MAIN V4 ---
def run_tta(image_size=300, batch_size=32, tta_steps=5):
    # Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running V4 (Inference TTA) on {device}...")

    # MLflow Setup
    os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "dapif865faf65e4f29f9f213de9b6f2ffa3c"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/v4")

    # Load Data (Chỉ cần tập Test)
    CSV_DIR = 'dataset_splits'
    test_df = pd.read_csv(f'{CSV_DIR}/processed_test.csv')
    print(f"📊 Test Data: {len(test_df)} samples")

    # --- DATALOADER CHO TTA ---
    # QUAN TRỌNG: is_train=True để kích hoạt Augmentation (Xoay, Lật)
    # QUAN TRỌNG: shuffle=False để đảm bảo thứ tự ảnh không đổi qua các bước TTA
    tta_loader = DataLoader(
        ISICDataset(test_df, image_size, is_train=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8, pin_memory=True
    )

    # --- LOAD MODEL V3 ---
    model_path = "checkpoints/best_v3.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Không tìm thấy model V3! Hãy chạy V3 trước.")

    print("🏗️ Loading Model V3 structure...")
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
        print("=" * 40)

        print(classification_report(labels, (avg_probs >= 0.5).astype(int), target_names=['Benign', 'Malignant']))

        # Log metrics
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # Lưu kết quả dự đoán ra CSV để phân tích sau (nếu cần)
        result_df = pd.DataFrame({'label': labels, 'prob_tta': avg_probs})
        result_df.to_csv("v4_tta_predictions.csv", index=False)
        print("💾 Saved predictions to v4_tta_predictions.csv")


if __name__ == '__main__':
    # Bạn có thể tăng tta_steps lên 8 hoặc 10 nếu thời gian cho phép
    run_tta(tta_steps=5)