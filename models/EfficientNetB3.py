from mlflow.models import infer_signature
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

# Cấu hình MLflow để kết nối với Databricks
os.environ["DATABRICKS_HOST"] = "https://dbc-cba55001-5dea.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi987a9e46da628dbdb4a22949054afa24"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Workspace/Users/nht.master.k20@gmail.com/SkinDiseaseClassificationEFFB3")


def start_mlflow_run(run_name, mode, image_size):
    """Khởi tạo MLflow run với tên cụ thể"""
    run_name = run_name or f"EfficientNetB3_{mode}_{image_size}"
    return mlflow.start_run(run_name=run_name)


def log_training_params(mode, image_size, batch_size, epochs, early_stop_patience,
                        train_size, val_size, test_size, device, lr=1e-4, weight_decay=0.01,
                        class_weights=None, lr_strategy="uniform", gradient_clip=1.0):
    """Log tất cả các tham số training vào MLflow"""
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
    """Log metrics cho mỗi epoch"""
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
    """Log metrics cuối cùng trên test set"""
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "best_val_f1": best_val_f1,
    })


def log_model_artifact(model_path):
    """Log model checkpoint như artifact"""
    mlflow.log_artifact(model_path)


def log_model_registry(model, mode):
    run_name = f"efficientnetb3_skincancer_{mode}_run"
    registered_model_name = f"efficientnetb3_skincancer_{mode}.default.EfficientNetB3"

    # Đặt model về CPU để log ổn định
    model_cpu = model.to("cpu").eval()

    # Tạo input_example đúng định dạng EfficientNetB3
    # (3, 300, 300) hoặc size bạn đang dùng
    dummy_input = torch.randn(1, 3, 300, 300)

    with torch.no_grad():
        dummy_output = model_cpu(dummy_input)

    signature = infer_signature(
        model_input=dummy_input.numpy(),
        model_output=dummy_output.numpy()
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.pytorch.log_model(
            pytorch_model=model_cpu,
            name="model",
            registered_model_name=registered_model_name,
            input_example=dummy_input.numpy(),
            signature=signature
        )

    print(f"Model logged + Registered vào Unity Catalog: {registered_model_name}")


def train_one_epoch(model, loader, optimizer, criterion, scaler, gradient_clip=1.0):
    """
    Train model trong 1 epoch

    Args:
        model: Model cần train
        loader: DataLoader cho training data
        optimizer: Optimizer (AdamW)
        criterion: Loss function (CrossEntropyLoss với class weights)
        scaler: GradScaler cho mixed precision training
        gradient_clip: Giá trị max norm cho gradient clipping

    Returns:
        Average training loss của epoch
    """
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        # Mixed precision training để tối ưu memory và tốc độ
        with autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping để tăng tính ổn định khi training
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, show_report=False):
    """
    Đánh giá model trên validation/test set

    Args:
        model: Model cần đánh giá
        loader: DataLoader cho validation/test data
        criterion: Loss function
        show_report: Có hiển thị classification report chi tiết hay không

    Returns:
        avg_loss, accuracy, macro_precision, macro_recall, macro_f1
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.cuda()
            labels = labels.cuda()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính metrics với macro average để cân bằng giữa các class
    # Quan trọng với imbalanced dataset
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    avg_loss = total_loss / len(loader)

    # Hiển thị metrics chi tiết cho từng class (Benign vs Malignant)
    if show_report:
        print("\n" + "=" * 60)
        print("Per-class Metrics:")
        print("=" * 60)
        print(classification_report(all_labels, all_preds,
                                    target_names=['Benign (0)', 'Malignant (1)'],
                                    digits=4, zero_division=0))

    return avg_loss, accuracy, macro_precision, macro_recall, macro_f1


def train(mode='raw', image_size=300, batch_size=32, epochs=10):
    torch.cuda.empty_cache()

    run_name = f"EfficientNetB3_{mode}_imgsize{image_size}_bs{batch_size}_ep{epochs}_macro"

    # Đường dẫn đến các file CSV chứa dataset splits
    train_path = ""
    val_path = ""
    test_path = ""

    if mode == 'raw':
        train_path = 'dataset_splits/train_raw.csv'
        val_path = 'dataset_splits/val.csv'
        test_path = 'dataset_splits/test.csv'
    elif mode == 'aug':
        train_path = 'dataset_splits_aug/train_augment.csv'
        val_path = 'dataset_splits_aug/val.csv'
        test_path = 'dataset_splits_aug/test.csv'
    elif mode == 'clean':
        train_path = 'dataset_splits_aug_clean/clean_train_augment.csv'
        val_path = 'dataset_splits_aug_clean/clean_val.csv'
        test_path = 'dataset_splits_aug_clean/clean_test.csv'

    # Load dataframes
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val  : {len(val_df)}")
    print(f"  Test : {len(test_df)}")

    # Kiểm tra phân bố class để xác định mức độ imbalance
    print(f"\nClass distribution:")
    train_benign = (train_df['malignant'] == 0).sum()
    train_malignant = (train_df['malignant'] == 1).sum()
    val_benign = (val_df['malignant'] == 0).sum()
    val_malignant = (val_df['malignant'] == 1).sum()
    test_benign = (test_df['malignant'] == 0).sum()
    test_malignant = (test_df['malignant'] == 1).sum()

    print(f"  Train - Benign: {train_benign} ({train_benign / len(train_df) * 100:.1f}%), "
          f"Malignant: {train_malignant} ({train_malignant / len(train_df) * 100:.1f}%)")
    print(f"  Val   - Benign: {val_benign} ({val_benign / len(val_df) * 100:.1f}%), "
          f"Malignant: {val_malignant} ({val_malignant / len(val_df) * 100:.1f}%)")
    print(f"  Test  - Benign: {test_benign} ({test_benign / len(test_df) * 100:.1f}%), "
          f"Malignant: {test_malignant} ({test_malignant / len(test_df) * 100:.1f}%)")

    # Tạo datasets và dataloaders
    train_dataset = ISICDataset(train_df, img_size=image_size)
    val_dataset = ISICDataset(val_df, img_size=image_size)
    test_dataset = ISICDataset(test_df, img_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    # Setup model, optimizer, và các components khác
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2)
    model = model.to(device)

    # Learning rate thấp hơn cho fine-tuning pretrained model
    lr = 1e-4
    weight_decay = 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Tính class weights để xử lý imbalanced dataset
    # Class có ít samples sẽ có weight cao hơn
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=train_df['malignant'].values
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    print(f"\nClass weights: Benign={class_weights[0]:.4f}, Malignant={class_weights[1]:.4f}")

    # Loss function với class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler('cuda')

    # Cosine Annealing scheduler để giảm learning rate dần
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Early stopping parameters
    best_f1 = -1  # Khởi tạo -1 để epoch đầu tiên luôn save model
    patience_counter = 0
    early_stop_patience = 5
    gradient_clip = 1.0

    print(f"\nStarting training:")
    print(f"  Image size : {image_size}")
    print(f"  Batch size : {batch_size}")
    print(f"  Max epochs : {epochs}")
    print(f"  Learning rate : {lr}")
    print(f"  Gradient clip : {gradient_clip}")
    print(f"  Metrics avg : macro")
    print(f"  Device     : {device}\n")

    # Bắt đầu MLflow run để track experiment
    with start_mlflow_run(run_name, mode, image_size):

        # Log tất cả hyperparameters
        log_training_params(
            mode=mode,
            image_size=image_size,
            batch_size=batch_size,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=len(test_df),
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            class_weights=class_weights,
            lr_strategy="uniform",
            gradient_clip=gradient_clip
        )

        # Tạo thư mục lưu checkpoints
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"best_efficientnet_b3_{mode}_{image_size}.pth")

        print(f"Model checkpoint will be saved to: {model_path}\n")

        # Training loop
        for epoch in range(epochs):
            # Train 1 epoch
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, gradient_clip)

            # Validate trên validation set
            val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
                model, val_loader, criterion, show_report=False
            )

            # Cập nhật learning rate theo scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics vào MLflow
            log_epoch_metrics(epoch, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1, current_lr)

            # In kết quả của epoch
            print(f"\nEpoch {epoch + 1}/{epochs} | LR: {current_lr:.2e}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss  : {val_loss:.4f}")
            print(f"  Accuracy  : {val_acc * 100:.2f}%")
            print(f"  Precision : {val_precision * 100:.2f}%")
            print(f"  Recall    : {val_recall * 100:.2f}%")
            print(f"  F1 Score  : {val_f1 * 100:.2f}%")

            # Save best model dựa trên macro F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0

                # Lưu model state, optimizer state, và metrics
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'metrics': {
                        'accuracy': val_acc,
                        'precision': val_precision,
                        'recall': val_recall,
                        'f1': val_f1,
                        'loss': val_loss
                    }
                }, model_path)

                # Log model artifact vào MLflow
                log_model_artifact(model_path)

                print(f"Saved best model! (F1: {val_f1 * 100:.2f}%)")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stop_patience})")

            # Early stopping nếu không có cải thiện sau patience epochs
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                mlflow.log_param("actual_epochs", epoch + 1)
                break

        print(f"\n{'=' * 60}")
        print(f"Training completed! Best F1: {best_f1 * 100:.2f}%")
        print(f"{'=' * 60}")

        # Đánh giá trên test set
        print("\nEvaluating on test set...")

        # Load best model checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            loaded_f1 = checkpoint['best_f1']
            print(f"Loaded best model from epoch {checkpoint['epoch']} (F1: {loaded_f1 * 100:.2f}%)")
        else:
            print(f"Warning: {model_path} not found. Using current model state.")

        # Validate trên test set với per-class report
        test_loss, test_acc, test_precision, test_recall, test_f1 = validate(
            model, test_loader, criterion, show_report=True
        )

        # Log test metrics vào MLflow
        log_test_metrics(test_loss, test_acc, test_precision, test_recall, test_f1, best_f1)

        # In kết quả cuối cùng
        print(f"\n{'=' * 60}")
        print(f"TEST SET RESULTS:")
        print(f"{'=' * 60}")
        print(f"  Accuracy  : {test_acc * 100:.2f}%")
        print(f"  Precision : {test_precision * 100:.2f}%")
        print(f"  Recall    : {test_recall * 100:.2f}%")
        print(f"  F1 Score  : {test_f1 * 100:.2f}%")
        print(f"  Loss      : {test_loss:.4f}")
        print(f"{'=' * 60}\n")

        # Đăng ký model vào MLflow Model Registry
        log_model_registry(model, mode)

        return {
            'best_val_f1': best_f1,
            'test_metrics': {
                'accuracy': test_acc,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1,
                'loss': test_loss
            }
        }