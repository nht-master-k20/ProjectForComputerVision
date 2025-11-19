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

os.environ["DATABRICKS_HOST"] = "https://dbc-5d852ff6-7674.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapid40cb896d1a4c41fa62835b61811d2e1"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Workspace/Users/nguyenduonghai07@gmail.com/SkinDiseaseClassificationEFFB3")


def start_mlflow_run(run_name, mode, image_size):
    """Start MLflow run with a specific name"""
    run_name = run_name or f"EfficientNetB3_{mode}_{image_size}"
    return mlflow.start_run(run_name=run_name)


def log_training_params(mode, image_size, batch_size, epochs, early_stop_patience,
                        train_size, val_size, test_size, device, lr=1e-4, weight_decay=0.01,
                        class_weights=None):
    """Log all training parameters to MLflow"""
    params = {
        "model": "tf_efficientnet_b3.ns_jft_in1k",
        "mode": mode,
        "image_size": image_size,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "weight_decay": weight_decay,
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
    """Log metrics for each epoch"""
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
    """Log final test metrics"""
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "best_val_f1": best_val_f1,
    })


def log_model_artifact(model_path):
    """Log model checkpoint as artifact"""
    mlflow.log_artifact(model_path)


def log_model_registry(model, mode):
    """Log model to MLflow Model Registry"""
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name=f"EfficientNetB3_SkinCancer_{mode}"
    )


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, show_report=False):
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

    # Calculate metrics with macro average for balanced evaluation
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    avg_loss = total_loss / len(loader)

    if show_report:
        print("\n" + "=" * 60)
        print("Per-class Metrics:")
        print("=" * 60)
        print(classification_report(all_labels, all_preds,
                                    target_names=['Benign (0)', 'Malignant (1)'],
                                    digits=4, zero_division=0))

    return avg_loss, accuracy, macro_precision, macro_recall, macro_f1


def train(mode='raw', image_size=300, batch_size=128, epochs=30):
    run_name = f"EfficientNetB3_imgsize{image_size}_bs{batch_size}_ep{epochs}_macro"

    train_path = ""
    val_path = ""
    test_path = ""

    if mode == 'raw':
        train_path = 'dataset_splits/train_raw.csv'
        val_path = 'dataset_splits/val.csv'
        test_path = 'dataset_splits/test.csv'
    # elif mode == 'aug':
    #     train_path =
    #     val_path =
    #     test_path =
    # elif mode == 'clean':
    #     train_path =
    #     val_path =
    #     test_path =

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val  : {len(val_df)}")
    print(f"  Test : {len(test_df)}")

    # Check class distribution
    print(f"\nClass distribution:")
    train_benign = (train_df['target'] == 0).sum()
    train_malignant = (train_df['target'] == 1).sum()
    val_benign = (val_df['target'] == 0).sum()
    val_malignant = (val_df['target'] == 1).sum()
    test_benign = (test_df['target'] == 0).sum()
    test_malignant = (test_df['target'] == 1).sum()

    print(f"  Train - Benign: {train_benign} ({train_benign / len(train_df) * 100:.1f}%), "
          f"Malignant: {train_malignant} ({train_malignant / len(train_df) * 100:.1f}%)")
    print(f"  Val   - Benign: {val_benign} ({val_benign / len(val_df) * 100:.1f}%), "
          f"Malignant: {val_malignant} ({val_malignant / len(val_df) * 100:.1f}%)")
    print(f"  Test  - Benign: {test_benign} ({test_benign / len(test_df) * 100:.1f}%), "
          f"Malignant: {test_malignant} ({test_malignant / len(test_df) * 100:.1f}%)")

    train_dataset = ISICDataset(train_df, img_size=image_size)
    val_dataset = ISICDataset(val_df, img_size=image_size)
    test_dataset = ISICDataset(test_df, img_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2)
    model = model.to(device)

    lr = 1e-4
    weight_decay = 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=train_df['target'].values
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    print(f"\nClass weights: Benign={class_weights[0]:.4f}, Malignant={class_weights[1]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler('cuda')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_f1 = -1
    patience_counter = 0
    early_stop_patience = 5

    print(f"\nStarting training:")
    print(f"  Image size : {image_size}")
    print(f"  Batch size : {batch_size}")
    print(f"  Max epochs : {epochs}")
    print(f"  Learning rate : {lr}")
    print(f"  Metrics avg : macro")
    print(f"  Device     : {device}\n")

    # Start MLflow run
    with start_mlflow_run(run_name, mode, image_size):

        # Log all parameters
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
            class_weights=class_weights
        )

        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"best_efficientnet_b3_{mode}_{image_size}.pth")

        print(f"Model checkpoint will be saved to: {model_path}\n")

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
                model, val_loader, criterion, show_report=(epoch == 0)
            )

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics to MLflow
            log_epoch_metrics(epoch, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1, current_lr)

            print(f"\nEpoch {epoch + 1}/{epochs} | LR: {current_lr:.2e}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss  : {val_loss:.4f}")
            print(f"  Accuracy  : {val_acc * 100:.2f}%")
            print(f"  Precision : {val_precision * 100:.2f}%")
            print(f"  Recall    : {val_recall * 100:.2f}%")
            print(f"  F1 Score  : {val_f1 * 100:.2f}%")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0

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

                # Log model artifact to MLflow
                log_model_artifact(model_path)

                print(f"  Saved best model! (F1: {val_f1 * 100:.2f}%)")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stop_patience})")

            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                mlflow.log_param("actual_epochs", epoch + 1)
                break

        print(f"\n{'=' * 60}")
        print(f"Training completed! Best F1: {best_f1 * 100:.2f}%")
        print(f"{'=' * 60}")

        # Test on test set
        print("\nEvaluating on test set...")

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            loaded_f1 = checkpoint['best_f1']
            print(f"Loaded best model from epoch {checkpoint['epoch']} (F1: {loaded_f1 * 100:.2f}%)")
        else:
            print(f"Warning: {model_path} not found. Using current model state.")

        test_loss, test_acc, test_precision, test_recall, test_f1 = validate(
            model, test_loader, criterion, show_report=True
        )

        # Log test metrics
        log_test_metrics(test_loss, test_acc, test_precision, test_recall, test_f1, best_f1)

        print(f"\n{'=' * 60}")
        print(f"TEST SET RESULTS:")
        print(f"{'=' * 60}")
        print(f"  Accuracy  : {test_acc * 100:.2f}%")
        print(f"  Precision : {test_precision * 100:.2f}%")
        print(f"  Recall    : {test_recall * 100:.2f}%")
        print(f"  F1 Score  : {test_f1 * 100:.2f}%")
        print(f"  Loss      : {test_loss:.4f}")
        print(f"{'=' * 60}\n")

        # Log model to MLflow Model Registry
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