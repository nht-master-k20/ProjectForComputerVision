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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
                        train_size, val_size, test_size, device, lr=1e-3, weight_decay=0.01,
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
    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        dummy_output = model_cpu(dummy_input)
    signature = infer_signature(model_input=dummy_input.numpy(), model_output=dummy_output.numpy())
    with mlflow.start_run(run_name=run_name):
        mlflow.pytorch.log_model(
            pytorch_model=model_cpu,
            name="model",
            registered_model_name=registered_model_name,
            input_example=dummy_input.numpy(),
            signature=signature
        )
    if is_main_process():
        print(f"Model logged + Registered vào Unity Catalog: {registered_model_name}")


# helper: check đang ở process chính (rank 0) hoặc không dùng DDP
def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# helper: gather lists across processes when DDP active
def gather_lists_across_ranks(local_list):
    """
    If dist is initialized, gather Python objects across ranks using all_gather_object.
    Returns flattened combined list on ALL ranks (so each rank has full list). We will
    usually only use it on rank 0 for computing metrics/logging.
    """
    if not dist.is_available() or not dist.is_initialized():
        return local_list

    gathered = [None for _ in range(dist.get_world_size())]
    try:
        dist.all_gather_object(gathered, local_list)
    except Exception:
        # fallback: if all_gather_object not available, return local only
        return local_list
    # flatten
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

    # sync loss across ranks: sum of total_loss and count
    if ddp and dist.is_initialized():
        t_loss = torch.tensor(total_loss, device='cuda')
        t_count = torch.tensor(count, device='cuda')
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
        total_loss = t_loss.item()
        count = t_count.item()

    avg_loss = total_loss / max(1, count)
    return avg_loss


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

    # gather preds/labels across ranks
    if ddp and dist.is_initialized():
        gathered_preds = gather_lists_across_ranks(all_preds)
        gathered_labels = gather_lists_across_ranks(all_labels)
        # gather total_loss and counts to compute average loss across all ranks
        local_count = len(all_labels)
        t_loss = torch.tensor(total_loss, device='cuda')
        t_count = torch.tensor(local_count, device='cuda')
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
        total_loss = t_loss.item()
        total_count = t_count.item()
        avg_loss = total_loss / max(1, total_count)
        # compute metrics on rank 0 (or every rank — same numbers)
        accuracy = 0.0; macro_precision = 0.0; macro_recall = 0.0; macro_f1 = 0.0
        if is_main_process():
            accuracy = accuracy_score(gathered_labels, gathered_preds)
            macro_precision = precision_score(gathered_labels, gathered_preds, average='macro', zero_division=0)
            macro_recall = recall_score(gathered_labels, gathered_preds, average='macro', zero_division=0)
            macro_f1 = f1_score(gathered_labels, gathered_preds, average='macro', zero_division=0)
            if show_report:
                print("\n" + "=" * 60)
                print("Per-class Metrics (gathered across ranks):")
                print("=" * 60)
                print(classification_report(gathered_labels, gathered_preds,
                                            target_names=['Benign (0)', 'Malignant (1)'],
                                            digits=4, zero_division=0))
    else:
        avg_loss = total_loss / max(1, len(loader))
        accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
        macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
        if show_report:
            print("\n" + "=" * 60)
            print("Per-class Metrics:")
            print("=" * 60)
            print(classification_report(all_labels, all_preds,
                                        target_names=['Benign (0)', 'Malignant (1)'],
                                        digits=4, zero_division=0))

    return avg_loss, accuracy, macro_precision, macro_recall, macro_f1


def maybe_init_ddp(ddp, local_rank):
    if ddp:
        # init process group using env:// (torchrun will set env variables)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return local_rank
    return None


def train(mode='raw', image_size=300, batch_size=32, epochs=10):
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # paths
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
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # datasets
    train_loader = DataLoader(ISICDataset(train_df, img_size=image_size),
                              batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_df, img_size=image_size),
                            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(ISICDataset(test_df, img_size=image_size),
                             batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # model
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2)
    model = model.to(device)

    # optimizer, criterion, scheduler
    lr = 1e-3
    weight_decay = 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['malignant'].values)
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # MLflow run
    if is_main_process():
        run_name = f"EfficientNetB3_{mode}_{image_size}_bs{batch_size}_ep{epochs}"
        mlflow_run = start_mlflow_run(run_name, mode, image_size)
        log_training_params(mode, image_size, batch_size, epochs,
                            early_stop_patience=5,
                            train_size=len(train_df),
                            val_size=len(val_df),
                            test_size=len(test_df),
                            device=device,
                            lr=lr,
                            weight_decay=weight_decay,
                            class_weights=class_weights)
    else:
        mlflow_run = None

    # training loop
    best_f1 = -1
    patience_counter = 0
    gradient_clip = 1.0

    model_path = f"checkpoints/best_efficientnet_b3_{mode}_{image_size}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, gradient_clip)
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, show_report=False)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if is_main_process():
            log_epoch_metrics(epoch, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1, current_lr)
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss {train_loss:.4f} | Val F1 {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_f1': best_f1}, model_path)
                log_model_artifact(model_path)
            else:
                patience_counter += 1

            if patience_counter >= 5:
                print("Early stopping")
                mlflow.log_param("actual_epochs", epoch + 1)
                break

    # final test
    if is_main_process():
        print("Evaluating on test set...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_precision, test_recall, test_f1 = validate(model, test_loader, criterion,
                                                                             show_report=True)
        log_test_metrics(test_loss, test_acc, test_precision, test_recall, test_f1, best_f1)
        log_model_registry(model, mode)
