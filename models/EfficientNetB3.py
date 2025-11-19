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


def train(mode='raw', image_size=300, batch_size=32, epochs=10, ddp=False, local_rank=None):
    """
    Main train function compatible with main.py calling signature.
    - ddp: bool -> whether to initialize distributed training (torchrun)
    - local_rank: int -> local GPU id for this process (set by torchrun)
    """
    torch.cuda.empty_cache()

    # Initialize DDP if requested
    if ddp:
        local_rank = maybe_init_ddp(ddp, local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        local_rank = None
        world_size = 1
        rank = 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and local_rank is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    run_name = f"EfficientNetB3_{mode}_imgsize{image_size}_bs{batch_size}_ep{epochs}_macro"

    # paths based on mode
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
        raise ValueError("Unknown mode: " + str(mode))

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if is_main_process():
        print(f"Dataset sizes:")
        print(f"  Train: {len(train_df)}")
        print(f"  Val  : {len(val_df)}")
        print(f"  Test : {len(test_df)}")

    # class distribution print only on main process
    if is_main_process():
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

    # datasets + samplers
    train_dataset = ISICDataset(train_df, img_size=image_size)
    val_dataset = ISICDataset(val_df, img_size=image_size)
    test_dataset = ISICDataset(test_df, img_size=image_size)

    if ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=8, pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                 num_workers=8, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=8, pin_memory=True)

    # model, optimizer, criterion, scaler
    model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=2)
    model = model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    lr = 1e-3
    weight_decay = 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=train_df['malignant'].values
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    if is_main_process():
        print(f"\nClass weights: Benign={class_weights[0]:.4f}, Malignant={class_weights[1]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()  # no device arg

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_f1 = -1
    patience_counter = 0
    early_stop_patience = 5
    gradient_clip = 1.0

    if is_main_process():
        print(f"\nStarting training:")
        print(f"  Image size : {image_size}")
        print(f"  Batch size : {batch_size}")
        print(f"  Max epochs : {epochs}")
        print(f"  Learning rate : {lr}")
        print(f"  Gradient clip : {gradient_clip}")
        print(f"  Metrics avg : macro")
        print(f"  Device     : {device}\n")

    # MLflow run only started on main process to avoid duplicate runs
    if is_main_process():
        mlflow_run = start_mlflow_run(run_name, mode, image_size)
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
    else:
        mlflow_run = None

    checkpoint_dir = "checkpoints"
    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
    # barrier so that dir exists before other ranks try to save (if they ever do)
    if ddp and dist.is_initialized():
        dist.barrier()

    model_path = os.path.join(checkpoint_dir, f"best_efficientnet_b3_{mode}_{image_size}.pth")

    for epoch in range(epochs):
        if ddp:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, gradient_clip, ddp=ddp)

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, show_report=False, ddp=ddp)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics only from main process
        if is_main_process():
            log_epoch_metrics(epoch, train_loss, val_loss, val_acc, val_recall, val_precision, val_f1, current_lr)

        if is_main_process():
            print(f"\nEpoch {epoch + 1}/{epochs} | LR: {current_lr:.2e}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss  : {val_loss:.4f}")
            print(f"  Accuracy  : {val_acc * 100:.2f}%")
            print(f"  Precision : {val_precision * 100:.2f}%")
            print(f"  Recall    : {val_recall * 100:.2f}%")
            print(f"  F1 Score  : {val_f1 * 100:.2f}%")

        # Save best model only on main process
        if is_main_process():
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # save full checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    # if DDP, save model.module.state_dict()
                    'model_state_dict': (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
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
                log_model_artifact(model_path)
                print(f"Saved best model! (F1: {val_f1 * 100:.2f}%)")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stop_patience})")

            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                mlflow.log_param("actual_epochs", epoch + 1)
                break

    if is_main_process():
        print(f"\n{'=' * 60}")
        print(f"Training completed! Best F1: {best_f1 * 100:.2f}%")
        print(f"{'=' * 60}")

    # Evaluate on test set
    if is_main_process():
        print("\nEvaluating on test set...")

    # load best checkpoint on main process
    if is_main_process() and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # load into underlying model (if ddp we load into model.module)
        target = model.module if isinstance(model, DDP) else model
        target.load_state_dict(checkpoint['model_state_dict'])
        loaded_f1 = checkpoint.get('best_f1', None)
        print(f"Loaded best model from epoch {checkpoint.get('epoch', '?')} (F1: {loaded_f1 * 100 if loaded_f1 else 'N/A'}%)")
    elif is_main_process():
        print(f"Warning: {model_path} not found. Using current model state.")

    # For evaluation, broadcast weights from main process to others if DDP active so each rank has same weights
    if ddp and dist.is_initialized():
        # let main process have the most up-to-date weights; broadcast them to others
        # save temp state dict on rank0 and broadcast tensors
        if is_main_process():
            state_dict = (model.module.state_dict() if isinstance(model, DDP) else model.state_dict())
        else:
            state_dict = None
        # broadcast using torch.distributed.broadcast_object_list if available
        try:
            all_state = [state_dict]
            dist.broadcast_object_list(all_state, src=0)
            state_dict = all_state[0]
            target = model.module if isinstance(model, DDP) else model
            target.load_state_dict(state_dict)
        except Exception:
            pass  # best-effort

    test_loss, test_acc, test_precision, test_recall, test_f1 = validate(model, test_loader, criterion, show_report=True, ddp=ddp)

    # Only main process logs test metrics and register model
    if is_main_process():
        log_test_metrics(test_loss, test_acc, test_precision, test_recall, test_f1, best_f1)
        # Register model using underlying module if DDP
        model_for_registry = model.module if isinstance(model, DDP) else model
        log_model_registry(model_for_registry, mode)

    # cleanup ddp
    if ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

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