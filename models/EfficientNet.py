from torch.utils.data import DataLoader
from scripts.get_dataset import get_dataset
import timm
import torch.nn as nn
import torch
from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.cuda()
            labels = labels.cuda()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return total_loss / len(loader), acc


def train(mode='raw', image_size=384, batch_size=16, epochs=10):
    train_path = ""
    val_path = ""
    test_path = ""

    if mode == 'raw':
        train_path =
        val_path =
        test_path =
    elif mode == 'aug':
        train_path =
        val_path =
        test_path =
    elif mode == 'clean':
        train_path =
        val_path =
        test_path =

    train_dataset, val_dataset, test_dataset = get_dataset(train_path, val_path, test_path, image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = timm.create_model("tf_efficientnet_b3_ns", pretrained=True, num_classes=2)
    model = model.to("cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss  : {val_loss:.4f}")
        print(f"  Val Acc   : {val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_effnet.pth")
            print("Saved best model!")