"""
SOLVE - ResNet Transfer Learning for Clothing Classification
Fashion-MNIST Dataset | Target Accuracy: 0.95
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
import os
import time

# ============ Configuration ============
CONFIG = {
    "approach": "ResNet18 Transfer Learning v3",
    "iteration": 3,
    "batch_size": 64,
    "epochs": 40,
    "learning_rate": 0.0003,
    "weight_decay": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "changes": "Unfreeze all layers, heavier dropout(0.5), mixup augmentation, longer training",
}


# ============ Model ============
def create_resnet18(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


def mixup_data(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


# ============ Data ============
def get_data_loaders(batch_size):
    transform_train = transforms.Compose([
        transforms.Resize(96),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.RandomErasing(p=0.2),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# ============ Training ============
def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_mixup:
            mixed_images, targets_a, targets_b, lam = mixup_data(images, labels)
            outputs = model(mixed_images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        outputs_clean = model(images) if use_mixup else outputs
        correct += (outputs_clean.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    print(f"=== SOLVE ResNet - Iteration {CONFIG['iteration']} ===")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")
    print(f"Device: {CONFIG['device']}")

    device = torch.device(CONFIG["device"])
    train_loader, test_loader = get_data_loaders(CONFIG["batch_size"])

    model = create_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    best_acc = 0
    history = []
    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | LR: {lr:.6f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc, "lr": lr,
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Target: 0.9500")
    print(f"{'PASSED' if best_acc >= 0.95 else 'NOT REACHED'}")
    print(f"Training Time: {elapsed:.1f}s")

    results = {
        "approach": CONFIG["approach"], "iteration": CONFIG["iteration"],
        "best_accuracy": best_acc, "target": 0.95, "passed": best_acc >= 0.95,
        "parameters": param_count, "training_time_seconds": elapsed,
        "config": CONFIG, "history": history,
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/iteration_{CONFIG['iteration']}.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_acc


if __name__ == "__main__":
    main()
