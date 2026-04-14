import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        p = F.softmax(logits, dim=1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_(1e-8, 1.0)
        focal = (-(1-pt)**self.gamma * torch.log(pt)).mean()
        return self.alpha * focal + (1 - self.alpha) * ce


def move_batch_to_device(batch, device):
    g, cont, disc, y = batch
    return g.to(device), cont.to(device), disc.to(device), y.to(device)


def build_criterion(cfg):
    loss_type = getattr(cfg, "loss_type", "ce").lower()
    class_weights = getattr(cfg, "class_weights", None)
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=cfg.device)

    if loss_type == "focal":
        alpha = getattr(cfg, "focal_alpha", 0.5)
        gamma = getattr(cfg, "focal_gamma", 2.0)


        return FocalCrossEntropyLoss(alpha=alpha, gamma=gamma, class_weights=weight_tensor)

    return nn.CrossEntropyLoss(weight=weight_tensor)


def train_self(cfg, model, train_loader, optimizer, scheduler=None, val_loader=None):
    """Standard few-shot training that uses fixed samples per class."""
    model_path=os.path.join(cfg.result_path,f"mmfewshotsids_{cfg.dataset}.pt")
    criterion = build_criterion(cfg).to(cfg.device)
    best_metric = -np.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        loop = tqdm(train_loader, desc=f"Train {epoch}/{cfg.epochs}", leave=False)
        for batch in loop:
            g, cont, disc, y = move_batch_to_device(batch, cfg.device)
            optimizer.zero_grad()
            logits = model(g, cont, disc)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_acc += (preds == y).float().mean().item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / max(1, len(train_loader))
        avg_acc = total_acc / max(1, len(train_loader))
        print(f"[Epoch {epoch}] TrainLoss={avg_loss:.4f} | TrainAcc={avg_acc:.4f}")

        if val_loader is not None:
            metrics = evaluate(model, val_loader, cfg.device)
            metric_score = metrics["F1"]
        else:
            metric_score = avg_acc

        if metric_score > best_metric:
            best_metric = metric_score
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] Saved new best model to {model_path}")

    if best_metric < 0:
        torch.save(model.state_dict(), model_path)


def train_transfer(cfg, model, source_loader, target_loader, optimizer, scheduler=None):
    """Two-stage training mirroring the transfer-enhanced setup."""
    pretrain_epochs = getattr(cfg, "transfer_pretrain_epochs", cfg.epochs)
    finetune_epochs = getattr(cfg, "transfer_finetune_epochs", cfg.epochs)

    original_epochs = cfg.epochs
    cfg.epochs = pretrain_epochs
    train_self(cfg, model, source_loader, optimizer, scheduler=scheduler)

    cfg.epochs = finetune_epochs
    train_self(cfg, model, target_loader, optimizer, scheduler=scheduler)
    cfg.epochs = original_epochs


def evaluate(
    model,
    loader,
    device,
    num_classes=None,
    class_labels=None,
    save_confusion_path="img/confusion_matrix.png",
):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            g, cont, disc, y = move_batch_to_device(batch, device)
            logits = model(g, cont, disc)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            trues.append(y.cpu().numpy())

    if not preds:
        return {"ACC": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if num_classes is not None and class_labels is not None:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        os.makedirs(os.path.dirname(save_confusion_path) or ".", exist_ok=True)
        plt.savefig(save_confusion_path, bbox_inches="tight")
        plt.close(fig)

    return {"ACC": acc, "F1": f1, "Precision": precision, "Recall": recall}
