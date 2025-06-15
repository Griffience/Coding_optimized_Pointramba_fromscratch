#路径修正
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.tramba_ultra import TrambaUltra
from datasets.modelnet40 import ModelNet40
from datasets.scanobjectnn import ScanObjectNN
from utils.logger import Logger
from utils.config import load_config

def main():
    config = load_yaml_config("configs/finetune_cls.yaml")
    exp_cfg, train_cfg, model_cfg = config["experiment"], config["train"], config["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(exp_cfg["save_dir"], exp_cfg["exp_name"])
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Visualize"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))

    if train_cfg["dataset"] == "ModelNet40":
        dataset = ModelNet40(partition="train")
        test_dataset = ModelNet40(partition="test")
    elif train_cfg["dataset"] == "ScanObjectNN":
        dataset = ScanObjectNN(partition="train")
        test_dataset = ScanObjectNN(partition="test")
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=4)

    model = TrambaUltra(model_cfg, task="classification", num_classes=model_cfg["num_classes"]).to(device)
    if train_cfg["pretrain_path"]:
        ckpt = torch.load(train_cfg["pretrain_path"], map_location=device)
        model.load_state_dict(ckpt)
        Logger.info(f"Loaded pretrain weights from {train_cfg['pretrain_path']}")

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if train_cfg["use_amp"] else None

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: e / max(1, train_cfg["warmup_epochs"]))],
        milestones=[train_cfg["warmup_epochs"]],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"] - train_cfg["warmup_epochs"])

    best_acc, no_improve = 0.0, 0

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for pts, label in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            pts, label = pts.to(device), label.to(device)
            optimizer.zero_grad()
            with autocast(enabled=train_cfg["use_amp"]):
                logits = model(pts)
                loss = criterion(logits, label)

            if train_cfg["use_amp"]:
                scaler.scale(loss).backward()
                clip_grad_norm_(model.parameters(), max_norm=train_cfg["clip_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=train_cfg["clip_grad_norm"])
                optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == label).sum().item()
            total += label.size(0)

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for pts, label in tqdm(test_loader, desc=f"[Valid] Epoch {epoch}"):
                pts, label = pts.to(device), label.to(device)
                logits = model(pts)
                correct += (logits.argmax(dim=-1) == label).sum().item()
                total += label.size(0)

        acc = correct / total

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Acc/Valid", acc, epoch)

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", "best_finetune_cls.pth"))
            Logger.info(f"Epoch {epoch} | [Saved Best] Acc={acc:.4f}")
        else:
            no_improve += 1

        Logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Valid Acc: {acc:.4f} | Best: {best_acc:.4f}")

        if no_improve >= train_cfg["early_stop_patience"]:
            Logger.info("[Early Stopping triggered]")
            break

    writer.close()

if __name__ == "__main__":
    main()
