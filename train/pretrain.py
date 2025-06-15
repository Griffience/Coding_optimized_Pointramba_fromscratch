# train/pretrain.py

#路径修正
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from models.tramba_ultra import TrambaUltra
from datasets.shapenetpart import ShapeNetPart
from utils.tools import random_mask_point_groups
from utils.logger import Logger
from utils.config import load_config

def pretrain():
    config = load_config("configs/pretrain.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_cfg = config["experiment"]
    train_cfg = config["train"]
    model_cfg = config["model"]

    output_dir = os.path.join(exp_cfg["save_dir"], exp_cfg["exp_name"])
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Visualize"), exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs_pretrain"))

    # Data
    full_dataset = ShapeNetPart(split='train')
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=4)

    # Model
    model = TrambaUltra(model_cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scaler = GradScaler() if train_cfg["use_amp"] else None

    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: e / max(1, train_cfg["warmup_epochs"]) if e < train_cfg["warmup_epochs"] else 1.0
    )
    cosine = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"] - train_cfg["warmup_epochs"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[train_cfg["warmup_epochs"]])

    best_loss = 1e10

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0

        for pts, _, _ in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            pts = pts.to(device)
            masked_pts, mask_idx = random_mask_point_groups(pts, num_group=model_cfg["num_group"],
                                                             group_size=model_cfg["group_size"],
                                                             mask_ratio=train_cfg["mask_ratio"])
            optimizer.zero_grad()

            with autocast(enabled=train_cfg["use_amp"]):
                pred_center, gt_center = model(masked_pts, return_center=True, mask_idx=mask_idx)
                loss = torch.mean((pred_center - gt_center) ** 2)

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

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pts, _, _ in tqdm(valid_loader, desc=f"[Valid] Epoch {epoch}"):
                pts = pts.to(device)
                masked_pts, mask_idx = random_mask_point_groups(pts, num_group=model_cfg["num_group"],
                                                                 group_size=model_cfg["group_size"],
                                                                 mask_ratio=train_cfg["mask_ratio"])

                pred_center, gt_center = model(masked_pts, return_center=True, mask_idx=mask_idx)
                loss = torch.mean((pred_center - gt_center) ** 2)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Valid", avg_val_loss, epoch)

        Logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", "best_pretrain.pth"))
            Logger.info(f"[Saved Best Model] Best Val Loss = {best_loss:.6f}")

    writer.close()

if __name__ == "__main__":
    pretrain()
