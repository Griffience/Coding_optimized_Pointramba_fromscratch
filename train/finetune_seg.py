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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from models.tramba_ultra import TrambaUltra
from datasets.shapenetpart import ShapeNetPart
from utils.seg_loss import SegLoss
from utils.logger import Logger
from utils.tools import visualize_partseg_batch
from utils.config import load_config

def calculate_shape_IoU(pred_np, seg_np, label, start_index, seg_num):
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        parts = range(start_index[label[shape_idx]], start_index[label[shape_idx]] + seg_num[label[shape_idx]])
        part_ious = []
        for part in parts:
            I = np.sum((pred_np[shape_idx] == part) & (seg_np[shape_idx] == part))
            U = np.sum((pred_np[shape_idx] == part) | (seg_np[shape_idx] == part))
            part_ious.append(1.0 if U == 0 else I / float(U))
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def main():
    config = load_yaml_config("configs/finetune_seg.yaml")
    exp_cfg, train_cfg, model_cfg, meta_cfg = config["experiment"], config["train"], config["model"], config["meta"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(exp_cfg["save_dir"], exp_cfg["exp_name"])
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Visualize"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))

    train_loader = DataLoader(ShapeNetPart(split="trainval"), batch_size=train_cfg["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(ShapeNetPart(split="test"), batch_size=train_cfg["batch_size"], shuffle=False, num_workers=4)

    model = TrambaUltra(model_cfg, task="segmentation", num_classes=model_cfg["seg_num_all"]).to(device)
    if train_cfg["pretrain_path"]:
        ckpt = torch.load(train_cfg["pretrain_path"], map_location=device)
        model.load_state_dict(ckpt)
        Logger.info(f"Loaded pretrain weights from {train_cfg['pretrain_path']}")

    if train_cfg["freeze_backbone"]:
        for name, param in model.named_parameters():
            if "task_head" not in name:
                param.requires_grad = False
        Logger.info("Backbone frozen.")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: e / max(1, train_cfg["warmup_epochs"]))],
        milestones=[train_cfg["warmup_epochs"]],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"] - train_cfg["warmup_epochs"])

    criterion = SegLoss()
    scaler = GradScaler() if train_cfg["use_amp"] else None

    best_miou = 0.0
    no_improve = 0

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0

        for pts, label, seg in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            pts, label, seg = pts.to(device), label.to(device), seg.to(device)
            optimizer.zero_grad()
            with autocast(enabled=train_cfg["use_amp"]):
                seg_pred = model(pts)
                loss = criterion(seg_pred.view(-1, model_cfg["seg_num_all"]), seg.view(-1))

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

        model.eval()
        all_preds, all_segs, all_labels = [], [], []
        with torch.no_grad():
            for pts, label, seg in tqdm(test_loader, desc=f"[Valid] Epoch {epoch}"):
                pts, label, seg = pts.to(device), label.to(device), seg.to(device)
                seg_pred = model(pts)
                preds = seg_pred.argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_segs.append(seg.cpu())
                all_labels.append(label.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_segs = torch.cat(all_segs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        miou = np.mean(calculate_shape_IoU(all_preds, all_segs, all_labels, meta_cfg["start_index"], meta_cfg["seg_num"]))

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("mIoU/Valid", miou, epoch)

        if epoch % exp_cfg["vis_interval"] == 0:
            visualize_partseg_batch(pts.cpu(), preds, seg.cpu(), label.cpu(), exp_name=exp_cfg["exp_name"])

        if miou > best_miou:
            best_miou = miou
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", "best_finetune_seg.pth"))
            Logger.info(f"Epoch {epoch} | [Saved Best] mIoU={miou:.4f}")
        else:
            no_improve += 1

        Logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | mIoU: {miou:.4f} | Best: {best_miou:.4f}")
        if no_improve >= train_cfg["early_stop_patience"]:
            Logger.info("[Early Stopping triggered]")
            break

    writer.close()

if __name__ == "__main__":
    main()
