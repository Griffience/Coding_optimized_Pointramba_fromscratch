#路径修正
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from models.grouping import farthest_point_sample, square_distance, index_points, batched_index_group

###############################
# 🎨 可视化工具
###############################

def save_ply_color(points, labels, save_path, color_map=None):
    """
    保存点云到 PLY 文件，带颜色。
    
    Args:
        points (np.ndarray): (N, 3)
        labels (np.ndarray): (N,)
        save_path (str): 保存路径
        color_map (np.ndarray): (num_classes, 3)，默认随机生成
    """
    assert points.shape[0] == labels.shape[0], "Points and labels shape mismatch"
    if color_map is None:
        np.random.seed(42)
        color_map = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
    
    colors = color_map[labels % 50]  # 防止越界
    vertices = [tuple(p.tolist() + c.tolist()) for p, c in zip(points, colors)]
    vertices_np = np.array(vertices,
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    el = PlyElement.describe(vertices_np, 'vertex')
    PlyData([el]).write(save_path)
    print(f"[PLY Saved] {save_path}")

def visualize_partseg_batch(batch_points, batch_pred, batch_gt, batch_label, exp_name, save_root="Seg_output/Visualize"):
    """
    批量保存 ShapeNet Part 分割的预测和GT结果。

    Args:
        batch_points (Tensor): (B, N, 3)
        batch_pred (Tensor): (B, N)
        batch_gt (Tensor): (B, N)
        batch_label (Tensor): (B,)
        exp_name (str): 实验名称
    """
    assert exp_name is not None, "exp_name must be provided for visualization saving!"
    class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife',
                     'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    np.random.seed(42)
    color_map = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)

    for i in range(batch_points.shape[0]):
        points = batch_points[i].cpu().numpy()
        pred   = batch_pred[i].cpu().numpy()
        gt     = batch_gt[i].cpu().numpy()
        label  = batch_label[i].item()
        classname = class_choices[label]

        pred_path = os.path.join(save_root, exp_name, classname, f"sample_{i}_pred.ply")
        gt_path   = os.path.join(save_root, exp_name, classname, f"sample_{i}_gt.ply")

        save_ply_color(points, pred, pred_path, color_map=color_map)
        save_ply_color(points, gt, gt_path, color_map=color_map)

###############################
# 🛠️ 自监督Mask辅助工具
###############################

def random_mask_point_groups(points, num_groups=128, group_size=32, mask_ratio=0.4):
    """
    随机 mask 掉点云的 group，用于自监督。
    
    Args:
        points (Tensor): (B, N, 3)
        num_groups (int): 分组数
        group_size (int): 每组大小
        mask_ratio (float): 掩码比例
        
    Returns:
        visible_groups (Tensor): (B, num_visible_groups, group_size, 3)
        masked_groups (Tensor): (B, num_masked_groups, group_size, 3)
        mask_idx (Tensor): (B, num_masked_groups)
    """
    

    B, N, _ = points.shape
    fps_idx = farthest_point_sample(points, num_groups)          # (B, G)
    centers = index_points(points, fps_idx)                      # (B, G, 3)
    dists = square_distance(centers, points)                     # (B, G, N)
    group_idx = dists.argsort()[:, :, :group_size]               # (B, G, S)
    grouped_points = index_points(points, group_idx)             # (B, G, S, 3)

    num_mask = int(num_groups * mask_ratio)
    rand_order = torch.rand(B, num_groups, device=points.device)
    sort_idx = rand_order.argsort(dim=-1)
    mask_idx = sort_idx[:, :num_mask]
    visible_idx = sort_idx[:, num_mask:]

    visible_groups = batched_index_group(grouped_points, visible_idx)
    masked_groups = batched_index_group(grouped_points, mask_idx)

    return visible_groups, masked_groups, mask_idx





def accuracy(pred, target):
    """
    Args:
        pred: (B, num_classes) logits
        target: (B,) ground truth labels
    Returns:
        acc: (float) top-1 accuracy
    """
    pred_class = pred.argmax(dim=-1)
    correct = pred_class.eq(target).sum().item()
    total = target.numel()
    acc = correct / total
    return acc

