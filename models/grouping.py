import torch

def square_distance(src, dst):
    """
    src: (B, N, 3)
    dst: (B, M, 3)
    return: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    """
    Args:
        xyz: (B, N, 3)
        npoint: int
    Returns:
        centroids: (B, npoint)
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Args:
        points: (B, N, C)
        idx: (B, S)
    Returns:
        new_points: (B, S, C)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape.append(points.shape[-1])
    idx_flat = idx.view(B, -1)
    new_points = torch.gather(points, 1, idx_flat.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return new_points.view(*view_shape)

class GroupDivider(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            center: (B, num_group, 3)
            group_feature: (B, num_group, 3)
        """
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.num_group)
        center = index_points(xyz, fps_idx)

        dists = square_distance(center, xyz)
        group_idx = dists.argsort(dim=-1)[..., :self.group_size]

        grouped_xyz = index_points(xyz, group_idx)  # (B, num_group, group_size, 3)
        group_feature = grouped_xyz.mean(dim=2)     # (B, num_group, 3)
        return center, group_feature
