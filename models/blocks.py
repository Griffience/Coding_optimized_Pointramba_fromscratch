import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                batch_first=True,
                norm_first=True,
                activation="gelu"
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.mlp(self.norm(x))

class MambaEncoder(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(dim) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BiCrossFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        """
        Args:
            x1: (B, G, D)
            x2: (B, G, D)
        """
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)

        attn = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = self.proj(out)
        return out + x1
