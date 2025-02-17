import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbed(nn.Module):
    """Splits the image into patches and embeds them.
       For CIFAR-10, we use a smaller patch size.
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # e.g., (32/4)^2 = 64
        # Convolutional projection for non-overlapping patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Learnable positional embeddings to encode spatial information
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)                # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = x + self.pos_embed
        return x

class LocalUpdate(nn.Module):
    """Local update using a simple per-token MLP."""
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x):
        # x: [B, N, embed_dim]
        return self.fc(x)

class TokenMixer(nn.Module):
    """
    Token mixing module that lets tokens communicate without using self-attention.
    It first normalizes each token embedding, then mixes information across tokens 
    by applying an MLP along the token dimension.
    """
    def __init__(self, num_tokens, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(num_tokens, num_tokens),
            nn.GELU(),
            nn.Linear(num_tokens, num_tokens)
        )
    
    def forward(self, x):
        # x: [B, N, embed_dim]
        x_norm = self.norm(x)
        x_trans = x_norm.transpose(1, 2)  # [B, embed_dim, N]
        x_mixed = self.mlp(x_trans)
        x_mixed = x_mixed.transpose(1, 2)  # [B, N, embed_dim]
        return x_mixed

class SimpleAttention(nn.Module):
    """
    Simple linear attention module.
    Projects the input into query, key, and value spaces.
    Then applies a positive feature map (ELU + 1) and computes attention in linear time:
      output = (φ(Q) ( (φ(K)^T V) )) / (φ(Q) (φ(K)^T 1))
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x: [B, N, D]
        Q = self.W_Q(x)  # [B, N, D]
        K = self.W_K(x)  # [B, N, D]
        V = self.W_V(x)  # [B, N, D]
        # Apply a positive feature map (ELU + 1)
        Q_prime = F.elu(Q) + 1  # [B, N, D]
        K_prime = F.elu(K) + 1  # [B, N, D]
        # Compute the key-value summary: [B, D, D]
        KV = torch.einsum('bnd,bne->bde', K_prime, V)
        # Compute the numerator: [B, N, D]
        numerator = torch.einsum('bnd,bde->bne', Q_prime, KV)
        # Compute the normalization factor:
        # First, sum K_prime over tokens: [B, D]
        K_sum = K_prime.sum(dim=1)
        normalization = torch.einsum('bnd,bd->bn', Q_prime, K_sum)  # [B, N]
        normalization = normalization.unsqueeze(-1)  # [B, N, 1]
        eps = 1e-6
        out = numerator / (normalization + eps)
        return out

class DiffuRNNLayer(nn.Module):
    """
    A single DiffuRNN layer updating token representations with:
      - A diffusion update using a learnable kernel.
      - A local update (per-token MLP).
      - A token mixing update (MLP across tokens).
      - A simple linear attention update.
      - A feed-forward block.
    """
    def __init__(self, num_tokens, embed_dim):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        
        # Learnable diffusion kernel: shape [num_tokens, num_tokens]
        self.K = nn.Parameter(torch.randn(num_tokens, num_tokens))
        # Learnable time-step parameter controlling the diffusion rate.
        self.delta_t = nn.Parameter(torch.tensor(0.1))
        
        # Local update module.
        self.local_update = LocalUpdate(embed_dim)
        # Token mixing update module.
        self.token_mixer = TokenMixer(num_tokens, embed_dim)
        # Simple linear attention module.
        self.simple_attn = SimpleAttention(embed_dim)
        # Feed-forward block.
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Layer normalization for stability.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [B, N, embed_dim]
        Diffusion update: delta_t * (K_pos @ x - (row_sum * x)),
        where K_pos is the non-negative kernel and row_sum is the sum over rows.
        """
        B, N, D = x.shape
        
        # Ensure the kernel is non-negative.
        K_pos = F.softplus(self.K)  # shape: [N, N]
        row_sum = K_pos.sum(dim=1, keepdim=True)  # [N, 1]
        diff_update = self.delta_t * (
            torch.einsum('ij,bjd->bid', K_pos, x) - (row_sum * x)
        )
        
        local_update = self.local_update(x)
        token_update = self.token_mixer(x)
        attn_update = self.simple_attn(x)
        
        x = x + diff_update + local_update + token_update + attn_update
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        
        return x

class DEN(nn.Module):
    """
    Vision model using DiffuRNN layers with the novel simple linear attention module.
    For CIFAR-10, images are 32x32, and we set:
      - patch_size=4 (yielding 64 tokens),
      - embed_dim=192,
      - depth=6 layers,
      - num_classes=10.
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=6):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_tokens = self.patch_embed.num_patches
        
        self.layers = nn.ModuleList([
            DiffuRNNLayer(num_tokens, embed_dim)
            for _ in range(depth)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x)  # [B, num_tokens, embed_dim]
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)       # [B, embed_dim, num_tokens]
        x = self.pool(x).squeeze(-1) # [B, embed_dim]
        x = self.head(x)
        return x


