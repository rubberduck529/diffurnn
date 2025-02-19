import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_sinusoid_encoding(n_position, d_model):
    """Generate sinusoidal positional embeddings.
       Returns a tensor of shape [n_position, d_model].
    """
    pe = torch.zeros(n_position, d_model)
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class TokenEmbed(nn.Module):
    """
    Embeds input token IDs and adds sinusoidal positional embeddings.
    Suitable for NLP tasks such as those in GLUE.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the token embeddings.
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids):
        # input_ids: [B, seq_length]
        x = self.token_embedding(input_ids)  # [B, seq_length, embed_dim]
        pos_embed = get_sinusoid_encoding(x.size(1), x.size(2)).to(x.device)
        x = x + pos_embed.unsqueeze(0)
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
    It normalizes token embeddings and then applies depth‑wise 1D convolutions
    to mix information along the sequence dimension. This design works for 
    arbitrary sequence lengths.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        # Two depth‑wise conv layers with kernel size 3 (padding=1) mix tokens.
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, 
                      kernel_size=3, padding=1, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, 
                      kernel_size=3, padding=1, groups=embed_dim)
        )
    
    def forward(self, x):
        # x: [B, N, embed_dim]
        x_norm = self.norm(x)
        x_trans = x_norm.transpose(1, 2)  # [B, embed_dim, N]
        x_mixed = self.conv(x_trans)
        x_mixed = x_mixed.transpose(1, 2)  # [B, N, embed_dim]
        return x_mixed

class SimpleAttention(nn.Module):
    """
    Simple linear attention module.
    Projects input into query, key, and value spaces and applies a positive feature map.
    Computes attention in linear time.
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
        Q_prime = F.elu(Q) + 1  
        K_prime = F.elu(K) + 1  
        # Compute key-value summary: [B, D, D]
        KV = torch.einsum('bnd,bne->bde', K_prime, V)
        # Numerator: [B, N, D]
        numerator = torch.einsum('bnd,bde->bne', Q_prime, KV)
        # Normalization factor:
        K_sum = K_prime.sum(dim=1)  # [B, D]
        normalization = torch.einsum('bnd,bd->bn', Q_prime, K_sum).unsqueeze(-1)  # [B, N, 1]
        eps = 1e-6
        out = numerator / (normalization + eps)
        return out

class DiffuRNNLayer(nn.Module):
    """
    A single DiffuRNN layer that updates token representations with:
      - A diffusion update (approximated by a depth‑wise conv).
      - A local update (per-token MLP).
      - A token mixing update (1D convolution across tokens).
      - A simple linear attention update.
      - A feed‑forward block.
    This design supports arbitrary sequence lengths.
    """
    def __init__(self, embed_dim):
        super().__init__()
        # Replace the fixed kernel with a depth‑wise conv to approximate diffusion.
        self.delta_t = nn.Parameter(torch.tensor(0.1))
        self.diffusion_conv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, 
                                        kernel_size=3, padding=1, groups=embed_dim)
        
        self.local_update = LocalUpdate(embed_dim)
        self.token_mixer = TokenMixer(embed_dim)
        self.simple_attn = SimpleAttention(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [B, N, embed_dim]
        Diffusion update approximated by:
            delta_t * (diffusion_conv(x) - x)
        """
        # Apply diffusion (simulate message passing from neighboring tokens)
        x_conv = self.diffusion_conv(x.transpose(1, 2)).transpose(1, 2)
        diff_update = self.delta_t * (x_conv - x)
        
        local_update = self.local_update(x)
        token_update = self.token_mixer(x)
        attn_update = self.simple_attn(x)
        
        x = x + diff_update + local_update + token_update + attn_update
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x

class DiffuRNNForGLUE(nn.Module):
    """
    A model for GLUE tasks that uses DiffuRNN layers and simple linear attention.
    Instead of processing image patches, it embeds token IDs (using nn.Embedding)
    and adds sinusoidal positional encodings.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        num_classes (int): Number of output classes (e.g. for classification).
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of DiffuRNN layers.
    """
    def __init__(self, vocab_size, num_classes, embed_dim=192, depth=6):
        super().__init__()
        self.token_embed = TokenEmbed(vocab_size, embed_dim)
        # The sequence length is determined at runtime.
        self.layers = nn.ModuleList([
            DiffuRNNLayer(embed_dim)
            for _ in range(depth)
        ])
        # Pool across the token dimension (e.g. averaging over all tokens)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, input_ids):
        # input_ids: [B, seq_length]
        x = self.token_embed(input_ids)  # [B, seq_length, embed_dim]
        for layer in self.layers:
            x = layer(x)
        # Pool across the token dimension.
        x = x.transpose(1, 2)       # [B, embed_dim, seq_length]
        x = self.pool(x).squeeze(-1) # [B, embed_dim]
        x = self.head(x)            # [B, num_classes]
        return x
