import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=64, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        x = self.projection(x)  # Shape: [batch, emb_size, h/patch, w/patch]
        x = x.flatten(2).transpose(1, 2)  # Shape: [batch, num_patches, emb_size]
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        batch_size, tokens, emb_size = x.shape
        qkv = self.qkv(x).reshape(batch_size, tokens, 3, self.num_heads, emb_size // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * (emb_size ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, tokens, emb_size)
        return self.projection(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, forward_expansion, dropout):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(emb_size)
        self.mha = MultiHeadAttention(emb_size, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Linear(forward_expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.dropout1(self.mha(self.layernorm1(x)))
        x = x + self.feed_forward(self.layernorm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=64, img_size=32, num_classes=10, depth=7, num_heads=8, forward_expansion=4, dropout=0.0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer_encoders = nn.Sequential(*[TransformerEncoderLayer(emb_size, num_heads, forward_expansion, dropout) for _ in range(depth)])
        self.layernorm = nn.LayerNorm(emb_size)
        self.mlp_head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoders(x)
        x = self.layernorm(x[:, 0])
        return self.mlp_head(x)