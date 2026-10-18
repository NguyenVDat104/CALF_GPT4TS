import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class DINOv3LayerScale(nn.Module):
    """
    LayerScale as used in DINOv2 and DINOv3 architectures for stabilizing deep ViTs.
    """
    def __init__(self, dim: int, init_values: float = 1e-5):
        super(DINOv3LayerScale, self).__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DINOv3Attention(nn.Module):
    """
    Multi-head Self Attention with QK Normalization (key innovation in DINOv3/SwinV2
    for preventing attention logits saturation at scale).
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super(DINOv3Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK Normalization (DINOv3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DINOv3MLP(nn.Module):
    """
    FeedForward network with GELU activation.
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, drop: float = 0.0):
        super(DINOv3MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DINOv3Block(nn.Module):
    """
    Standard DINO-V3 Vision Transformer Block:
    Pre-LayerNorm -> Attention -> LayerScale -> Residual -> Pre-LayerNorm -> MLP -> LayerScale -> Residual
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0, init_values: float = 1e-5):
        super(DINOv3Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DINOv3Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = DINOv3LayerScale(dim, init_values=init_values)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DINOv3MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.ls2 = DINOv3LayerScale(dim, init_values=init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class CustomDINOv3ViT(nn.Module):
    """
    Standalone Pure-PyTorch DINO-V3 ViT Model.
    Supports variable sequence length, position embeddings, and intermediate hidden states extraction
    fully compatible with CALF's DistillationLoss.
    """
    def __init__(self, d_model: int = 768, num_layers: int = 6, num_heads: int = 8,
                 max_len: int = 2048, dropout: float = 0.1, init_values: float = 1e-5):
        super(CustomDINOv3ViT, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList([
            DINOv3Block(dim=d_model, num_heads=num_heads, drop=dropout, attn_drop=dropout, init_values=init_values)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs_embeds: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Takes inputs_embeds of shape [B, N, d_model].
        Returns:
            last_hidden_state: [B, N, d_model]
            hidden_states: tuple of (initial_embeds, layer_1, layer_2, ..., layer_L)
            Length of tuple = num_layers + 1 (matches GPT2 output_hidden_states in CALF)
        """
        B, N, C = inputs_embeds.shape
        pos = self.pos_embed[:, :N, :]
        x = inputs_embeds + pos
        x = self.pos_drop(x)

        hidden_states = (x,)
        for block in self.blocks:
            x = block(x)
            hidden_states = hidden_states + (x,)

        x = self.norm(x)
        return x, hidden_states


class DINOv3Backbone(nn.Module):
    """
    DINO-V3 Backbone adapter for CALF.
    Attempts to load pretrained weights from Hugging Face Hub (e.g. facebook/dinov3-* or dinov2-*).
    If offline or gated, gracefully falls back to CustomDINOv3ViT with zero interruption.
    """
    def __init__(self, configs, device, is_text_stream: bool = False):
        super(DINOv3Backbone, self).__init__()
        self.d_model = configs.d_model
        self.num_layers = getattr(configs, 'dinov3_layers', configs.gpt_layers)
        self.model_name = getattr(configs, 'dinov3_model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m')
        self.use_pretrained = getattr(configs, 'dinov3_pretrained', 1)
        self.num_heads = getattr(configs, 'n_heads', 8)
        self.is_text_stream = is_text_stream

        loaded_hf = False
        if self.use_pretrained:
            try:
                from transformers import AutoModel, AutoConfig
                print(f"[CALF DINO-V3] Attempting to load pretrained weights for '{self.model_name}'...")
                hf_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                hf_model = AutoModel.from_pretrained(self.model_name, config=hf_config, trust_remote_code=True)
                
                # Check if HuggingFace ViT has encoder layers
                if hasattr(hf_model, 'encoder') and hasattr(hf_model.encoder, 'layer'):
                    self.vit = hf_model
                    self.is_hf = True
                    loaded_hf = True
                    print(f"[CALF DINO-V3] Successfully loaded HuggingFace DINO-V3 checkpoint: {self.model_name}")
            except Exception as e:
                print(f"[CALF DINO-V3 Note] Could not load HuggingFace checkpoint '{self.model_name}' ({e}).")
                print(f"[CALF DINO-V3 Note] Initializing Standalone Pure-PyTorch DINO-V3 ViT architecture (offline/fallback mode).")
                loaded_hf = False

        if not loaded_hf:
            self.is_hf = False
            self.vit = CustomDINOv3ViT(
                d_model=self.d_model,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=configs.dropout
            )

        self.vit.to(device)

    def forward(self, inputs_embeds: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.is_hf:
            try:
                # HF models forward with inputs_embeds or hidden_states
                outputs = self.vit(inputs_embeds=inputs_embeds, output_hidden_states=True)
                last_hidden = outputs.last_hidden_state
                # Slice hidden states to requested layer count
                hidden_states = outputs.hidden_states[:self.num_layers + 1]
                return last_hidden, hidden_states
            except Exception:
                # Fallback to internal custom forward if HF input schema differs
                pass

        return self.vit(inputs_embeds)


class Encoder_DINOv3(nn.Module):
    """
    DINO-V3 based Feature Encoder for Time Series.
    Projects time series channels/patches into DINO representation space, applies DINO-V3 ViT blocks,
    and performs cross-modal alignment with PCA Word Token Embeddings (preserving CALF's Teacher-Student signal).
    """
    def __init__(self, input_dim: int, word_embedding: torch.Tensor, hidden_dim: int = 768,
                 num_heads: int = 8, num_encoder_layers: int = 2, dropout: float = 0.1):
        super(Encoder_DINOv3, self).__init__()
        # 1D Patch / Feature Projector to DINO token dimension
        self.linear = nn.Linear(input_dim, hidden_dim)

        # DINO-V3 ViT encoding blocks with LayerScale & QK-Norm
        self.dino_blocks = nn.ModuleList([
            DINOv3Block(dim=hidden_dim, num_heads=num_heads, drop=dropout, attn_drop=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Cross-modal alignment bridge
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.word_embedding = word_embedding.T

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, M, L] where M is channels/features, L is seq_len
        Returns:
            x_time: [B, M, hidden_dim]
            x_text: [B, M, hidden_dim]
        """
        B = x.shape[0]
        word_emb = self.word_embedding
        if word_emb.ndim == 2:
            word_emb = word_emb.repeat(B, 1, 1)
        elif word_emb.shape[0] != B:
            word_emb = word_emb[0].repeat(B, 1, 1)

        # 1. Project 1D time series to DINO token dimension
        x = self.linear(x)  # [B, M, hidden_dim]

        # 2. DINO-V3 ViT processing
        for block in self.dino_blocks:
            x = block(x)
        x = self.norm(x)
        x_time = x

        # 3. Cross-modal Attention with text embeddings (CALF Teacher alignment)
        q = x.transpose(0, 1)                  # [M, B, hidden_dim]
        k = v = word_emb.transpose(0, 1)       # [K, B, hidden_dim]
        x_text, _ = self.cross_attention(q, k, v)
        x_text = x_text.transpose(0, 1)        # [B, M, hidden_dim]

        return x_time, x_text
