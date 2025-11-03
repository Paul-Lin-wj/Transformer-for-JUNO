import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================Transformer============================


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.01):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by the number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        bsz, seq_len, _ = Q.size()
        q = self.wq(Q).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(K).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(V).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))  # Apply mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.embed_dim)
        )
        return self.fc(output)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.01):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.silu = nn.SiLU()
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w1(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        h = self.norm1(x + self.attn(x, x, x, mask))
        return self.norm2(h + self.ff(h))


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim=3,
        embed_dim=64,
        num_heads=2,
        ff_dim=128,
        num_layers=1,
        dropout=0.0,
        output_dim=6,
        max_seq_len=1000,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # feature embedding
        self.feature_projection = nn.Linear(input_dim, embed_dim)
        # self.position_embedding = nn.Embedding(
        #     num_embeddings=max_seq_len, embedding_dim=embed_dim
        # )
        self.norm_after_position = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def make_padding_mask(self, x):
        return x.abs().sum(dim=-1) == 0

    def create_positional_encoding(self, seq_len, device):
        """Generate sinusoidal position embeddings for the given sequence length."""
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(
            1
        )  # Shape: (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float()
            * -(math.log(10000.0) / self.embed_dim)
        ).to(
            device
        )  # Shape: (embed_dim // 2,)

        positional_encoding = torch.zeros(
            seq_len, self.embed_dim, device=device
        )  # Shape: (seq_len, embed_dim)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)  # Even index: sin
        positional_encoding[:, 1::2] = torch.cos(position * div_term)  # Odd index: cos

        return positional_encoding.unsqueeze(
            0
        )  # Add batch dimension: Shape: (1, seq_len, embed_dim)

    def forward(self, x):

        # mask
        mask = self.make_padding_mask(x)

        # token embedding
        x = self.feature_projection(x)

        # position embedding (time information)
        seq_len = x.size(1)
        device = x.device
        position_embeddings = self.create_positional_encoding(
            seq_len, device
        )  # Shape: (1, seq_len, embed_dim)
        x += position_embeddings

        # seq_len = x.size(1)
        # position_ids = (
        #     torch.arange(seq_len, device=x.device)
        #     .unsqueeze(0)
        #     .expand(x.size(0), seq_len)
        # )
        # x = x + self.position_embedding(position_ids)
        x = self.norm_after_position(x)

        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        # print("device:", x.device)
        not_pad = (~mask).float()
        x_valid = x * not_pad.unsqueeze(-1)
        x_pooled = x_valid.sum(dim=1) / (not_pad.sum(dim=1, keepdim=True) + 1e-8)
        output = self.fc_out(x_pooled)
        return output


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return (x / (norm + self.eps)) * self.weight


# =========================new transformer=========================
# class MultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.01):
#         super().__init__()
#         assert embed_dim % num_heads == 0
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)  # fused qkv
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.dropout_p = dropout

#     def forward(self, x, key_padding_mask=None):
#         # x: (B, S, C)
#         B, S, C = x.shape
#         qkv = (
#             self.qkv(x)
#             .view(B, S, 3, self.num_heads, self.head_dim)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, S, D)

#         # 将 padding mask 转成可广播到 (B, H, S_q, S_k) 的形状
#         attn_mask = None
#         if key_padding_mask is not None:
#             # key_padding_mask: (B, S) -> True 表示需要mask
#             attn_mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, S)

#         out = F.scaled_dot_product_attention(
#             q,
#             k,
#             v,
#             attn_mask=attn_mask,  # 支持 bool 或 float(-inf)
#             dropout_p=self.dropout_p if self.training else 0.0,
#             is_causal=False,
#         )  # -> (B, H, S, D)

#         out = out.transpose(1, 2).contiguous().view(B, S, C)
#         return self.out_proj(out)


# class FeedForward(nn.Module):
#     def __init__(self, embed_dim, ff_dim, dropout=0.01):
#         super().__init__()
#         # SwiGLU: proj -> gate
#         self.w1 = nn.Linear(embed_dim, ff_dim * 2, bias=False)
#         self.w2 = nn.Linear(ff_dim, embed_dim, bias=False)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x1, x2 = self.w1(x).chunk(2, dim=-1)
#         x = F.silu(x2) * x1  # SwiGLU
#         x = self.dropout(x)
#         return self.w2(x)


# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, norm=RMSNorm):
#         super().__init__()
#         self.norm1 = norm(embed_dim)
#         self.attn = MultiheadAttention(embed_dim, num_heads, dropout)
#         self.norm2 = norm(embed_dim)
#         self.ff = FeedForward(embed_dim, ff_dim, dropout)

#     def forward(self, x, mask=None):
#         x = x + self.attn(self.norm1(x), key_padding_mask=mask)  # Pre-LN
#         x = x + self.ff(self.norm2(x))  # Pre-LN
#         return x


# class TransformerModel(nn.Module):
#     def __init__(
#         self,
#         input_dim=3,
#         embed_dim=128,
#         num_heads=2,
#         ff_dim=256,
#         num_layers=4,
#         dropout=0.1,
#         output_dim=6,
#         use_rope=False,
#         patch_size=0,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.use_rope = use_rope
#         self.patch_size = patch_size

#         self.feature_projection = nn.Linear(input_dim, embed_dim, bias=False)
#         self.norm_after_position = RMSNorm(embed_dim)

#         self.blocks = nn.ModuleList(
#             [
#                 TransformerBlock(embed_dim, num_heads, ff_dim, dropout, norm=RMSNorm)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.final_norm = RMSNorm(embed_dim)
#         self.fc_out = nn.Linear(embed_dim, output_dim)

#         # 可选：预注册最大长度的正弦位置编码（或 RoPE 参数）
#         self.register_buffer(
#             "pos_cache", None, persistent=False
#         )  # 需要时按 max_len 生成并缓存

#     def make_padding_mask(self, x, lengths=None):
#         if lengths is not None:
#             B, S, _ = x.shape
#             ar = torch.arange(S, device=x.device)[None, :]
#             return ar >= lengths[:, None]  # (B, S)
#         # 不推荐用“全零判断”，但保留以兼容
#         return x.abs().sum(dim=-1) == 0

#     def positional_encoding(self, seq_len, device):
#         if self.pos_cache is None or self.pos_cache.size(1) < seq_len:
#             pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
#             div = torch.exp(
#                 torch.arange(0, self.embed_dim, 2, device=device).float()
#                 * -(math.log(10000.0) / self.embed_dim)
#             )
#             pe = torch.zeros(seq_len, self.embed_dim, device=device)
#             pe[:, 0::2] = torch.sin(pos * div)
#             pe[:, 1::2] = torch.cos(pos * div)
#             self.pos_cache = pe.unsqueeze(0)  # (1, S, C)
#         return self.pos_cache[:, :seq_len, :]

#     def maybe_downsample(self, x, mask):
#         if self.patch_size and self.patch_size > 1:
#             B, S, C = x.shape
#             pad = (-S) % self.patch_size
#             if pad:
#                 x = F.pad(x, (0, 0, 0, pad))
#                 mask = F.pad(mask, (0, pad), value=True)
#             x = x.view(B, (S + pad) // self.patch_size, self.patch_size, C).mean(dim=2)
#             # 聚合 mask：只要patch里有非pad，就标记为非pad
#             mask = mask.view(B, (S + pad) // self.patch_size, self.patch_size).all(
#                 dim=2
#             )
#         return x, mask

#     def forward(self, x, lengths=None):
#         mask = self.make_padding_mask(x, lengths)  # (B, S) True=pad

#         # token embedding + pos
#         x = self.feature_projection(x)
#         pos = self.positional_encoding(x.size(1), x.device)
#         x = self.norm_after_position(x + pos)

#         # 可选下采样
#         x, mask = self.maybe_downsample(x, mask)

#         # Transformer blocks (SDPA inside)
#         for blk in self.blocks:
#             x = blk(x, mask=mask)

#         x = self.final_norm(x)

#         # masked mean-pool
#         not_pad = (~mask).float()
#         x_valid = x * not_pad.unsqueeze(-1)
#         x_pooled = x_valid.sum(dim=1) / (not_pad.sum(dim=1, keepdim=True) + 1e-8)
#         return self.fc_out(x_pooled)
