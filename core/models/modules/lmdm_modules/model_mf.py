"""
MeanFlow 전용 MotionDecoderMF
"""
from typing import Callable
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import functional as F

from .rotary_embedding_torch import RotaryEmbedding
from .utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
from .model import TransformerEncoderLayer, FiLMTransformerDecoderLayer, DecoderLayerStack


class MotionDecoderMF(nn.Module):
    """
    MeanFlow 전용 decoder:
      - forward(x, cond_frame, cond_embed, r, t, cond_drop_prob)
      - time embedding을 (r,t) 둘 다로 conditioning
    """

    def __init__(
        self,
        nfeats: int,
        seq_len: int = 100,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        time_fuse: str = "sum",  # "sum" or "concat"
        **kwargs
    ) -> None:
        super().__init__()

        self.time_fuse = time_fuse
        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(latent_dim, dropout, batch_first=True)

        # time embedding processing (shared)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        if time_fuse == "sum":
            self.time_fuse_mlp = None
            fused_dim = latent_dim * 4
        elif time_fuse == "concat":
            self.time_fuse_mlp = nn.Sequential(
                nn.Linear(latent_dim * 8, latent_dim * 4),
                nn.Mish(),
            )
            fused_dim = latent_dim * 4
        else:
            raise ValueError(f"Unknown time_fuse: {time_fuse}")

        self.to_time_cond = nn.Sequential(nn.Linear(fused_dim, latent_dim))
        self.to_time_tokens = nn.Sequential(
            nn.Linear(fused_dim, latent_dim * 2),
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats * 2, latent_dim)

        # cond encoder
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(
        self,
        x: Tensor,
        cond_frame: Tensor,
        cond_embed: Tensor,
        r: Tensor,
        t: Tensor,
        cond_drop_prob: float = 0.0,
    ):
        """
        r,t: shape broadcastable to [B,1,1] or [B]
        """
        batch_size, device = x.shape[0], x.device
        dtype = x.dtype

        # normalize shapes for time embedding
        if r.dim() == 3:
            r_in = r.view(batch_size)
        else:
            r_in = r
        if t.dim() == 3:
            t_in = t.view(batch_size)
        else:
            t_in = t

        r_in = r_in.to(device=device, dtype=dtype)
        t_in = t_in.to(device=device, dtype=dtype)

        # concat last frame, project to latent space
        x = torch.cat([x, cond_frame.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)

        # conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        # cond tokens
        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # MeanFlow time embedding
        t_hidden = self.time_mlp(t_in)
        r_hidden = self.time_mlp(r_in)

        if self.time_fuse == "sum":
            fused = t_hidden + r_hidden
        else:
            fused = self.time_fuse_mlp(torch.cat([t_hidden, r_hidden], dim=-1))

        # project to FiLM and time tokens
        t_film = self.to_time_cond(fused)
        t_tokens = self.to_time_tokens(fused)

        # FiLM conditioning with audio
        null_cond_hidden = self.null_cond_hidden.to(t_film.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t_film = t_film + cond_hidden

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        # decode
        output = self.seqTransDecoder(x, cond_tokens, t_film)
        output = self.final_layer(output)
        return output



