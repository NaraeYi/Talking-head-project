from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import functional as F

from .rotary_embedding_torch import RotaryEmbedding
from .utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
import math


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x


class PoseResidualBranch(nn.Module):
    """
    pose branch: lightweight adapter that predicts only pitch/yaw/roll residuals
    from the shared transformer feature. It does not run a separate diffusion
    process and does not touch expression/lip dimensions.
    """

    def __init__(
        self,
        latent_dim: int,
        pose_dim: int = 198,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim > 0 else max(64, latent_dim // 4)
        self.residual_scale = float(residual_scale)

        # pose branch: small MLP adapter, sharing the base transformer output.
        self.adapter = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.to_residual = nn.Linear(hidden_dim, pose_dim)
        self.to_gate = nn.Linear(hidden_dim, pose_dim)

        # pose branch: start as an exact no-op for compatibility with pretrained checkpoints.
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.constant_(self.to_gate.bias, gate_bias)

    def forward(self, shared_feature: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.adapter(shared_feature)
        residual = self.to_residual(hidden) * self.residual_scale
        gate = torch.sigmoid(self.to_gate(hidden))
        return residual * gate, gate


class MotionDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 100,  # 4 seconds, 25 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats * 2, latent_dim)
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

        # pose branch: optional pitch/yaw/roll residual adapter after the base decoder.
        self.use_pose_branch = bool(kwargs.get("use_pose_branch", False))
        self.pose_branch_slices = ((1, 67), (67, 133), (133, 199))
        self.pose_branch = (
            PoseResidualBranch(
                latent_dim=latent_dim,
                pose_dim=sum(e - s for s, e in self.pose_branch_slices),
                hidden_dim=int(kwargs.get("pose_branch_hidden_dim", 128)),
                dropout=float(kwargs.get("pose_branch_dropout", 0.0)),
                residual_scale=float(kwargs.get("pose_branch_residual_scale", 1.0)),
                gate_bias=float(kwargs.get("pose_branch_gate_bias", -2.0)),
            )
            if self.use_pose_branch
            else None
        )
        
        self.epsilon = 0.00001

    def _apply_pose_branch(self, output: Tensor, shared_feature: Tensor) -> Tensor:
        if self.pose_branch is None:
            return output

        # pose branch: add residual only to pitch/yaw/roll; leave scale/t/exp unchanged.
        pose_residual, _ = self.pose_branch(shared_feature)
        output = output.clone()
        offset = 0
        for start, end in self.pose_branch_slices:
            width = end - start
            output[..., start:end] = output[..., start:end] + pose_residual[..., offset:offset + width]
            offset += width
        return output

    def guided_forward(self, x, cond_frame, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_frame, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_frame, cond_embed, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def forward(
        self, x: Tensor, cond_frame: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device

        # concat last frame, project to latent space
        x = torch.cat([x, cond_frame.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)
        x = self.input_projection(x)
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        # create audio conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        cond_tokens = self.cond_projection(cond_embed)
        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # create the diffusion timestep embedding, add the extra audio projection
        t_hidden = self.time_mlp(times)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        # Pass through the transformer decoder
        # attending to the conditional embedding
        output = self.seqTransDecoder(x, cond_tokens, t)
        shared_feature = output

        output = self.final_layer(shared_feature)
        output = self._apply_pose_branch(output, shared_feature)

        return output
    
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
        self.time_mlp_t = TimestepEmbedder(latent_dim * 4)
        self.time_mlp_r = TimestepEmbedder(latent_dim * 4)

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

        # pose branch: optional pitch/yaw/roll residual adapter after the base decoder.
        self.use_pose_branch = bool(kwargs.get("use_pose_branch", False))
        self.pose_branch_slices = ((1, 67), (67, 133), (133, 199))
        self.pose_branch = (
            PoseResidualBranch(
                latent_dim=latent_dim,
                pose_dim=sum(e - s for s, e in self.pose_branch_slices),
                hidden_dim=int(kwargs.get("pose_branch_hidden_dim", 128)),
                dropout=float(kwargs.get("pose_branch_dropout", 0.0)),
                residual_scale=float(kwargs.get("pose_branch_residual_scale", 1.0)),
                gate_bias=float(kwargs.get("pose_branch_gate_bias", -2.0)),
            )
            if self.use_pose_branch
            else None
        )

    def _apply_pose_branch(self, output: Tensor, shared_feature: Tensor) -> Tensor:
        if self.pose_branch is None:
            return output

        # pose branch: add residual only to pitch/yaw/roll; leave scale/t/exp unchanged.
        pose_residual, _ = self.pose_branch(shared_feature)
        output = output.clone()
        offset = 0
        for start, end in self.pose_branch_slices:
            width = end - start
            output[..., start:end] = output[..., start:end] + pose_residual[..., offset:offset + width]
            offset += width
        return output

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
        t_hidden = self.time_mlp_t(t_in)
        r_hidden = self.time_mlp_r(t_in - r_in)
        
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
        shared_feature = output

        output = self.final_layer(shared_feature)
        output = self._apply_pose_branch(output, shared_feature)
        return output
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
