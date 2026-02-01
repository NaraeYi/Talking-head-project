import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp

'''
Flow Matching 기반 MeanFlow 모델/손실/샘플러 한 파일에 구현.
MeanFlowLMDM: 학습용 flow_matching_loss(jvp 포함), 추론용 meanflow_sample(1-step/다단계), use_accelerator 지원.
내부에 MeanFlow 전용 모션 디코더(MotionDecoderMean), cosine/linear interpolant, time_sampler(logit_normal/uniform) 포함.
'''

# ---- rotary embedding (from original Ditto meanflow) ----
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, sin, cos):   # 트랜스포머(Transformer)의 Rotary Position Embedding (RoPE) 로직
    cos = cos.squeeze(0).squeeze(1)
    sin = sin.squeeze(0).squeeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def fixed_pos_embedding(x, seq_dim=2):
    dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(0, x.shape[seq_dim]), inv_freq)
    return torch.sin(sinusoid_inp).to(x.device), torch.cos(sinusoid_inp).to(x.device)


# ---- utils ----
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule="cosine", n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    if schedule == "linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep)
    elif schedule == "cosine":
        s = 8e-3
        steps = n_timestep + 1
        x = torch.linspace(0, n_timestep, steps)
        alphas_cumprod = torch.cos(((x / n_timestep) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError
    return betas


# ---- Motion decoder (meanflow variant) ----
class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sin, cos):
        # Handle different tensor shapes (3D or 4D)
        original_shape = x.shape
        if x.dim() == 3:
            B, N, C = x.shape
        elif x.dim() == 4:
            # If 4D tensor, treat as (B, N, H, W) and flatten H*W into C
            B, N, H, W = x.shape
            C = H * W
            x = x.view(B, N, C)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary_emb(q, k, sin, cos)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.o_proj(x)

        # Restore original shape if it was 4D
        if len(original_shape) == 4:
            x = x.view(original_shape)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, sin, cos):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, context.shape[1], 2, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q, k = apply_rotary_emb(q, k, sin, cos)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.o_proj(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.1):
        super().__init__()
        inner = dim * ff_mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return self.net(x)


class MotionDecoderMean(nn.Module):
    def __init__(
        self,
        nfeats=265,
        seq_len=80,
        latent_dim=512, # 전체 모델 차원
        ff_size=1024,
        num_layers=8,
        num_heads=8,    # attention head 수, head_dim = latent_dim // n_heads = 512 ÷ 8 = 64
        dropout=0.1,
        cond_feature_dim=1103,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(nfeats, latent_dim)
        self.cond_proj = nn.Linear(cond_feature_dim, latent_dim)
        self.cond_frame_proj = nn.Linear(nfeats, latent_dim)  ##### For motion condition frame

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(latent_dim),
                        SelfAttention(latent_dim, n_heads=num_heads, dropout=dropout),
                        nn.LayerNorm(latent_dim),
                        CrossAttention(latent_dim, latent_dim, n_heads=num_heads, dropout=dropout),
                        FeedForward(latent_dim, ff_mult=ff_size // latent_dim, dropout=dropout),
                    ]
                )
            )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, nfeats),
        )

    def forward(self, x, cond_frame, cond, r, t, cond_drop_prob=0.0):
        # cond_drop for classifier-free guidance (keep interface)
        if self.training and cond_drop_prob > 0:
            drop_mask = torch.rand((cond.shape[0], 1, 1), device=cond.device) < cond_drop_prob
            cond = torch.where(drop_mask, torch.zeros_like(cond), cond)

        sin, cos = fixed_pos_embedding(x)
        cond_embed = self.cond_proj(cond)  # (B, L, D) - Audio condition
        # x = self.in_proj(x) + self.cond_proj(cond_frame).unsqueeze(1)
        x = self.in_proj(x) + self.cond_frame_proj(cond_frame).unsqueeze(1)  # Motion condition frame

        for ln1, attn_self, ln2, attn_cross, ff in self.layers:
            x = x + attn_self(ln1(x), sin, cos)
            x = x + attn_cross(ln2(x), cond_embed, sin, cos)
            x = x + ff(x)

        out = self.out_proj(x)
        return out  # velocity prediction

    def guided_forward(self, x, cond_frame, cond, t, cfg_weight=2.0):
        # keep compatibility with diffusion interface: predict x0 (noise-free)
        v = self.forward(x, cond_frame, cond, t, t, 0.0)
        # For meanflow, predicted "x_start" analogous to latent; reuse velocity
        return x - v


# ---- MeanFlow LMDM (training + inference hooks) ----
class MeanFlowLMDM(nn.Module):
    def __init__(
        self,
        motion_feat_dim=265,
        audio_feat_dim=1103,
        seq_frames=80,
        device="cuda",
        time_sampler="logit_normal",
    ):
        super().__init__()
        self.motion_feat_dim = motion_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.seq_frames = seq_frames
        self.device = device
        self.time_sampler = time_sampler

        self.n_timestep = 1000
        self.guidance_weight = 2.0
        self.clip_denoised = False

        self.model = MotionDecoderMean(
            nfeats=motion_feat_dim,
            seq_len=seq_frames,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=audio_feat_dim,
        )

        self.init_diff()
        self.to(device)

    def init_diff(self):
        betas = make_beta_schedule(schedule="cosine", n_timestep=self.n_timestep)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        self.register_buffer("sqrt_recip1m_alphas_cumprod", torch.sqrt(1.0 / (1.0 - alphas_cumprod)))

    def maybe_clip(self, x):
        if self.clip_denoised:
            return torch.clamp(x, -1.0, 1.0)
        return x

    def predict_noise_from_start(self, x_t, t, x0):
        a = extract(self.sqrt_recip1m_alphas_cumprod, t, x_t.shape)
        b = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (a * x_t - x0 / b)

    def interpolant(self, t, path_type="linear"):
        if path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = torch.ones_like(t) * (-1)
            d_sigma_t = torch.ones_like(t) * 1
        elif path_type == "cosine":
            alpha_t = torch.cos(t * math.pi / 2)
            sigma_t = torch.sin(t * math.pi / 2)
            d_alpha_t = -math.pi / 2 * torch.sin(t * math.pi / 2)
            d_sigma_t = math.pi / 2 * torch.cos(t * math.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_time_steps(self, batch_size, device, time_mu=-0.4, time_sigma=1.0, ratio_r_not_equal_t=0.75):
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * time_sigma + time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")

        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        fraction_equal = 1.0 - ratio_r_not_equal_t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        r = torch.where(equal_mask, t, r)
        return r, t

    def model_predictions(self, x, cond_frame, cond, t):
        x_start = self.model.guided_forward(x, cond_frame, cond, t, cfg_weight=self.guidance_weight)
        x_start = self.maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def flow_matching_loss(self, x0, cond_frame, cond):
        B = x0.size(0)
        device = x0.device

        # # Debug: print shapes
        # print(f"flow_matching_loss - x0.shape: {x0.shape}, cond_frame.shape: {cond_frame.shape}, cond.shape: {cond.shape}")

        # ##### Multi-GPU batch splitting using accelerator
        # if hasattr(self, 'accelerator') and self.accelerator is not None:
        #     num_processes = self.accelerator.num_processes
        #     process_index = self.accelerator.process_index

        #     # Split batch across processes
        #     chunk_size = B // num_processes
        #     start_idx = process_index * chunk_size
        #     end_idx = start_idx + chunk_size if process_index < num_processes - 1 else B

        #     x0 = x0[start_idx:end_idx]
        #     cond_frame = cond_frame[start_idx:end_idx] if cond_frame is not None else None
        #     cond = cond[start_idx:end_idx] if cond is not None else None
        #     B = x0.size(0)
        # ##### Multi-GPU batch splitting using accelerator

        r, t = self.sample_time_steps(B, device)
        eps = torch.randn_like(x0,device=device)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        x_t = alpha_t * x0 + sigma_t * eps
        v_t = d_alpha_t * x0 + d_sigma_t * eps
        time_diff = (t - r).view(-1, 1, 1, 1)

        #### Ensure x_t is 3D for the model
        if x_t.dim() > 3:
            # If somehow x_t became 4D+, reshape to 3D
            x_t = x_t.view(x_t.shape[0], -1, x_t.shape[-1])

        v_pred = self.model(x_t, cond_frame, cond, r, t, 0.0)

        primals = (x_t, r, t)
        # tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        tangents = (v_t, torch.zeros_like(r, device=device), torch.ones_like(t, device=device))

        def fn_current(z, cur_r, cur_t):
            return self.model(z, cond_frame, cond, cur_r, cur_t)

        _, dudt = jvp(fn_current, primals, tangents)
        v_gt = v_t - time_diff * dudt

        # loss = torch.mean((v_pred - v_gt) ** 2)
        loss = torch.mean((v_pred - v_gt) ** 2, dim=-1).to(device)

        ##### Reduce loss across all processes for multi-GPU training
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            loss = self.accelerator.reduce(loss, reduction="mean")

        loss_dict = {"flow_loss": loss.detach()}
        return loss, loss_dict

    # training-compatible interface
    def diffusion(self, x, cond_frame, cond, t_override=None):
        loss, loss_dict = self.flow_matching_loss(x, cond_frame, cond)
        return loss, loss_dict

    def use_accelerator(self, accelerator):
        self.accelerator = accelerator  ##### Multi-GPU batch splitting using accelerator
        self.model = accelerator.prepare(self.model)
        self.device = accelerator.device
        self.to(self.device)

    @torch.no_grad()
    def meanflow_sample(self, kp_cond, aud_cond, num_steps=1):
        self.model.eval()
        cond_frame = kp_cond
        cond = aud_cond

        if cond_frame.dim() == 1:
            cond_frame = cond_frame.unsqueeze(0)
        device = cond_frame.device

        z_t = torch.randn((cond_frame.shape[0], self.seq_frames, self.motion_feat_dim), device=device)
        t = torch.ones((cond_frame.shape[0],), device=device)
        for _ in range(num_steps):
            v = self.model(z_t, cond_frame, cond, torch.zeros_like(t), t, 0.0)
            z_t = z_t - v
        return z_t


    # # mean flow sampler(inference)
    # @torch.no_grad()
    # def meanflow_sample(self, kp_cond, aud_cond, num_steps=1, cfg_scale=1.0):
    #     """
    #     MeanFlow sampler supporting both single-step and multi-step generation
        
    #     Based on Eq.(12): z_r = z_t - (t-r)u(z_t, r, t)
    #     For single-step: z_0 = z_1 - u(z_1, 0, 1)
    #     For multi-step: iteratively apply the Eq.(12) with intermediate steps
        
    #     Args:
    #         kp_cond: condition frame [B, motion_feat_dim] or [1, motion_feat_dim]
    #         aud_cond: audio condition [B, seq_len, audio_feat_dim] or [1, seq_len, audio_feat_dim]
    #         num_steps: number of integration steps
    #         cfg_scale: classifier-free guidance scale (not used currently, kept for compatibility)
    #     """
    #     self.model.eval()
        
    #     cond_frame = kp_cond
    #     cond = aud_cond
        
    #     # Get batch size from cond_frame
    #     if cond_frame.dim() == 1:
    #         batch_size = 1
    #         cond_frame = cond_frame.unsqueeze(0)
    #     else:
    #         batch_size = cond_frame.shape[0]
        
    #     device = cond_frame.device
        
    #     # Initial latent: z_1 ~ N(0, I)
    #     shape = (batch_size, self.seq_frames, self.motion_feat_dim)
    #     z = torch.randn(shape, device=device)
        
    #     if num_steps == 1:
    #         # Single-step: z_0 = z_1 - u(z_1, 0, 1)
    #         r = torch.zeros(batch_size, device=device)  # target time (0)
    #         t = torch.ones(batch_size, device=device)   # current time (1)
            
    #         # Predict velocity: u(z_t, r, t)
    #         u = self.model.forward(
    #             z,              # current latent z_t
    #             cond_frame,     # appearance condition
    #             cond,           # audio condition
    #             r,              # times_r: target time
    #             t,              # times_t: current time
    #             0.0             # cond_drop_prob = 0 (no CFG)
    #         )
            
    #         # Apply MeanFlow equation: z_r = z_t - (t-r)*u(z_t, r, t)
    #         # For single-step: z_0 = z_1 - (1-0)*u = z_1 - u
    #         x0 = z - u
            
    #     else:
    #         # Multi-step: iteratively apply MeanFlow equation
    #         # Time steps from 1.0 to 0.0
    #         time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
            
    #         for i in range(num_steps):
    #             t_cur = time_steps[i]      # current time t
    #             t_next = time_steps[i + 1] # target time r
                
    #             # Create time tensors for batch
    #             t = torch.full((batch_size,), t_cur, device=device)
    #             r = torch.full((batch_size,), t_next, device=device)
                
    #             # Predict velocity: u(z_t, r, t)
    #             u = self.model.model.forward(
    #                 z,              # current latent z_t
    #                 cond_frame,     # appearance condition
    #                 cond,           # audio condition
    #                 r,              # times_r: target time (next step)
    #                 t,              # times_t: current time
    #                 0.0             # cond_drop_prob = 0 (no CFG)
    #             )
                
    #             # Apply MeanFlow equation: z_r = z_t - (t-r)*u(z_t, r, t)
    #             z = z - (t_cur - t_next) * u
            
    #         x0 = z
        
    #     return x0  # pred_kp_seq: 예측된 motion latent
