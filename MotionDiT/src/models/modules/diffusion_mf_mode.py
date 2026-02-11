import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp



class MotionMeanFlow(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        loss_type="l2",
        cond_drop_prob=0.2,
        part_w_dict=None,
        use_last_frame_loss=False,
        use_reg_loss=False,
        dim_ws=None,
        lambda_pva=0.0,
        # MeanFlow options
        path_type="linear",            # "linear" or "cosine"
        time_sampler="uniform",        # "uniform" or "logit_normal"
        time_mu=-0.4,
        time_sigma=1.0,
        ratio_r_not_equal_t=0.75,      # fraction where r != t
        weighting="uniform",           # "uniform" or "adaptive"
        adaptive_p=1.0,
        # CFG-like target options (training)
        use_cfg_target=False,
        cfg_omega=1.0,
        cfg_kappa=0.0,
        cfg_min_t=0.0,
        cfg_max_t=0.8,
        # Other
        detach_cond=True,
        use_r0_recon=False,            # False: x_hat = z - (t-r)*u, True: x_hat = z - t*u
        meanflow_mode="improved",  # "meanflow" or "improved" (default: "improved")
    ):
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.repr_dim = repr_dim

        self.cond_drop_prob = float(cond_drop_prob)
        self.lambda_pva = float(lambda_pva)
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

        # MeanFlow configs
        self.path_type = path_type
        self.time_sampler = time_sampler
        self.time_mu = float(time_mu)
        self.time_sigma = float(time_sigma)
        self.ratio_r_not_equal_t = float(ratio_r_not_equal_t)
        self.weighting = weighting
        self.adaptive_p = float(adaptive_p)
        self.meanflow_mode = meanflow_mode.lower()  # "meanflow" or "improved"
        if self.meanflow_mode not in ["meanflow", "improved"]:
            raise ValueError(f"meanflow_mode must be 'meanflow' or 'improved', got {meanflow_mode}")

        # CFG-like configs (training target)
        self.use_cfg_target = bool(use_cfg_target)
        self.cfg_omega = float(cfg_omega)
        self.cfg_kappa = float(cfg_kappa)
        self.cfg_min_t = float(cfg_min_t)
        self.cfg_max_t = float(cfg_max_t)

        self.detach_cond = bool(detach_cond)
        self.use_r0_recon = bool(use_r0_recon)

        # Ditto-specific
        self.part_w_dict = part_w_dict or {
            "scale": [0, 1, 1],
            "pitch": [1, 67, 1],
            "yaw": [67, 133, 1],
            "roll": [133, 199, 1],
            "t": [199, 202, 1],
            "exp": [202, 265, 1],
        }
        self.use_last_frame_loss = use_last_frame_loss
        self.use_reg_loss = use_reg_loss

        if dim_ws is not None:
            self.register_buffer("dim_ws", torch.from_numpy(dim_ws))
        else:
            self.dim_ws = None
    
    def interpolant(self, t):
        """
        Returns alpha_t, sigma_t, d_alpha_t, d_sigma_t.
        t shape: [B,1,1] (or broadcastable)
        """
        if self.path_type == "linear":
            alpha_t = 1.0 - t
            sigma_t = t
            d_alpha_t = -torch.ones_like(t)
            d_sigma_t = torch.ones_like(t)
        elif self.path_type == "cosine":
            # alpha = cos(pi/2 t), sigma = sin(pi/2 t)
            half_pi = 0.5 * torch.tensor(np.pi, device=t.device, dtype=t.dtype)
            alpha_t = torch.cos(half_pi * t)
            sigma_t = torch.sin(half_pi * t)
            d_alpha_t = -half_pi * torch.sin(half_pi * t)
            d_sigma_t = half_pi * torch.cos(half_pi * t)
        else:
            raise ValueError(f"Unknown path_type: {self.path_type}")

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_time_steps(self, bsz, device, dtype):
        """
        Samples r,t with t>=r. Optionally forces a fraction to have r=t.
        Returns r,t shaped [B,1,1].
        """
        if self.time_sampler == "uniform":
            ts = torch.rand(bsz, 2, device=device, dtype=dtype)
        elif self.time_sampler == "logit_normal":
            ns = torch.randn(bsz, 2, device=device, dtype=dtype)
            ns = ns * self.time_sigma + self.time_mu
            ts = torch.sigmoid(ns)
        else:
            raise ValueError(f"Unknown time_sampler: {self.time_sampler}")

        ts, _ = torch.sort(ts, dim=1)  # ensure r <= t
        r = ts[:, 0]
        t = ts[:, 1]

        # Force a fraction to have r=t
        # ratio_r_not_equal_t: fraction with r != t
        frac_equal = 1.0 - self.ratio_r_not_equal_t
        if frac_equal > 0:
            equal_mask = (torch.rand(bsz, device=device, dtype=dtype) < frac_equal)
            r = torch.where(equal_mask, t, r)

        # reshape to broadcast-friendly
        return r.view(bsz, 1, 1), t.view(bsz, 1, 1)
    
    def sample_t_r(self, bsz, device, dtype):
        t = torch.rand(bsz, 1, 1, device=device, dtype=dtype)
        r = torch.rand(bsz, 1, 1, device=device, dtype=dtype) * t
        return t, r
    
    # -------------------------- Ditto PVA loss -------------------------- #

    def _get_pva_loss(
        self,
        pred,
        gt,
        part_w_dict,
        cond_frame=None,
        use_last_frame_loss=False,
        use_reg_loss=False,
        dim_ws=None,
    ):
        def _part_loss(s, e, w, rw=0.0):
            if s <= 0:
                s = 0
            if e <= 0:
                e = gt.shape[-1]

            dim_w = w
            if dim_ws is not None:
                dim_w = dim_ws[s:e][None, None] * w  # [1,1,dim]

            p1 = pred[..., s:e]
            p2 = gt[..., s:e]

            v1 = p1[:, 1:] - p1[:, :-1]
            v2 = p2[:, 1:] - p2[:, :-1]

            a1 = v1[:, 1:] - v1[:, :-1]
            a2 = v2[:, 1:] - v2[:, :-1]

            _p = (self.loss_fn(p1, p2, reduction="none") * dim_w).mean()
            _v = (self.loss_fn(v1, v2, reduction="none") * dim_w).mean()
            _a = (self.loss_fn(a1, a2, reduction="none") * dim_w).mean()

            _l = torch.zeros((), device=pred.device, dtype=pred.dtype)
            if use_last_frame_loss:
                if cond_frame is None:
                    raise ValueError("cond_frame is required when use_last_frame_loss=True")
                p0 = p1[:, 0:1]
                gt0 = cond_frame[..., s:e][:, None]
                _l = (self.loss_fn(p0, gt0, reduction="none") * dim_w).mean()

            _r = torch.zeros((), device=pred.device, dtype=pred.dtype)
            if use_reg_loss:
                _r = torch.abs(p1).mean() * rw

            return _p, _v, _a, _l, _r

        loss_dict = {}
        for k, (s, e, w) in part_w_dict.items():
            rw = 0.0 if k == "scale" else 1e-4
            _p, _v, _a, _l, _r = _part_loss(s, e, w, rw)
            loss_dict[f"{k}_P"] = _p
            loss_dict[f"{k}_V"] = _v
            loss_dict[f"{k}_A"] = _a
            if use_last_frame_loss:
                loss_dict[f"{k}_L"] = _l
            if use_reg_loss:
                loss_dict[f"{k}_R"] = _r

        return loss_dict

    # -------------------------- MeanFlow loss -------------------------- #

    def p_losses(self, x, cond_frame, cond):
        """
        MeanFlow objective with optional CFG-like target and optional adaptive weighting.
        x: clean motion latent (x0), shape [B,L,D]
        """
        b, device, dtype = x.shape[0], x.device, x.dtype

        # bf16/fp16 mixed precision 사용 시 jvp 연산에서 타입 불일치 에러를 방지하기 위해
        # 현재 활성화된 autocast dtype으로 입력값들을 맞춥니다.
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()

        # move conds and ensure correct dtype
        cond_frame = cond_frame.to(device=device, dtype=dtype)
        cond = cond.to(device=device, dtype=dtype)
        x = x.to(device=device, dtype=dtype)

        # optionally detach cond inputs (if they are not meant to be trained end-to-end)
        cond_frame_ = cond_frame.detach() if self.detach_cond else cond_frame
        cond_ = cond.detach() if self.detach_cond else cond

        # sample r,t
        r, t = self.sample_time_steps(b, device, dtype)
        time_diff = (t - r)

        # noise endpoint
        e = torch.randn_like(x)

        # interpolant
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)

        # z_t and instantaneous velocity v_t
        z = alpha_t * x + sigma_t * e
        v = d_alpha_t * x + d_sigma_t * e

        # base u (with gradient)
        def fn(z_, r_, t_, drop_prob):
            return self.model(z_, cond_frame_, cond_, r_, t_, cond_drop_prob=drop_prob)

        u = fn(z, r, t, self.cond_drop_prob)

        # IMF에서만 필요: v_theta(z_t,t) := u_theta(z_t, t, t)
        if self.meanflow_mode == "improved":
            v_theta = fn(z, t, t, self.cond_drop_prob)

        # ==================== MeanFlow (Original) ====================
        if self.meanflow_mode == "meanflow":
            if self.use_cfg_target:
                t_scalar = t.view(b)
                cfg_mask = (t_scalar >= self.cfg_min_t) & (t_scalar <= self.cfg_max_t)

                if cfg_mask.any():
                    idx_cfg = torch.where(cfg_mask)[0]
                    idx_nocfg = torch.where(~cfg_mask)[0]

                    u_target = torch.zeros_like(u)

                    # ---- CFG subset ----
                    z_cfg = z[idx_cfg]
                    v_cfg = v[idx_cfg]
                    r_cfg = r[idx_cfg]
                    t_cfg = t[idx_cfg]
                    dt_cfg = time_diff[idx_cfg]

                    with torch.no_grad():
                        u_cond = fn(z_cfg, r_cfg, t_cfg, drop_prob=0.0)
                        u_uncond = fn(z_cfg, r_cfg, t_cfg, drop_prob=1.0)
                        v_tilde = (
                            self.cfg_omega * v_cfg
                            + self.cfg_kappa * u_cond
                            + (1.0 - self.cfg_omega - self.cfg_kappa) * u_uncond
                        )

                    with torch.autocast(device_type='cuda', enabled=False), \
                         torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                        z_cfg_fp32 = z_cfg.float()
                        r_cfg_fp32 = r_cfg.float()
                        t_cfg_fp32 = t_cfg.float()
                        v_tilde_fp32 = v_tilde.float()

                        def fn_cfg_fp32(z_, r_, t_):
                            return fn(z_, r_, t_, drop_prob=self.cond_drop_prob).float()

                        _, dudt_cfg = jvp(
                            fn_cfg_fp32,
                            (z_cfg_fp32, r_cfg_fp32, t_cfg_fp32),
                            (v_tilde_fp32, torch.zeros_like(r_cfg_fp32), torch.ones_like(t_cfg_fp32)),
                        )
                    u_target[idx_cfg] = v_tilde - dt_cfg * dudt_cfg.to(dtype)

                    # ---- non-CFG subset ----
                    if idx_nocfg.numel() > 0:
                        z_nc = z[idx_nocfg]
                        v_nc = v[idx_nocfg]
                        r_nc = r[idx_nocfg]
                        t_nc = t[idx_nocfg]
                        dt_nc = time_diff[idx_nocfg]

                        with torch.autocast(device_type='cuda', enabled=False), \
                             torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                            z_nc_fp32 = z_nc.float()
                            r_nc_fp32 = r_nc.float()
                            t_nc_fp32 = t_nc.float()
                            v_nc_fp32 = v_nc.float()

                            def fn_nc_fp32(z_, r_, t_):
                                return fn(z_, r_, t_, drop_prob=self.cond_drop_prob).float()

                            _, dudt_nc = jvp(
                                fn_nc_fp32,
                                (z_nc_fp32, r_nc_fp32, t_nc_fp32),
                                (v_nc_fp32, torch.zeros_like(r_nc_fp32), torch.ones_like(t_nc_fp32)),
                            )
                        u_target[idx_nocfg] = v_nc - dt_nc * dudt_nc.to(dtype)
                else:
                    # no cfg sample in this batch
                    with torch.autocast(device_type='cuda', enabled=False), \
                         torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                        z_fp32 = z.float()
                        r_fp32 = r.float()
                        t_fp32 = t.float()
                        v_fp32 = v.float()

                        def fn_std_fp32(z_, r_, t_):
                            return fn(z_, r_, t_, drop_prob=self.cond_drop_prob).float()

                        _, dudt = jvp(
                            fn_std_fp32,
                            (z_fp32, r_fp32, t_fp32),
                            (v_fp32, torch.zeros_like(r_fp32), torch.ones_like(t_fp32)),
                        )
                    u_target = v - time_diff * dudt.to(dtype)
            else:
                # standard MeanFlow (no CFG)
                with torch.autocast(device_type='cuda', enabled=False), \
                     torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    z_fp32 = z.float()
                    r_fp32 = r.float()
                    t_fp32 = t.float()
                    v_fp32 = v.float()

                    def fn_std_fp32(z_, r_, t_):
                        return fn(z_, r_, t_, drop_prob=self.cond_drop_prob).float()

                    _, dudt = jvp(
                        fn_std_fp32,
                        (z_fp32, r_fp32, t_fp32),
                        (v_fp32, torch.zeros_like(r_fp32), torch.ones_like(t_fp32)),
                    )
                u_target = v - time_diff * dudt.to(dtype)

            # MeanFlow regression loss
            err = u - u_target.detach()

        # ==================== Improved MeanFlow ====================
        else:  # self.meanflow_mode == "improved"
            if self.use_cfg_target:
                t_scalar = t.view(b)
                cfg_mask = (t_scalar >= self.cfg_min_t) & (t_scalar <= self.cfg_max_t)

                if cfg_mask.any():
                    idx_cfg = torch.where(cfg_mask)[0]
                    idx_nocfg = torch.where(~cfg_mask)[0]

                    # ---- CFG subset ----
                    z_cfg = z[idx_cfg]
                    v_cfg = v[idx_cfg]
                    r_cfg = r[idx_cfg]
                    t_cfg = t[idx_cfg]
                    dt_cfg = time_diff[idx_cfg]

                    with torch.no_grad():
                        u_cond = fn(z_cfg, r_cfg, t_cfg, drop_prob=0.0)
                        u_uncond = fn(z_cfg, r_cfg, t_cfg, drop_prob=1.0)
                        v_tilde = (
                            self.cfg_omega * v_cfg
                            + self.cfg_kappa * u_cond
                            + (1.0 - self.cfg_omega - self.cfg_kappa) * u_uncond
                        )

                    v_target_cfg = v_tilde
                    v_theta_cfg = v_theta[idx_cfg]
                    u_cfg = u[idx_cfg]

                    with torch.autocast(device_type='cuda', enabled=False), \
                         torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                        z_cfg_fp32 = z_cfg.float()
                        r_cfg_fp32 = r_cfg.float()
                        t_cfg_fp32 = t_cfg.float()
                        vtheta_cfg_fp32 = v_theta_cfg.float()

                        def fn_cfg_fp32(z_, r_, t_):
                            return fn(z_, r_, t_, drop_prob=self.cond_drop_prob).float()

                        _, dudt_cfg = jvp(
                            fn_cfg_fp32,
                            (z_cfg_fp32, r_cfg_fp32, t_cfg_fp32),
                            (vtheta_cfg_fp32, torch.zeros_like(r_cfg_fp32), torch.ones_like(t_cfg_fp32)),
                        )

                    V_cfg = u_cfg + dt_cfg * dudt_cfg.to(dtype).detach()
                    err_cfg = V_cfg - v_target_cfg.detach()

                    # ---- non-CFG subset ----
                    if idx_nocfg.numel() > 0:
                        z_nc = z[idx_nocfg]
                        v_nc = v[idx_nocfg]
                        r_nc = r[idx_nocfg]
                        t_nc = t[idx_nocfg]
                        dt_nc = time_diff[idx_nocfg]

                        v_theta_nc = v_theta[idx_nocfg]
                        u_nc = u[idx_nocfg]

                        with torch.autocast(device_type='cuda', enabled=False), \
                             torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                            z_nc_fp32 = z_nc.float()
                            r_nc_fp32 = r_nc.float()
                            t_nc_fp32 = t_nc.float()
                            v_nc_fp32 = v_nc.float()

                            def fn_nc_fp32(z_, r_, t_):
                                return fn(z_, r_, t_, drop_prob=self.cond_drop_prob).float()

                            _, dudt_nc = jvp(
                                fn_nc_fp32,
                                (z_nc_fp32, r_nc_fp32, t_nc_fp32),
                                (v_nc_fp32, torch.zeros_like(r_nc_fp32), torch.ones_like(t_nc_fp32)),
                            )

                        v_target_nc = v_nc
                        V_nc = u_nc + dt_nc * dudt_nc.to(dtype).detach()
                        err = V_nc - v_target_nc.detach()
                    else:
                        err = err_cfg
                else:
                    # no cfg sample in this batch
                    with torch.autocast(device_type='cuda', enabled=False), \
                         torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                        z_fp32 = z.float()
                        r_fp32 = r.float()
                        t_fp32 = t.float()

                        v_theta_fp32 = fn(z_fp32, t_fp32, t_fp32, drop_prob=self.cond_drop_prob).float()

                        def fn_std_fp32(z_, r_, t_):
                            return fn(z_, r_, t_, drop_prob=self.cond_drop_prob).float()

                        u_fp32, dudt_fp32 = jvp(
                            fn_std_fp32,
                            (z_fp32, r_fp32, t_fp32),
                            (v_theta_fp32, torch.zeros_like(r_fp32), torch.ones_like(t_fp32)),
                        )
                    u = u_fp32.to(dtype)
                    dudt = dudt_fp32.to(dtype)
                    V_pred = u + time_diff * dudt.detach()
                    err = V_pred - v.detach()
            else:
                # standard Improved MeanFlow (no CFG)
                with torch.autocast(device_type='cuda', enabled=False), \
                     torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    z_fp32 = z.float()
                    r_fp32 = r.float()
                    t_fp32 = t.float()

                    v_theta_fp32 = fn(z_fp32, t_fp32, t_fp32, drop_prob=self.cond_drop_prob).float()

                    def fn_std_fp32(z_, r_, t_):
                        return fn(z_, r_, t_, drop_prob=self.cond_drop_prob).float()

                    u_fp32, dudt_fp32 = jvp(
                        fn_std_fp32,
                        (z_fp32, r_fp32, t_fp32),
                        (v_theta_fp32, torch.zeros_like(r_fp32), torch.ones_like(t_fp32)),
                    )

                    u = u_fp32.to(dtype)
                    dudt = dudt_fp32.to(dtype)
                    V_pred = u + time_diff * dudt.detach()

            # Improved MeanFlow regression loss
            if not (self.use_cfg_target and cfg_mask.any()):
                err = V_pred - v.detach()
        # per-sample squared error
        loss_mid = (err ** 2).reshape(b, -1).mean(dim=1)  # [B]

        if self.weighting == "adaptive":
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.adaptive_p)
            loss_mf = (weights * loss_mid).mean()
        else:
            loss_mf = loss_mid.mean()

        # reconstruct x0-like quantity for PVA
        if self.use_r0_recon:
            x_hat = z - t * u
        else:
            x_hat = z - time_diff * u  # z_r estimate; if r≈0 this approximates x0

        loss_dict_pva = self._get_pva_loss(
            pred=x_hat,
            gt=x,
            part_w_dict=self.part_w_dict,
            cond_frame=cond_frame,
            use_last_frame_loss=self.use_last_frame_loss,
            use_reg_loss=self.use_reg_loss,
            dim_ws=self.dim_ws,
        )
        loss_pva = sum(loss_dict_pva.values())

        total_loss = loss_mf + self.lambda_pva * loss_pva

        # logging dict
        loss_dict = {"mf": loss_mf, **loss_dict_pva}
        return total_loss, loss_dict

    def loss(self, x, cond_frame, cond):
        return self.p_losses(x, cond_frame, cond)

    def forward(self, x, cond_frame, cond):
        return self.loss(x, cond_frame, cond)

    # -------------------------- Sampling -------------------------- #

    @torch.no_grad()
    def sample_one_step(self, shape, cond_frame, cond, noise=None, cfg_scale=0.0):
        """
        1-step sampling: x = e - u(e, r=0, t=1)
        Optional inference CFG (standard classifier-free guidance):
            u_guided = (1+cfg_scale)*u_cond - cfg_scale*u_uncond
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        cond_frame = cond_frame.to(device=device, dtype=dtype)
        cond = cond.to(device=device, dtype=dtype)

        e = torch.randn(shape, device=device, dtype=dtype) if noise is None else noise.to(device=device, dtype=dtype)
        b = shape[0]
        r = torch.zeros(b, 1, 1, device=device, dtype=dtype)
        t = torch.ones(b, 1, 1, device=device, dtype=dtype)

        if cfg_scale > 0.0:
            u_cond = self.model(e, cond_frame, cond, r, t, cond_drop_prob=0.0)
            u_uncond = self.model(e, cond_frame, cond, r, t, cond_drop_prob=1.0)
            u = (1.0 + cfg_scale) * u_cond - cfg_scale * u_uncond
        else:
            u = self.model(e, cond_frame, cond, r, t, cond_drop_prob=0.0)

        return e - u

    def render_sample(
        self,
        shape,
        cond_frame,
        cond,
        normalizer=None,
        epoch=None,
        render_out=None,
        last_half=None,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder=None,
        start_point=None,
        render=True,
        cfg_scale=0.0,   # MeanFlow 전용 옵션(추가)
    ):
        if isinstance(shape, tuple):
            if mode == "normal":
                samples = self.sample_one_step(shape, cond_frame, cond, noise=noise, cfg_scale=cfg_scale).detach().cpu()
            else:
                raise ValueError(f"Unsupported mode for MeanFlow: {mode}")
        else:
            samples = shape

        if render_out is None:
            return samples

        if name is None:
            name = [f"sample_{i}.npy" for i in range(samples.shape[0])]

        os.makedirs(render_out, exist_ok=True)
        for i in range(samples.shape[0]):
            np.save(f"{render_out}/{epoch}_{os.path.basename(name[i])[:-4]}.npy", samples[i].numpy())

        return samples