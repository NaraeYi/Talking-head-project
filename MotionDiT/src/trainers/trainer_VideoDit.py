"""
Video DiT Trainer
기존 trainer.py(LMDM) 구조를 그대로 따르면서
Video DiT + MotionEncoder를 학습하는 trainer.

학습 대상:  AppearanceProjector + MotionEncoder + LTXVideoDiTWrapper(transformer + motion_proj + appearance_proj)
Frozen:     Ditto Stage1 (Audio2Feat + Motion DiT) + Appearance Extractor + Temporal VAE
학습 방식:  Standard Flow Matching (MeanFlow 미적용)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import time
import math
import traceback
import numpy as np
import imageio.v2 as imageio
from tqdm import trange, tqdm
from typing import Optional
from contextlib import nullcontext

from ..models.LMDM_VideoDit import (
    AppearanceProjector,
    MotionEncoder,
    TemporalVAE,
    LTXVideoDiTWrapper,
    VideoFlowMatching,
)
from ..datasets.video_dit_dataset import VideoDiTDataset
from ..utils.utils import DictAverageMeter, dump_pkl


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")
    wandb = None


def _build_video_dit_checkpoint_config(opt) -> dict:
    return {
        "vol_channels": getattr(opt, "vol_channels", 32),
        "vol_depth": getattr(opt, "vol_depth", 16),
        "pool_hw": getattr(opt, "pool_hw", 2),
        "appearance_dim": getattr(opt, "appearance_dim", 256),
        "motion_dim": getattr(opt, "motion_dim", 265),
        "motion_encoder_hidden": getattr(opt, "motion_encoder_hidden", 512),
        "motion_encoder_out_dim": getattr(opt, "motion_encoder_out_dim", 2048),
        "latent_channels": getattr(opt, "latent_channels", 128),
        "patch_size": getattr(opt, "patch_size", 1),
        "video_dit_hidden": getattr(opt, "video_dit_hidden", 2048),
        "video_dit_layers": getattr(opt, "video_dit_layers", 28),
        "video_dit_heads": getattr(opt, "video_dit_heads", 32),
        "ltx_pretrained_path": getattr(opt, "ltx_pretrained_path", ""),
        "vae_pretrained_path": getattr(opt, "vae_pretrained_path", ""),
    }


def _warn_if_ckpt_config_mismatch(ckpt_cfg: dict, opt):
    if not ckpt_cfg:
        return

    current_cfg = _build_video_dit_checkpoint_config(opt)
    mismatches = []
    for key, expected in ckpt_cfg.items():
        if key not in current_cfg:
            continue
        current = current_cfg[key]
        if expected != current:
            mismatches.append((key, expected, current))

    if not mismatches:
        return

    print("[VideoDiTTrainer] WARNING: checkpoint config differs from current options.")
    print("[VideoDiTTrainer] Resume will use current CLI/model options, so mismatched layers or dims can load partially with strict=False.")
    for key, ckpt_value, current_value in mismatches:
        print(f"  - {key}: checkpoint={ckpt_value!r}, current={current_value!r}")


class _TrainableVideoDiTBundle(nn.Module):
    """Bundle all trainable modules so Accelerate/DDP wraps a single module."""

    def __init__(
        self,
        appearance_projector: AppearanceProjector,
        motion_encoder: MotionEncoder,
        flow_matching: VideoFlowMatching,
    ):
        super().__init__()
        self.appearance_projector = appearance_projector
        self.motion_encoder = motion_encoder
        self.flow_matching = flow_matching

    def forward(
        self,
        *,
        clean_latent: torch.Tensor,
        motion_seq: torch.Tensor,
        f_s: torch.Tensor,
    ):
        appearance_cond = self.appearance_projector(f_s)
        motion_cond = self.motion_encoder(motion_seq)
        return self.flow_matching(
            clean_latent=clean_latent,
            motion_cond=motion_cond,
            appearance_cond=appearance_cond,
        )

    def sample_latent(
        self,
        *,
        motion_seq: torch.Tensor,
        f_s: torch.Tensor,
        latent_shape: tuple,
        num_steps: int = 25,
        cfg_scale: float = 3.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        appearance_cond = self.appearance_projector(f_s)
        motion_cond = self.motion_encoder(motion_seq)
        return self.flow_matching.sample(
            motion_cond=motion_cond,
            appearance_cond=appearance_cond,
            latent_shape=latent_shape,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            device=device,
            dtype=dtype,
        )


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────

class VideoDiTTrainer:
    """
    기존 Trainer(LMDM)와 동일한 인터페이스.

    _init_model     → AppearanceProjector + MotionEncoder + LTXVideoDiTWrapper + VideoFlowMatching
    _init_dataset   → VideoDiTDataset
    _init_optim     → Adam  (AppearanceProjector + MotionEncoder + Video DiT)
    _train_one_step → AppearanceProjector(f_s) + VideoFlowMatching.forward()
    """

    def __init__(self, opt):
        self.opt = opt
        self.wandb_run = None
        self.exp_path = None
        self.loss_logger = None
        self.use_torch_ddp = False
        self.world_size = 1
        self.local_rank = 0
        self.data_sampler = None

        print(time.asctime(), "_init_accelerate")
        self._init_accelerate()

        print(time.asctime(), "_init_vae")
        self.vae = self._init_vae()

        print(time.asctime(), "_init_model")
        self.trainable = self._init_model()

        print(time.asctime(), "_init_dataset")
        self.data_loader = self._init_dataset()

        print(time.asctime(), "_init_optim")
        self.optim = self._init_optim()

        print(time.asctime(), "_set_accelerate")
        self._set_accelerate()

        print(time.asctime(), "_init_log")
        self._init_log()

    # ── accelerate ──────────────────────────────

    def _init_accelerate(self):
        opt = self.opt
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.world_size = max(1, world_size)

        if self.world_size > 1:
            backend = getattr(opt, "distributed_backend", "gloo")
            self.use_torch_ddp = True
            self.accelerator = None
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.process_index = int(os.environ.get("RANK", str(self.local_rank)))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            self.is_main = self.process_index == 0
            if self.is_main:
                print(
                    f"[DDP] backend={backend}, world_size={self.world_size}, "
                    f"device={self.device}"
                )
            return

        if getattr(opt, "use_accelerate", False):
            from accelerate import Accelerator
            precision = getattr(opt, "precision", "bf16")
            self.accelerator = Accelerator(split_batches=False, mixed_precision=precision)
            self.device = self.accelerator.device
            self.is_main = self.accelerator.is_main_process
            self.process_index = self.accelerator.process_index
            self.local_rank = 0
            if self.is_main:
                print(
                    f"[Accelerate] distributed_type={self.accelerator.distributed_type}, "
                    f"num_processes={self.accelerator.num_processes}, "
                    f"device={self.device}"
                )
        else:
            self.accelerator = None
            self.device = torch.device("cuda")
            self.is_main = True
            self.process_index = 0
            self.local_rank = 0

    def _set_accelerate(self):
        if self.use_torch_ddp:
            self.trainable = DDP(
                self.trainable,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            return

        if self.accelerator is None:
            return
        self.trainable, self.optim, self.data_loader = self.accelerator.prepare(
            self.trainable, self.optim, self.data_loader
        )
        self.accelerator.wait_for_everyone()

    # ── VAE (frozen) ────────────────────────────

    def _init_vae(self):
        vae_path = getattr(self.opt, "vae_pretrained_path", None)
        vae = TemporalVAE(
            pretrained_path=vae_path,
            device=self.device,
            fail_on_fallback=getattr(self.opt, "require_vae_pretrained", True),
        )
        # VAE는 항상 frozen
        for p in vae.vae.parameters():
            p.requires_grad_(False)
        return vae

    # ── Model ───────────────────────────────────

    def _init_model(self):
        opt = self.opt

        # AppearanceProjector: f_s (32,16,64,64) → appearance tokens (N_a, appearance_dim)
        # AppearanceExtractor는 frozen. AppearanceProjector는 Video DiT와 함께 학습.
        appearance_projector = AppearanceProjector(
            vol_channels=getattr(opt, "vol_channels", 32),
            vol_depth=getattr(opt, "vol_depth", 16),
            pool_hw=getattr(opt, "pool_hw", 2),
            out_dim=getattr(opt, "appearance_dim", 256),
        ).to(self.device)

        motion_encoder = MotionEncoder(
            motion_dim=getattr(opt, "motion_dim", 265),
            hidden_dim=getattr(opt, "motion_encoder_hidden", 512),
            out_dim=getattr(opt, "motion_encoder_out_dim", 2048),
        ).to(self.device)

        dit = LTXVideoDiTWrapper(
            pretrained_path=getattr(opt, "ltx_pretrained_path", ""),
            latent_channels=getattr(opt, "latent_channels", 128),
            motion_cond_dim=getattr(opt, "motion_encoder_out_dim", 2048),
            appearance_cond_dim=getattr(opt, "appearance_dim", 256),
            patch_size=getattr(opt, "patch_size", 1),
            device=self.device,
            allow_scratch=not getattr(opt, "require_ltx_pretrained", True),
            inner_dim=getattr(opt, "video_dit_hidden", 2048),
            num_layers=getattr(opt, "video_dit_layers", 28),
            num_heads=getattr(opt, "video_dit_heads", 32),
        ).to(self.device)

        flow_matching = VideoFlowMatching(
            model=dit,
            cond_drop_prob=getattr(opt, "cond_drop_prob", 0.1),
        ).to(self.device)

        # 체크포인트 로드
        ckpt_path = getattr(opt, "video_dit_checkpoint", "")
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            _warn_if_ckpt_config_mismatch(ckpt.get("video_dit_config", {}), opt)
            appearance_projector.load_state_dict(
                ckpt.get("appearance_projector_state_dict", {}), strict=False
            )
            motion_encoder.load_state_dict(
                ckpt.get("motion_encoder_state_dict", {}), strict=False
            )
            flow_matching.model.load_state_dict(
                ckpt.get("video_dit_state_dict", {}), strict=False
            )
            print(f"[VideoDiTTrainer] Checkpoint loaded: {ckpt_path}")

        trainable = _TrainableVideoDiTBundle(
            appearance_projector=appearance_projector,
            motion_encoder=motion_encoder,
            flow_matching=flow_matching,
        )

        n_params = sum(p.numel() for p in trainable.parameters())
        print(f"[VideoDiTTrainer] Trainable params: {n_params:,}")

        return trainable

    # ── Dataset ─────────────────────────────────

    def _init_dataset(self):
        opt = self.opt

        dataset = VideoDiTDataset(
            data_list_json=getattr(opt, "video_dit_data_list_json", ""),
            seq_frames=getattr(opt, "seq_frames", 25),
            fixed_start_frame=(
                None if getattr(opt, "fixed_start_frame", -1) < 0
                else getattr(opt, "fixed_start_frame", -1)
            ),
            out_h=getattr(opt, "out_h", 512),
            out_w=getattr(opt, "out_w", 512),
            use_precomputed_latent=getattr(opt, "use_precomputed_latent", False),
        )
        self.sample_dataset = dataset

        self.data_sampler = None
        if self.use_torch_ddp:
            self.data_sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.process_index,
                shuffle=True,
                drop_last=True,
            )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=getattr(opt, "batch_size", 1),
            num_workers=getattr(opt, "num_workers", 4),
            shuffle=self.data_sampler is None,
            sampler=self.data_sampler,
            pin_memory=True,
            drop_last=True,
        )
        return loader

    # ── Optimizer ───────────────────────────────

    def _init_optim(self):
        opt = self.opt
        params = list(self.trainable.parameters())
        lr = getattr(opt, "video_dit_lr", 1e-4)
        weight_decay = getattr(opt, "weight_decay", 0.02)
        optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        print(f"[VideoDiTTrainer] Optimizer: Adam, lr={lr}, weight_decay={weight_decay}")

        # 체크포인트에서 optimizer 상태 복원
        ckpt_path = getattr(opt, "video_dit_checkpoint", "")
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "optimizer_state_dict" in ckpt:
                optim.load_state_dict(ckpt["optimizer_state_dict"])
                print(f"[VideoDiTTrainer] Optimizer state restored.")
        return optim

    # ── Log ─────────────────────────────────────

    def _init_log(self):
        opt = self.opt
        if not self.is_main:
            return

        import datetime
        from datetime import timezone, timedelta
        KST = timezone(timedelta(hours=9))
        ts = datetime.datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        exp_name = getattr(opt, "experiment_name", "video_dit")
        exp_path = os.path.join(
            getattr(opt, "experiment_dir", "experiments"),
            f"{exp_name}_{ts}",
        )
        self.exp_path = exp_path
        self.ckpt_path = os.path.join(exp_path, "weights")
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.sample_path = os.path.join(exp_path, "samples")
        os.makedirs(self.sample_path, exist_ok=True)

        opt_pkl = os.path.join(exp_path, "opt.pkl")
        dump_pkl(vars(opt), opt_pkl)

        loss_log = os.path.join(exp_path, "loss.log")
        self.loss_logger = open(loss_log, "a")
        print(f"[VideoDiTTrainer] Experiment: {exp_path}")

        if WANDB_AVAILABLE and wandb is not None:
            try:
                run_name = os.path.basename(exp_path)
                print(
                    f"[WANDB] Initializing wandb - project: {opt.wandb_pj_name}, "
                    f"name: {run_name}"
                )
                print(f"[WANDB] wandb_log_freq: {opt.wandb_log_freq} iterations")
                self.wandb_run = wandb.init(
                    project=opt.wandb_pj_name,
                    name=run_name,
                    config=vars(opt),
                )
                print(
                    f"[WANDB] Initialized successfully! Run URL: "
                    f"{wandb.run.url if wandb.run else 'N/A'}"
                )
            except Exception as e:
                self.wandb_run = None
                print(f"[WANDB] Initialization failed: {e}")

    @staticmethod
    def _to_float(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().item())
        return float(value)

    def _log_wandb_step(self, metrics: dict):
        if not self.is_main or not WANDB_AVAILABLE or wandb is None:
            return

        log_freq = max(1, int(getattr(self.opt, "wandb_log_freq", 50)))
        if self.global_step % log_freq != 0:
            return

        log_dict = {
            "Global_Step": self.global_step,
            "Epoch": self.epoch,
            **{k: self._to_float(v) for k, v in metrics.items()},
        }
        wandb.log(log_dict, step=self.global_step)

    def _log_wandb_epoch(self, avg_metrics: dict):
        if not self.is_main or not WANDB_AVAILABLE or wandb is None:
            return

        log_dict = {
            "Epoch": self.epoch,
            **{f"epoch_avg_{k}": self._to_float(v) for k, v in avg_metrics.items()},
        }
        wandb.log(log_dict, step=self.global_step)

    def _finish_logging(self):
        if self.is_main and self.loss_logger is not None and not self.loss_logger.closed:
            self.loss_logger.close()

        if self.is_main and WANDB_AVAILABLE and wandb is not None and self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None

        if self.use_torch_ddp and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _unwrap_module(self, module):
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(module)
        return module

    def _autocast_context(self):
        if self.accelerator is not None:
            return self.accelerator.autocast()
        if self.use_torch_ddp:
            precision = getattr(self.opt, "precision", "bf16")
            if precision == "bf16":
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "fp16":
                return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @staticmethod
    def _align_decoded_video(video: torch.Tensor, target_frames: int) -> torch.Tensor:
        cur_frames = int(video.shape[2])
        if cur_frames < target_frames:
            pad = video[:, :, -1:].repeat(1, 1, target_frames - cur_frames, 1, 1)
            video = torch.cat([video, pad], dim=2)
        elif cur_frames > target_frames:
            video = video[:, :, :target_frames]
        return video

    @staticmethod
    def _video_to_uint8(video: torch.Tensor) -> np.ndarray:
        return (
            video[0]
            .detach()
            .clamp(0, 1)
            .mul(255)
            .byte()
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )

    def _save_video_mp4(self, path: str, frames: np.ndarray, fps: int = 25):
        writer = imageio.get_writer(
            path,
            fps=fps,
            format="FFMPEG",
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()

    def _save_epoch_sample(self, epoch: int):
        if not self.is_main or len(self.sample_dataset) == 0:
            return None

        sample_idx = getattr(self.opt, "sample_item_idx", 0) % len(self.sample_dataset)
        sample_start = getattr(self.opt, "sample_start_frame", 0)
        sample = self.sample_dataset.get_item(sample_idx, start=sample_start)

        motion_seq = sample["motion_seq"].unsqueeze(0).to(self.device)
        f_s = sample["f_s"].unsqueeze(0).to(self.device)
        seq_frames = int(motion_seq.shape[1])

        if sample["video"].dim() >= 4 and sample["video"].numel() > 1:
            gt_video = sample["video"].unsqueeze(0).to(self.device)
        else:
            gt_video = None

        trainable = self._unwrap_module(self.trainable)
        was_training = trainable.training
        trainable.eval()

        try:
            with torch.no_grad():
                if gt_video is None:
                    gt_latent = sample["vae_latent"].unsqueeze(0).to(self.device)
                    if not getattr(self.opt, "precomputed_latent_is_normalized", False):
                        gt_latent = self.vae.normalize_latent(gt_latent)
                    gt_video = self.vae.decode(gt_latent)
                else:
                    with self._autocast_context():
                        gt_latent = self.vae.encode(gt_video)

                gt_video = self._align_decoded_video(gt_video, seq_frames)

                with self._autocast_context():
                    pred_latent = trainable.sample_latent(
                        motion_seq=motion_seq,
                        f_s=f_s,
                        latent_shape=tuple(gt_latent.shape),
                        num_steps=getattr(self.opt, "sample_num_inference_steps", 25),
                        cfg_scale=getattr(self.opt, "sample_cfg_scale", 3.0),
                        device=self.device,
                        dtype=gt_latent.dtype,
                    )
                    pred_video = self.vae.decode(pred_latent)
                pred_video = self._align_decoded_video(pred_video, seq_frames)
        finally:
            trainable.train(was_training)

        out_path = os.path.join(self.sample_path, f"sample_epoch_{epoch:04d}.mp4")
        pred_np = self._video_to_uint8(pred_video)
        self._save_video_mp4(out_path, pred_np)

        if getattr(self.opt, "sample_save_comparison", False):
            gt_np = self._video_to_uint8(gt_video)
            separator = np.full((gt_np.shape[0], gt_np.shape[1], 8, 3), 255, dtype=np.uint8)
            comparison = np.concatenate([gt_np, separator, pred_np], axis=2)
            compare_path = os.path.join(self.sample_path, f"sample_epoch_{epoch:04d}_compare.mp4")
            self._save_video_mp4(compare_path, comparison)

        tqdm.write(f"[Sample] Saved epoch {epoch}: {out_path}")
        return out_path

    # ── Training step ────────────────────────────

    def _train_one_step(self, batch: dict):
        motion_seq = batch["motion_seq"]    # (B, T, 265)
        f_s        = batch["f_s"]           # (B, 32, 16, 64, 64)

        if self.accelerator is None:
            motion_seq = motion_seq.to(self.device)
            f_s        = f_s.to(self.device)

        # 1. VAE encode: GT face crop video → clean latent (frozen, no grad)
        use_precomputed = getattr(self.opt, "use_precomputed_latent", False)
        if use_precomputed and "vae_latent" in batch and batch["vae_latent"].dim() >= 4:
            clean_latent = batch["vae_latent"]
            if self.accelerator is None:
                clean_latent = clean_latent.to(self.device)
            if not getattr(self.opt, "precomputed_latent_is_normalized", False):
                clean_latent = self.vae.normalize_latent(clean_latent)
        else:
            video = batch["video"]  # (B, 3, T, H, W)
            if self.accelerator is None:
                video = video.to(self.device)
            with torch.no_grad():
                clean_latent = self.vae.encode(video)  # (B, 128, T//8, H//32, W//32)

        if self.accelerator is not None:
            with self.accelerator.autocast():
                loss, loss_dict = self.trainable(
                    clean_latent=clean_latent,
                    motion_seq=motion_seq,
                    f_s=f_s,
                )
        else:
            loss, loss_dict = self.trainable(
                clean_latent=clean_latent,
                motion_seq=motion_seq,
                f_s=f_s,
            )

        return loss, loss_dict

    def _loss_backward(self, loss):
        self.optim.zero_grad()
        grad_clip = getattr(self.opt, "grad_clip", 1.0)
        if self.accelerator is not None:
            self.accelerator.backward(loss)
            if grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    list(self.trainable.parameters()),
                    grad_clip,
                )
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.trainable.parameters()),
                    grad_clip,
                )
        self.optim.step()

    # ── Epoch ───────────────────────────────────

    def _train_one_epoch(self):
        self.trainable.train()

        dam = DictAverageMeter()
        pbar = tqdm(self.data_loader, disable=not self.is_main, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            self.global_step += 1
            loss, loss_dict = self._train_one_step(batch)
            self._loss_backward(loss)

            if self.is_main:
                metrics = {k: self._to_float(v) for k, v in loss_dict.items()}
                metrics["total_loss"] = self._to_float(loss)
                dam.update(metrics)
                self._log_wandb_step(metrics)

                avg_metrics = dam.average()
                loss_str = f"loss: {avg_metrics.get('total_loss', 0.0):.4f}"
                if "video_dit_loss" in avg_metrics:
                    loss_str += f" | fm: {avg_metrics['video_dit_loss']:.4f}"
                pbar.set_postfix_str(loss_str)

        return dam

    def _save_ckpt(self, epoch: int, avg_loss: float):
        if not self.is_main:
            return

        trainable = self._unwrap_module(self.trainable)
        ap_state  = trainable.appearance_projector.state_dict()
        me_state  = trainable.motion_encoder.state_dict()
        dit_state = trainable.flow_matching.model.state_dict()

        ckpt = {
            "checkpoint_format_version": 2,
            "epoch":       epoch,
            "global_step": self.global_step,
            "appearance_projector_state_dict": ap_state,
            "motion_encoder_state_dict":       me_state,
            "video_dit_state_dict":            dit_state,
            "optimizer_state_dict":            self.optim.state_dict(),
            "video_dit_config": _build_video_dit_checkpoint_config(self.opt),
            "avg_loss":    avg_loss,
        }
        path = os.path.join(self.ckpt_path, f"video_dit_{epoch:04d}.pt")
        torch.save(ckpt, path)
        tqdm.write(f"[CKPT] Saved epoch {epoch}: {path}")

    # ── Main loop ───────────────────────────────

    def train_loop(self):
        opt = self.opt
        epochs = getattr(opt, "epochs", 500)
        save_freq = getattr(opt, "save_ckpt_freq", 50)

        self.global_step = 0

        # 체크포인트에서 시작 epoch 복원
        start_epoch = 1
        ckpt_path = getattr(opt, "video_dit_checkpoint", "")
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if "epoch" in ckpt:
                    start_epoch = ckpt["epoch"] + 1
                    self.global_step = ckpt.get("global_step", 0)
                    print(f"[Resume] Starting from epoch {start_epoch}")
            except Exception as e:
                print(f"[Warning] Could not restore epoch info: {e}")

        try:
            epoch_pbar = trange(start_epoch, epochs + 1, disable=not self.is_main, desc="Training")
            for epoch in epoch_pbar:
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                elif self.use_torch_ddp and dist.is_initialized():
                    dist.barrier()

                self.epoch = epoch
                if self.data_sampler is not None:
                    self.data_sampler.set_epoch(epoch)
                dam = self._train_one_epoch()

                if self.is_main:
                    avg_metrics = dam.average() if dam.initialized else {}
                    avg_loss = float(avg_metrics.get("total_loss", 0.0))
                    epoch_pbar.set_postfix_str(f"avg_loss: {avg_loss:.4f}")

                    avg_loss_msg = "|"
                    for k, v in avg_metrics.items():
                        avg_loss_msg += f" {k}: {float(v):.6f} |"
                    msg = f"Epoch: {epoch}, Global_Steps: {self.global_step}, {avg_loss_msg}\n"
                    self.loss_logger.write(msg)
                    self.loss_logger.flush()
                    self._log_wandb_epoch(avg_metrics)

                    sample_freq = getattr(opt, "save_sample_freq", 1)
                    if sample_freq > 0 and epoch % sample_freq == 0:
                        self._save_epoch_sample(epoch)

                    if epoch % save_freq == 0:
                        self._save_ckpt(epoch, avg_loss)
        finally:
            self._finish_logging()

        print(time.asctime(), "Training done.")
