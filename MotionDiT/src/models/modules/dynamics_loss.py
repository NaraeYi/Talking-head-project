"""
특정 구조에 묶이지 않았고, 그냥 pred: (B,T,D)와 gt: (B,T,D)만 주면 돌아간다.
둘 다에서 재사용 가능한 공용 모듈
original Ditto에서는 pred=x_recon, gt=x_start
MeanFlow에서는 pred=x_hat, gt=x


Temporal dynamics matching losses for motion latent supervision.

This module is intentionally standalone so it can be integrated later into
either:

1. MeanFlow training with `pred=x_hat`, `gt=x`
2. Original diffusion training with `pred=x_recon`, `gt=x_start`

The design follows the research motivation:
- L_var : per-channel temporal variance matching
- L_spec: temporal FFT power spectrum matching with high-frequency emphasis
- L_diff: first/second temporal difference matching

It also includes two practical helpers that are useful for later experiments:
- lambda ramp scheduling across training progress
- GT-active channel weighting based on GT temporal energy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_sequence_pair(pred: torch.Tensor, gt: torch.Tensor) -> None:
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(
            f"Expected (B, T, D) tensors, got pred={tuple(pred.shape)}, gt={tuple(gt.shape)}"
        )
    if pred.shape != gt.shape:
        raise ValueError(
            f"Pred/GT shape mismatch: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}"
        )
    if pred.shape[1] < 3:
        raise ValueError(
            f"Temporal losses require at least 3 frames, got T={pred.shape[1]}"
        )


def _pointwise_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(pred, gt, reduction="none")
    if loss_type == "l2":
        return F.mse_loss(pred, gt, reduction="none")
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _weighted_mean(
    loss_map: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if weights is None:
        return loss_map.mean()

    if weights.ndim == loss_map.ndim - 1:
        weights = weights.unsqueeze(1)

    weights = weights.to(device=loss_map.device, dtype=loss_map.dtype)
    weighted = loss_map * weights
    denom = weights.expand_as(loss_map).sum().clamp_min(eps)
    return weighted.sum() / denom


def temporal_difference(seq: torch.Tensor, order: int = 1) -> torch.Tensor:
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    out = seq
    for _ in range(order):
        out = out[:, 1:] - out[:, :-1]
    return out


def temporal_variance(seq: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    return seq.var(dim=1, unbiased=unbiased)


def temporal_velocity_variance(
    seq: torch.Tensor,
    unbiased: bool = False,
) -> torch.Tensor:
    vel = temporal_difference(seq, order=1)
    return vel.var(dim=1, unbiased=unbiased)

# 66-bin logits -> degree 변환 함수 추가
# 66-bin logits -> softmax -> expected bin -> degree
def headpose_logits_to_degree(logits: torch.Tensor) -> torch.Tensor:
    """
    Decode LivePortrait/Ditto 66-bin head-pose logits to scalar degrees.

    The motion latent stores pitch/yaw/roll as classifier logits, but temporal
    dynamics are physically meaningful after softmax expectation over bins.
    """
    if logits.shape[-1] != 66:
        raise ValueError(
            f"Expected 66-bin head-pose logits, got shape={tuple(logits.shape)}"
        )

    dtype = logits.dtype
    probs = F.softmax(logits.float(), dim=-1)
    bin_idx = torch.arange(66, device=logits.device, dtype=probs.dtype)
    degree = (probs * bin_idx).sum(dim=-1, keepdim=True) * 3.0 - 97.5
    return degree.to(dtype=dtype)


def build_gt_active_channel_weights(
    gt: torch.Tensor,
    mode: str = "var",
    power: float = 1.0,
    min_weight: float = 0.1,
    max_weight: Optional[float] = None,
    normalize: str = "mean",
    detach: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build per-sample, per-channel weights from GT temporal activity.

    Returns:
        Tensor of shape (B, 1, D), normalized so average channel weight is ~1.
    """
    if gt.ndim != 3:
        raise ValueError(f"Expected gt shape (B, T, D), got {tuple(gt.shape)}")

    if mode == "var":
        activity = temporal_variance(gt, unbiased=False)
    elif mode == "vel_var":
        activity = temporal_velocity_variance(gt, unbiased=False)
    elif mode == "var_plus_vel":
        activity = (
            temporal_variance(gt, unbiased=False)
            + temporal_velocity_variance(gt, unbiased=False)
        )
    else:
        raise ValueError(f"Unsupported active weighting mode: {mode}")

    weights = (activity + eps).pow(power)

    if normalize == "mean":
        weights = weights / weights.mean(dim=-1, keepdim=True).clamp_min(eps)
    elif normalize == "max":
        weights = weights / weights.max(dim=-1, keepdim=True).values.clamp_min(eps)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unsupported normalize mode: {normalize}")

    weights = weights.clamp_min(min_weight)
    if max_weight is not None:
        weights = weights.clamp_max(max_weight)

    if detach:
        weights = weights.detach()

    return weights.unsqueeze(1)


def temporal_variance_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    channel_weights: Optional[torch.Tensor] = None,
    use_log: bool = True,
    loss_type: str = "l1",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    _validate_sequence_pair(pred, gt)

    pred_var = temporal_variance(pred, unbiased=False)
    gt_var = temporal_variance(gt, unbiased=False)

    if use_log:
        pred_stat = torch.log(pred_var + eps)
        gt_stat = torch.log(gt_var + eps)
    else:
        pred_stat = pred_var
        gt_stat = gt_var

    loss_map = _pointwise_loss(pred_stat, gt_stat, loss_type=loss_type).unsqueeze(1)
    loss = _weighted_mean(loss_map, channel_weights, eps=eps)

    stats = {
        "pred_var_mean": pred_var.mean().detach(),
        "gt_var_mean": gt_var.mean().detach(),
    }
    return loss, stats


def _build_frequency_weights(
    num_bins: int,
    highfreq_ratio: float = 0.5,
    highfreq_weight: float = 2.0,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not 0.0 <= highfreq_ratio <= 1.0:
        raise ValueError(f"highfreq_ratio must be in [0, 1], got {highfreq_ratio}")

    weights = torch.ones(num_bins, device=device, dtype=dtype)
    start_idx = int(num_bins * highfreq_ratio)
    weights[start_idx:] = highfreq_weight
    return weights


def temporal_spectrum_power(
    seq: torch.Tensor,
    *,
    remove_dc: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    centered = seq - seq.mean(dim=1, keepdim=True) if remove_dc else seq
    spectrum = torch.fft.rfft(centered, dim=1)
    power = spectrum.abs().pow(2)
    return power + eps


def temporal_spectral_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    channel_weights: Optional[torch.Tensor] = None,
    highfreq_ratio: float = 0.5,
    highfreq_weight: float = 2.0,
    use_log_power: bool = True,
    loss_type: str = "l1",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    _validate_sequence_pair(pred, gt)

    pred_pow = temporal_spectrum_power(pred, remove_dc=True, eps=eps)
    gt_pow = temporal_spectrum_power(gt, remove_dc=True, eps=eps)

    if use_log_power:
        pred_stat = torch.log(pred_pow)
        gt_stat = torch.log(gt_pow)
    else:
        pred_stat = pred_pow
        gt_stat = gt_pow

    loss_map = _pointwise_loss(pred_stat, gt_stat, loss_type=loss_type)
    freq_weights = _build_frequency_weights(
        loss_map.shape[1],
        highfreq_ratio=highfreq_ratio,
        highfreq_weight=highfreq_weight,
        device=loss_map.device,
        dtype=loss_map.dtype,
    ).view(1, -1, 1)
    loss_map = loss_map * freq_weights
    loss = _weighted_mean(loss_map, channel_weights, eps=eps)

    stats = {
        "pred_spec_mean": pred_pow.mean().detach(),
        "gt_spec_mean": gt_pow.mean().detach(),
    }
    return loss, stats


def temporal_difference_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    channel_weights: Optional[torch.Tensor] = None,
    velocity_weight: float = 1.0,
    acceleration_weight: float = 1.0,
    loss_type: str = "l1",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    _validate_sequence_pair(pred, gt)

    vel_pred = temporal_difference(pred, order=1)
    vel_gt = temporal_difference(gt, order=1)
    acc_pred = temporal_difference(pred, order=2)
    acc_gt = temporal_difference(gt, order=2)

    vel_map = _pointwise_loss(vel_pred, vel_gt, loss_type=loss_type)
    acc_map = _pointwise_loss(acc_pred, acc_gt, loss_type=loss_type)

    vel_loss = _weighted_mean(vel_map, channel_weights, eps=eps)
    acc_loss = _weighted_mean(acc_map, channel_weights, eps=eps)
    total = velocity_weight * vel_loss + acceleration_weight * acc_loss

    stats = {
        "vel_loss": vel_loss.detach(),
        "acc_loss": acc_loss.detach(),
    }
    return total, stats


def linear_ramp_weight(
    *,
    global_step: Optional[int] = None,
    total_steps: Optional[int] = None,
    progress: Optional[float] = None,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    max_value: float = 1.0,
) -> float:
    """
    Returns a scalar in [0, max_value] for delayed auxiliary-loss activation.
    """
    if progress is None:
        if global_step is None or total_steps is None or total_steps <= 0:
            return max_value
        progress = float(global_step) / float(total_steps)

    progress = max(0.0, min(1.0, float(progress)))
    start_ratio = float(start_ratio)
    end_ratio = float(end_ratio)

    if end_ratio <= start_ratio:
        return max_value if progress >= end_ratio else 0.0
    if progress <= start_ratio:
        return 0.0
    if progress >= end_ratio:
        return max_value

    local = (progress - start_ratio) / (end_ratio - start_ratio)
    return float(local * max_value)


@dataclass
class DynamicsLossConfig:
    lambda_var: float = 0.0
    lambda_spec: float = 0.0
    lambda_diff: float = 0.0

    var_use_log: bool = True
    spec_use_log_power: bool = True
    spec_highfreq_ratio: float = 0.5
    spec_highfreq_weight: float = 2.0

    diff_velocity_weight: float = 1.0
    diff_acceleration_weight: float = 1.0

    loss_type: str = "l1"

    schedule_start_ratio: float = 0.0
    schedule_end_ratio: float = 0.0  # dyn loss: constant weighting unless a ramp is explicitly requested
    schedule_max_value: float = 1.0

    active_weighting: str = "none"
    active_weight_mode: str = "var"
    active_weight_power: float = 1.0
    active_weight_min: float = 0.1
    active_weight_max: Optional[float] = None
    active_weight_normalize: str = "mean"
    active_weight_detach: bool = True

    # dyn loss: optionally restrict auxiliary losses to selected latent parts only.
    selected_parts: Tuple[str, ...] = ("all",)
    selected_dim_ranges: Optional[Tuple[Tuple[int, int], ...]] = None

    eps: float = 1e-6


class DynamicsMatchingLoss(nn.Module):
    """
    High-level wrapper that combines:
    - variance matching
    - spectral matching
    - temporal difference matching

    The output is ready to be added to the main training loss later.
    """

    def __init__(self, config: DynamicsLossConfig):
        super().__init__()
        self.config = config

    def _select_dims(self, seq: torch.Tensor) -> torch.Tensor:
        pose_parts = {"pitch", "yaw", "roll"}
        parts = self.config.selected_parts
        ranges = self.config.selected_dim_ranges
        if not ranges:
            if parts == ("all",) and seq.shape[-1] >= 199:
                return torch.cat(
                    [
                        seq[..., 0:1],
                        headpose_logits_to_degree(seq[..., 1:67]),
                        headpose_logits_to_degree(seq[..., 67:133]),
                        headpose_logits_to_degree(seq[..., 133:199]),
                        seq[..., 199:],
                    ],
                    dim=-1,
                )
            return seq

        selected = []
        for part, (s, e) in zip(parts, ranges):
            part_seq = seq[..., s:e]
            if part in pose_parts and e - s == 66:
                part_seq = headpose_logits_to_degree(part_seq)
            selected.append(part_seq)

        return torch.cat(selected, dim=-1)

    def _maybe_build_channel_weights(self, gt: torch.Tensor) -> Optional[torch.Tensor]:
        if self.config.active_weighting == "none":
            return None

        if self.config.active_weighting != "gt_active":
            raise ValueError(
                f"Unsupported active_weighting: {self.config.active_weighting}"
            )

        return build_gt_active_channel_weights(
            gt,
            mode=self.config.active_weight_mode,
            power=self.config.active_weight_power,
            min_weight=self.config.active_weight_min,
            max_weight=self.config.active_weight_max,
            normalize=self.config.active_weight_normalize,
            detach=self.config.active_weight_detach,
            eps=self.config.eps,
        )

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        global_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        progress: Optional[float] = None,
        channel_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_sequence_pair(pred, gt)

        # dyn loss: slice the latent first so auxiliaries can target selected parts.
        # LivePortrait pose logits are decoded to scalar degrees before L_var/L_spec/L_diff.
        pred = self._select_dims(pred)
        gt = self._select_dims(gt)

        cfg = self.config
        if channel_weights is None:
            channel_weights = self._maybe_build_channel_weights(gt)

        dyn_scale = linear_ramp_weight(
            global_step=global_step,
            total_steps=total_steps,
            progress=progress,
            start_ratio=cfg.schedule_start_ratio,
            end_ratio=cfg.schedule_end_ratio,
            max_value=cfg.schedule_max_value,
        )

        total = pred.new_zeros(())
        loss_dict: Dict[str, torch.Tensor] = {
            "dyn_scale": pred.new_tensor(dyn_scale),
            "dyn_selected_dim": pred.new_tensor(float(pred.shape[-1])),
        }

        if channel_weights is not None:
            loss_dict["dyn_channel_weight_mean"] = channel_weights.mean().detach()
            loss_dict["dyn_channel_weight_max"] = channel_weights.max().detach()

        if cfg.lambda_var > 0.0:
            l_var, stats = temporal_variance_loss(
                pred,
                gt,
                channel_weights=channel_weights,
                use_log=cfg.var_use_log,
                loss_type=cfg.loss_type,
                eps=cfg.eps,
            )
            total = total + cfg.lambda_var * l_var
            loss_dict["dyn_var"] = l_var.detach()
            loss_dict.update({f"dyn_{k}": v for k, v in stats.items()})

        if cfg.lambda_spec > 0.0:
            l_spec, stats = temporal_spectral_loss(
                pred,
                gt,
                channel_weights=channel_weights,
                highfreq_ratio=cfg.spec_highfreq_ratio,
                highfreq_weight=cfg.spec_highfreq_weight,
                use_log_power=cfg.spec_use_log_power,
                loss_type=cfg.loss_type,
                eps=cfg.eps,
            )
            total = total + cfg.lambda_spec * l_spec
            loss_dict["dyn_spec"] = l_spec.detach()
            loss_dict.update({f"dyn_{k}": v for k, v in stats.items()})

        if cfg.lambda_diff > 0.0:
            l_diff, stats = temporal_difference_loss(
                pred,
                gt,
                channel_weights=channel_weights,
                velocity_weight=cfg.diff_velocity_weight,
                acceleration_weight=cfg.diff_acceleration_weight,
                loss_type=cfg.loss_type,
                eps=cfg.eps,
            )
            total = total + cfg.lambda_diff * l_diff
            loss_dict["dyn_diff"] = l_diff.detach()
            loss_dict["dyn_diff_vel"] = stats["vel_loss"]
            loss_dict["dyn_diff_acc"] = stats["acc_loss"]

        total = total * dyn_scale
        loss_dict["dyn_total"] = total.detach()
        return total, loss_dict
