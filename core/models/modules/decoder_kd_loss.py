# decoder_kd_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
teacher decoder가 뱉는 RGB 이미지를 “soft target”으로 두고,
L_out_kd: student 출력 vs teacher 출력 (L1)
L_gt: (가능하면) student 출력 vs GT 프레임 (L1)
선택: L_feat_kd: teacher/student 중간 feature(원하면 hook로 추가)
"""

class DecoderKDLoss(nn.Module):
    def __init__(
        self,
        w_out_kd: float = 1.0,
        w_gt: float = 1.0,
        w_tv: float = 0.0,   # optional: 아주 약하게 주면 깜빡임/노이즈 완화에 도움될 때 있음
    ):
        super().__init__()
        self.w_out_kd = w_out_kd
        self.w_gt = w_gt
        self.w_tv = w_tv

    @staticmethod
    def total_variation(x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)
        return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()

    def forward(
        self,
        I_s: torch.Tensor,          # student output RGB
        I_t: torch.Tensor,          # teacher output RGB (detach 권장)
        I_gt: torch.Tensor = None,  # ground-truth RGB (있으면 best)
    ):
        loss = 0.0
        logs = {}

        # Output KD (teacher RGB를 soft target)
        l_out_kd = F.l1_loss(I_s, I_t)
        loss = loss + self.w_out_kd * l_out_kd
        logs["l_out_kd"] = l_out_kd.detach()

        # Supervised (GT가 있을 때)
        if I_gt is not None:
            l_gt = F.l1_loss(I_s, I_gt)
            loss = loss + self.w_gt * l_gt
            logs["l_gt"] = l_gt.detach()

        if self.w_tv > 0:
            l_tv = self.total_variation(I_s)
            loss = loss + self.w_tv * l_tv
            logs["l_tv"] = l_tv.detach()

        logs["loss"] = loss.detach()
        return loss, logs
