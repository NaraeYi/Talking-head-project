# Latent Motion Diffusion Model
import torch

from .modules.model import MotionDecoder, MotionDecoderMF
from .modules.diffusion import MotionDiffusion
from .modules.diffusion_mf_mode import MotionMeanFlow
from .modules.dynamics_loss import DynamicsLossConfig


FPS = 25
SEQ_SEC = 3.2


class LMDM:
    def __init__(
        self,
        motion_feat_dim=265,
        audio_feat_dim=1024+35,
        seq_frames=int(SEQ_SEC * FPS),
        part_w_dict=None,   # only for train
        checkpoint='',
        device='cuda',
        use_last_frame_loss=False,    # only for train
        use_reg_loss=False,    # only for train
        dim_ws=None,    # only for train
        use_meanflow=False,
        meanflow_mode="improved",  # "meanflow" or "improved"
        use_pose_branch=False,
        pose_branch_hidden_dim=128,
        pose_branch_dropout=0.0,
        pose_branch_residual_scale=1.0,
        pose_branch_gate_bias=-2.0,
        lambda_pva=0.0,
        dyn_lambda_var=0.0,
        dyn_lambda_spec=0.0,
        dyn_lambda_diff=0.0,
        dyn_var_use_log=True,
        dyn_spec_use_log_power=True,
        dyn_spec_highfreq_ratio=0.5,
        dyn_spec_highfreq_weight=2.0,
        dyn_diff_velocity_weight=1.0,
        dyn_diff_acceleration_weight=1.0,
        dyn_loss_type="l1",
        dyn_schedule_start_ratio=0.0,
        dyn_schedule_end_ratio=1.0,
        dyn_schedule_max_value=1.0,
        dyn_active_weighting="none",
        dyn_active_weight_mode="var",
        dyn_active_weight_power=1.0,
        dyn_active_weight_min=0.1,
        dyn_active_weight_max=None,
        dyn_active_weight_normalize="mean",
        dyn_active_weight_detach=True,
        dyn_loss_parts="all",
    ):
        self.motion_feat_dim = motion_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.seq_frames = seq_frames
        self.device = device

        # dyn loss: build one shared config object so original Ditto and MeanFlow
        # can use the same auxiliary-loss interface and options.
        dyn_part_w_dict = {
            "scale": (0, 1),
            "pitch": (1, 67),
            "yaw": (67, 133),
            "roll": (133, 199),
            "t": (199, 202),
            "exp": (202, 265),
        }
        dyn_selected_parts, dyn_selected_dim_ranges = self._parse_dyn_loss_parts(
            dyn_loss_parts,
            dyn_part_w_dict,
        )
        dyn_loss_config = DynamicsLossConfig(
            lambda_var=dyn_lambda_var,
            lambda_spec=dyn_lambda_spec,
            lambda_diff=dyn_lambda_diff,
            var_use_log=dyn_var_use_log,
            spec_use_log_power=dyn_spec_use_log_power,
            spec_highfreq_ratio=dyn_spec_highfreq_ratio,
            spec_highfreq_weight=dyn_spec_highfreq_weight,
            diff_velocity_weight=dyn_diff_velocity_weight,
            diff_acceleration_weight=dyn_diff_acceleration_weight,
            loss_type=dyn_loss_type,
            schedule_start_ratio=dyn_schedule_start_ratio,
            schedule_end_ratio=dyn_schedule_end_ratio,
            schedule_max_value=dyn_schedule_max_value,
            active_weighting=dyn_active_weighting,
            active_weight_mode=dyn_active_weight_mode,
            active_weight_power=dyn_active_weight_power,
            active_weight_min=dyn_active_weight_min,
            active_weight_max=dyn_active_weight_max,
            active_weight_normalize=dyn_active_weight_normalize,
            active_weight_detach=dyn_active_weight_detach,
            selected_parts=dyn_selected_parts,
            selected_dim_ranges=dyn_selected_dim_ranges,
        )

        if use_meanflow:
            model = MotionDecoderMF(
                nfeats=motion_feat_dim,
                seq_len=seq_frames,
                latent_dim=512,
                ff_size=1024,
                num_layers=8,
                num_heads=8,
                dropout=0.1,
                cond_feature_dim=audio_feat_dim,
                use_pose_branch=use_pose_branch,   # pose branch: shared-feature residual adapter.
                pose_branch_hidden_dim=pose_branch_hidden_dim,
                pose_branch_dropout=pose_branch_dropout,
                pose_branch_residual_scale=pose_branch_residual_scale,
                pose_branch_gate_bias=pose_branch_gate_bias,
            )

            diffusion = MotionMeanFlow(
                model,
                horizon=seq_frames,
                repr_dim=motion_feat_dim,

                # ---- MeanFlow 기본 ----
                loss_type="l2",
                cond_drop_prob=0.2,
                lambda_pva=lambda_pva,   # dyn loss: configurable MeanFlow P/V/A weight.
                dyn_loss_config=dyn_loss_config,   # dyn loss: shared auxiliary dynamics losses.

                # ---- PVA 설정(기존과 동일) ----
                part_w_dict=part_w_dict,
                use_last_frame_loss=use_last_frame_loss,
                use_reg_loss=use_reg_loss,
                dim_ws=dim_ws,

                # ---- MeanFlow 레퍼런스 옵션(직접 기입) ----
                path_type="linear",          # "linear" or "cosine"
                time_sampler="logit_normal", # "logit_normal" or "uniform" 
                time_mu=-0.4,
                time_sigma=1.0,
                ratio_r_not_equal_t=0.75,    # r != t 비율

                weighting="uniform",         # "uniform" or "adaptive"
                adaptive_p=1.0,

                # ---- CFG-like target (training) ----
                use_cfg_target=False,
                cfg_omega=1.0,
                cfg_kappa=0.0,
                cfg_min_t=0.0,
                cfg_max_t=0.8,

                # ---- 기타 ----
                detach_cond=True,
                use_r0_recon=True,
                meanflow_mode=meanflow_mode,
            )
        else:
            model = MotionDecoder(
                nfeats=motion_feat_dim,
                seq_len=seq_frames,
                latent_dim=512,
                ff_size=1024,
                num_layers=8,
                num_heads=8,
                dropout=0.1,
                cond_feature_dim=audio_feat_dim,
                use_pose_branch=use_pose_branch,   # pose branch: shared-feature residual adapter.
                pose_branch_hidden_dim=pose_branch_hidden_dim,
                pose_branch_dropout=pose_branch_dropout,
                pose_branch_residual_scale=pose_branch_residual_scale,
                pose_branch_gate_bias=pose_branch_gate_bias,
            )
            diffusion = MotionDiffusion(
                model,
                horizon=seq_frames,
                repr_dim=motion_feat_dim,
                n_timestep=1000,
                schedule="cosine",
                loss_type="l2",
                clip_denoised=True,
                predict_epsilon=False,
                guidance_weight=2,
                use_p2=False,
                cond_drop_prob=0.2,
                part_w_dict=part_w_dict,
                use_last_frame_loss=use_last_frame_loss,
                use_reg_loss=use_reg_loss,
                dim_ws=dim_ws,
                dyn_loss_config=dyn_loss_config,   # dyn loss: shared auxiliary dynamics losses.
            )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        if checkpoint:
            print('load ckpt')
            checkpoint = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        diffusion = diffusion.to(device)

        self.model = model
        self.diffusion = diffusion

    @staticmethod
    def _parse_dyn_loss_parts(dyn_loss_parts, part_ranges):
        # dyn loss: allow strings like "all" or "scale,pitch,yaw,roll,t"
        # so we can exclude exp without changing the core loss code again.
        if dyn_loss_parts is None:
            return ("all",), None

        if isinstance(dyn_loss_parts, str):
            raw_parts = [item.strip() for item in dyn_loss_parts.split(",") if item.strip()]
        else:
            raw_parts = [str(item).strip() for item in dyn_loss_parts if str(item).strip()]

        if not raw_parts or raw_parts == ["all"]:
            return ("all",), None

        invalid_parts = [part for part in raw_parts if part not in part_ranges]
        if invalid_parts:
            raise ValueError(
                f"Unsupported dyn_loss_parts={invalid_parts}. "
                f"Valid parts: {list(part_ranges.keys()) + ['all']}"
            )

        dim_ranges = tuple(part_ranges[part] for part in raw_parts)
        return tuple(raw_parts), dim_ranges

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def use_accelerator(self, accelerator):
        self.model = accelerator.prepare(self.model)
        self.diffusion = self.diffusion.to(accelerator.device)

    @torch.no_grad()
    def _run_diffusion_render_sample(self, kp_cond, aud_cond, noise=None):
        """
        kp_cond: [b, kp_dim], tensor
        aud_cond: [b, L, aud_dim], tensor
        pred_kp_seq: [b, L, kp_dim], tensor
        """
        device = self.device

        render_count = 1
        seq_frames = self.seq_frames
        motion_feat_dim = self.motion_feat_dim

        shape = (render_count, seq_frames, motion_feat_dim)
        cond_frame = kp_cond.to(device)
        cond = aud_cond.to(device)

        pred_kp_seq = self.diffusion.render_sample(
            shape,
            cond_frame,
            cond,
            normalizer=None,
            epoch=None,
            render_out=None,
            last_half=None,
            mode="normal",
            noise=noise,
        )
        return pred_kp_seq
