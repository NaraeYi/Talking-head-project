from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass
import os


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR)))
PROJECT_DIR = "/workspace/ditto/ditto-talkinghead-train"
DEFAULT_EXPERIMENT_DIR = os.path.join(PROJECT_DIR, 'experiments')


class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __repr__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)
    

@dataclass(repr=False)  # use repr from PrintableConfig
class TrainOptions(PrintableConfig):
    ########## experiment ##########
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR    # experiment_dir
    experiment_name: str = "decoded_posebranch_freeze_Lv75Ls05_spyrt"   ### experiment_name ditto_original_hdtf, ditto_meanflow_hdtf, ditto_improved_meanflow_hdtf

    ########## dataset ##########
    data_list_json: str = "/workspace/ditto/datasets/HDTF/HDTF_train/data_list.json"    ###  train data list: [[kps_npy, aud_npy, frame_num],]
    data_preload: bool = False  # data_preload flag
    data_preload_pkl: str = "/workspace/ditto/datasets/HDTF/HDTF_train/data_preload.pkl"  ###  save to data_preload_pkl
    reprepare_idx_map: bool = False    # reprepare_idx_map flag for dataset
    data_cache: bool = False    # data_cache flag

    mtn_mean_var_npy: str = ""    ### for mtn norm

    motion_feat_start: int = 0  # motion feat start dim
    motion_feat_offset_dim_se: tuple[int, ...] = ()    # cal offset dim range for part D-S

    use_emo: bool = True    # use_emo flag
    use_eye_open: bool = True    # use_eye_open flag
    use_eye_ball: bool = True    # use_eye_ball flag
    use_sc: bool = True    # use source canonical keypoints flag
    use_last_frame: bool = True    # use last frame as cond frame flag
    use_lmk: bool = False    # use mediapipe lmk as cond flag
    use_cond_end: bool = True    # use clip start and end frame as cond flag

    dataset_version: str = "v2"    # dataset version: [v1, v2]

    ########## model ##########
    motion_feat_dim: int = 265          # motion_feat_dim
    audio_feat_dim: int = 1103     # audio_feat_dim (1024 + 63 + 8 + 2 + 6)
    seq_frames: int = int(3.2 * 25)     # clip length
    use_meanflow: bool = False          ##### use MeanFlow (single-step) instead of diffusion
    meanflow_mode: str = "improved"    ##### "meanflow" (original) or "improved" (Improved MeanFlow)
    time_sampler: str = "logit_normal"  ##### meanflow time sampler: [uniform, logit_normal]

    # pose branch: lightweight pitch/yaw/roll residual adapter on top of the base transformer.
    use_pose_branch: bool = True   # True: use pose branch, False: use original transformer
    pose_branch_hidden_dim: int = 128
    pose_branch_dropout: float = 0.0
    pose_branch_residual_scale: float = 1.0  # 1.0 → 0.5/0.7으로 해볼 예정..
    pose_branch_gate_bias: float = -2.0

    ########## train ##########
    use_accelerate: bool = True         # use_accelerate flag for multi gpu
    precision: str = "no"             # precision: no, fp16, bf16
    epochs: int = 1000      # epochs
    batch_size: int = 1024    # batch_size 1024 → 2048
    num_workers: int = 0    # num_workers
    lr: float = 1e-5        # pose branch: backbone lr used when backbone is not frozen
    pose_branch_lr: float = 1e-4    # pose branch: higher lr for the residual adapter
    freeze_backbone_for_pose_branch: bool = True    # pose branch: train only pose_branch params for the first-stage experiment
    # optimizer: str = "adam" # optimizer: adam (MeanFlow 권장), adan (original일떄는 그냥 주석처리)
    part_w_dict_json: str = ""    ###  part loss weights dict json
    use_last_frame_loss: bool = False    # use_last_frame_loss flag
    use_reg_loss: bool = False    # use_reg_loss flag
    dim_ws_npy: str = ""    ###  dim_ws npy
    lambda_pva: float = 0.0    # dyn loss: MeanFlow only, weight for existing P/V/A auxiliary loss

    # dyn loss: temporal dynamics matching options (shared by original Ditto and MeanFlow)
    dyn_lambda_var: float = 0.075    # 0.05 → 0.1 → 0.075
    dyn_lambda_spec: float = 0.005   # 0.01 → 0.0(임시로 amplitude만 안정적으로 조절되는지 보고) → 0.005
    dyn_lambda_diff: float = 0.0
    dyn_var_use_log: bool = True
    dyn_spec_use_log_power: bool = True
    dyn_spec_highfreq_ratio: float = 0.5
    dyn_spec_highfreq_weight: float = 2.0
    dyn_diff_velocity_weight: float = 1.0
    dyn_diff_acceleration_weight: float = 1.0
    dyn_loss_type: str = "l1"
    dyn_schedule_start_ratio: float = 0.0
    dyn_schedule_end_ratio: float = 0.0    # dyn loss: 0.0 keeps constant weighting by default
    dyn_schedule_max_value: float = 1.0
    dyn_active_weighting: str = "none"
    dyn_active_weight_mode: str = "var"
    dyn_active_weight_power: float = 1.0
    dyn_active_weight_min: float = 0.1
    dyn_active_weight_max: Optional[float] = None
    dyn_active_weight_normalize: str = "mean"
    dyn_active_weight_detach: bool = True
    dyn_loss_parts: str = "scale,pitch,yaw,roll,t"    # dyn loss: e.g. "scale,pitch,yaw,roll,t" to exclude exp from L_var/L_spec / "all" to use all parts

    # checkpoint: str = "None"
    checkpoint: str = "/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch/models/lmdm_v0.4_hubert.pth"
    # checkpoint: str = "/workspace/ditto/ditto-talkinghead-train/experiments/ditto_original_posebranch_LvarLspec_spyrt_20260505_005857/weights/train_8.pt"
    # checkpoint: str = "/workspace/ditto/ditto-talkinghead-train/experiments/ditto_improved_meanflow_hdtf_20260115_180331/weights/train_92.pt"

    save_ckpt_freq: int = 1    # save ckpt freq (epoch)


    ########## sample ##########
    sample_every_epoch: bool = True    # epoch sample: save one sample video after every epoch
    sample_interval: int = 1    # epoch sample: fallback interval when sample_every_epoch is False
    sample_audio_path: str = "/workspace/ditto/ditto-talkinghead-train/example/audio.wav"    # sample audio path
    sample_source_path: str = "/workspace/ditto/ditto-talkinghead-train/example/image.png"    # sample source path
    sample_cfg_pkl: str = "/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"    # sample cfg pkl
    sample_data_root: str = "/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch"    # sample data root
    sample_checkpoint_path: str = ""    # sample checkpoint path
    # sample_output_path: str = "/workspace/ditto/ditto-talkinghead-train/ditto_original_output/1000step.mp4"    # sample output path

    ########## wandb ##########
    wandb_pj_name: str = "ditto-talkinghead-train_dynloss"    # wandb project name: ditto-talkinghead-train, ditto-talkinghead-train_mf
    wandb_run_name: str = "decoded_posebranch_freeze_Lv75Ls05_spyrt"    # dynloss_Lvar_only/dynloss_Lvar_Lspec epoch sample: optional explicit wandb run name for parallel experiments
    wandb_log_freq: int = 50    # wandb logging frequency (50 iterations)

def check_train_opt(opt: TrainOptions):
    assert opt.experiment_dir, opt.experiment_dir
    assert opt.experiment_name, opt.experiment_name
    assert opt.data_list_json, opt.data_list_json
