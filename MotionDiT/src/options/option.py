from __future__ import annotations
from typing import Tuple
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
    experiment_name: str = "ditto_improved_meanflow_hdtf"   ### experiment_name ditto_original_hdtf, ditto_meanflow_hdtf, ditto_improved_meanflow_hdtf

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
    use_meanflow: bool = True          ##### use MeanFlow (single-step) instead of diffusion
    time_sampler: str = "logit_normal"  ##### meanflow time sampler: [uniform, logit_normal]

    ########## train ##########
    use_accelerate: bool = True         # use_accelerate flag for multi gpu
    precision: str = "no"             # precision: no, fp16, bf16
    epochs: int = 1000      # epochs
    batch_size: int = 256   # batch_size
    num_workers: int = 0    # num_workers
    lr: float = 1e-4        # lr (MeanFlow 논문 권장: 1e-4)
    # optimizer: str = "adam" # optimizer: adam (MeanFlow 권장), adan (original일떄는 그냥 주석처리)
    part_w_dict_json: str = ""    ###  part loss weights dict json
    use_last_frame_loss: bool = False    # use_last_frame_loss flag
    use_reg_loss: bool = False    # use_reg_loss flag
    dim_ws_npy: str = ""    ###  dim_ws npy

    # checkpoint: str = "None"
    # checkpoint: str = "/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch/models/lmdm_v0.4_hubert.pth"
    checkpoint: str = "/workspace/ditto/ditto-talkinghead-train/experiments/ditto_improved_meanflow_hdtf_20260115_180331/weights/train_92.pt"

    save_ckpt_freq: int = 1    # save ckpt freq (epoch)


    ########## sample ##########
    sample_interval: int = 500    # sample interval (epoch)
    sample_audio_path: str = "/workspace/ditto/ditto-talkinghead-train/example/audio.wav"    # sample audio path
    sample_source_path: str = "/workspace/ditto/ditto-talkinghead-train/example/image.png"    # sample source path
    sample_cfg_pkl: str = "/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"    # sample cfg pkl
    sample_data_root: str = "/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch"    # sample data root
    sample_checkpoint_path: str = ""    # sample checkpoint path
    # sample_output_path: str = "/workspace/ditto/ditto-talkinghead-train/ditto_original_output/1000step.mp4"    # sample output path

    ########## wandb ##########
    wandb_pj_name: str = "ditto-talkinghead-train_imf"    # wandb project name: ditto-talkinghead-train, ditto-talkinghead-train_mf
    wandb_log_freq: int = 50    # wandb logging frequency (50 iterations)

def check_train_opt(opt: TrainOptions):
    assert opt.experiment_dir, opt.experiment_dir
    assert opt.experiment_name, opt.experiment_name
    assert opt.data_list_json, opt.data_list_json
