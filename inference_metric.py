"""
Test set에 대한 inference를 수행하여 결과 비디오를 생성하는 스크립트

사용법:
    # Original checkpoint 사용
    python inference_metric.py
    
    # 특정 checkpoint 사용
    python inference_metric.py --checkpoint_path /path/to/weights/train_50.pt
    
    # MeanFlow 모드
    python inference_metric.py --checkpoint_path /path/to/weights/train_50.pt --use_meanflow
"""

import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

# from stream_pipeline_offline import StreamSDK
from stream_pipeline_offline_retargeting_faster import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def get_model_name(checkpoint_path):
    """checkpoint_path에서 모델 이름 추출"""
    if checkpoint_path is None:
        return "original"
    
    # checkpoint_path 예: .../experiments/ditto_meanflow_hdtf_20251217_142752/weights/train_10.pt
    # 모델명: ditto_meanflow_hdtf_20251217_142752_train_10
    path = Path(checkpoint_path)
    weight_name = path.stem  # train_10
    
    # experiments 폴더 이름 추출
    if "experiments" in str(path):
        parts = path.parts
        exp_idx = parts.index("experiments")
        if exp_idx + 1 < len(parts):
            exp_name = parts[exp_idx + 1]  # ditto_meanflow_hdtf_20251217_142752
            return f"{exp_name}_{weight_name}"
    
    return weight_name


def run_single(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, max_seconds: float = None, more_kwargs: str | dict = {}):
    """단일 클립에 대한 inference 수행
    
    Args:
        max_seconds: 최대 오디오 길이 (초). None이면 전체 길이 사용
    """
    import soundfile as sf

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})
    
    tmp_output_path = output_path.replace(".mp4", "_tmp.mp4")

    # retargeting 설정
    setup_kwargs.update({
        "lp_retarget_enable": True,  # True False
        "lp_checkpoint_S": "/workspace/ditto/ditto-talkinghead-train/prepare_data_train/LivePortrait/pretrained_weights/stitching_retargeting_module.pth",
        "lp_models_yaml": "/workspace/ditto/ditto-talkinghead-train/prepare_data_train/LivePortrait/src/config/models.yaml",
        "lp_target_eye_ratio": 0.39,
        "lp_target_lip_ratio": 0.0,
        "lp_first_n": 10000,
        "lp_fade_n": 10,
        "lp_apply_to": "driving",
        "lp_device": "cuda:0",
    })

    SDK.setup(source_path, tmp_output_path, **setup_kwargs)
    
    audio, sr = librosa.core.load(audio_path, sr=16000)
    
    # 오디오 길이를 max_seconds에 맞춰 고정한다.
    # MEAD evaluation에서는 4.84s / 121 frames로 통일하고 싶을 때가 많아서,
    # 긴 오디오는 자르고 짧은 오디오는 무음으로 패딩한다.
    audio_modified = False
    if max_seconds is not None:
        max_samples = int(max_seconds * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            audio_modified = True
        elif len(audio) < max_samples:
            pad = np.zeros(max_samples - len(audio), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=0)
            audio_modified = True
        num_f = math.ceil(max_seconds * 25)
    else:
        num_f = math.ceil(len(audio) / 16000 * 25)
    
    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
    
    # Offline mode
    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()
    
    # 오디오 준비 (잘랐거나 패딩했으면 임시 파일로 저장)
    audio_for_merge = audio_path
    temp_audio_path = None
    
    if audio_modified:
        temp_audio_path = output_path.replace(".mp4", "_temp_audio.wav")
        sf.write(temp_audio_path, audio, 16000)
        audio_for_merge = temp_audio_path
    
    # Merge audio with video
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_for_merge}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)
    
    # Remove temp files
    if os.path.exists(SDK.tmp_output_path):
        os.remove(SDK.tmp_output_path)
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)


def get_test_clips(test_dir):
    """테스트 클립 목록 가져오기"""
    clips = []
    for clip_name in sorted(os.listdir(test_dir)):
        clip_path = os.path.join(test_dir, clip_name)
        if os.path.isdir(clip_path):
            audio_path = os.path.join(clip_path, "audio.wav")
            image_path = os.path.join(clip_path, "image.png")
            if os.path.exists(audio_path) and os.path.exists(image_path):
                clips.append({
                    "name": clip_name,
                    "audio_path": audio_path,
                    "source_path": image_path,
                })
    return clips


def main():
    parser = argparse.ArgumentParser(description="Test set inference for metrics")
    # parser.add_argument("--data_root", type=str, 
    #                     default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch",
    #                     help="path to model data_root")
    # parser.add_argument("--cfg_pkl", type=str, 
    #                     default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
                        # help="path to cfg_pkl")
    parser.add_argument("--data_root", type=str, 
                        default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_trt_Ampere_Plus",
                        help="path to model data_root")
    parser.add_argument("--cfg_pkl", type=str, 
                        default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
                        help="path to cfg_pkl")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/experiments/ditto_original_posebranch_freeze_Lv75Ls05_spyrt_20260506_183322/weights/train_20.pt",
    # parser.add_argument("--checkpoint_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch/models/lmdm_v0.4_hubert.pth", # ditto original pytorch model
                        help="path to trained checkpoint (optional, uses original if not specified)")
    # parser.add_argument("--checkpoint_path", type=str, default=None,
    #                      help="path to trained checkpoint (optional, uses original if not specified)")
    parser.add_argument("--use_meanflow", action="store_true", default=False,
                        help="use MeanFlow sampling instead of DDIM diffusion")

    # pose branch: keep metric inference aligned with pose-branch training checkpoints.
    parser.add_argument("--use_pose_branch", action="store_true", default=True,
                        help="enable pose branch decoder structure for pose-branch checkpoints")
    parser.add_argument("--pose_branch_hidden_dim", type=int, default=128,
                        help="pose branch hidden dimension")
    parser.add_argument("--pose_branch_dropout", type=float, default=0.0,
                        help="pose branch dropout")
    parser.add_argument("--pose_branch_residual_scale", type=float, default=1.0,
                        help="pose branch residual scale")
    parser.add_argument("--pose_branch_gate_bias", type=float, default=-2.0,
                        help="pose branch gate bias")

    parser.add_argument("--test_dir", type=str,
                        default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/testset",         #/workspace/ditto/datasets/Talk8/SUBSET_Talk8/testset
                        help="path to test set directory")
    parser.add_argument("--results_dir", type=str,
                        default="/workspace/ditto/ditto-talkinghead-train/dynpose_talk8/posebranch_freeze_Lv75Ls05_spyrt",         # /workspace/ditto/ditto-talkinghead-train/testset_results/imf_100epoch
                        help="path to results directory")
    parser.add_argument("--more_kwargs", type=str, default=None,
                        help="optional pkl path containing setup_kwargs/run_kwargs for retargeting and control")
    parser.add_argument("--seed", type=int, default=1024,
                        help="random seed")
    parser.add_argument("--max_seconds", type=float, default=10.0,  # MEAD 4.84s
                        help="maximum audio length in seconds (default: 10.0, use 0 for full length)")
    args = parser.parse_args()
    
    # max_seconds가 0이면 None으로 변환 (전체 길이 사용)
    if args.max_seconds <= 0:
        args.max_seconds = None
    
    # Set seed
    seed_everything(args.seed)
    
    
    # Get model name for output folder
    # model_name = get_model_name(args.checkpoint_path)
    # if args.use_meanflow:
    #     model_name += "_meanflow"
    
    # # Create output directory
    # output_dir = os.path.join(args.results_dir, model_name)
    
    # Create output directory (results_dir에 바로 저장)
    output_dir = args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Inference for Metrics")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_path or 'original'}")
    print(f"MeanFlow: {args.use_meanflow}")
    print(f"Pose branch: {args.use_pose_branch}")
    print(f"Max seconds: {args.max_seconds if args.max_seconds else 'Full length'}")
    print(f"Test dir: {args.test_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)
    
    # Get test clips
    test_clips = get_test_clips(args.test_dir)
    print(f"\nFound {len(test_clips)} test clips")
    
    if len(test_clips) == 0:
        print("[ERROR] No test clips found!")
        return
    
    # Initialize SDK
    print("\nInitializing SDK...")
    # pose branch: forward inference-time architecture flags so pose-branch
    # checkpoints can be rendered with the same decoder structure.
    pose_branch_kwargs = {
        "use_pose_branch": args.use_pose_branch,
        "pose_branch_hidden_dim": args.pose_branch_hidden_dim,
        "pose_branch_dropout": args.pose_branch_dropout,
        "pose_branch_residual_scale": args.pose_branch_residual_scale,
        "pose_branch_gate_bias": args.pose_branch_gate_bias,
    }
    if args.checkpoint_path:
        SDK = StreamSDK(
            args.cfg_pkl, 
            args.data_root, 
            checkpoint_path=args.checkpoint_path,
            use_meanflow=args.use_meanflow,
            **pose_branch_kwargs,
        )
    else:
        SDK = StreamSDK(
            args.cfg_pkl, 
            args.data_root,
            use_meanflow=args.use_meanflow,
            **pose_branch_kwargs,
        )
    
    # Run inference for each clip
    print("\nRunning inference...")
    success_count = 0
    failed_clips = []
    
    for clip in tqdm(test_clips, desc="Generating videos"):
        output_path = os.path.join(output_dir, f"{clip['name']}.mp4")
        
        try:
            run_single(
                SDK,
                audio_path=clip["audio_path"],
                source_path=clip["source_path"],
                output_path=output_path,
                max_seconds=args.max_seconds,
                more_kwargs=args.more_kwargs or {},
            )
            success_count += 1
        except Exception as e:
            print(f"\n[ERROR] {clip['name']}: {e}")
            failed_clips.append(clip['name'])
    
    # Summary
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)
    print(f"Success: {success_count}/{len(test_clips)}")
    print(f"Output: {output_dir}")
    
    if failed_clips:
        print(f"\nFailed clips ({len(failed_clips)}):")
        for clip in failed_clips:
            print(f"  - {clip}")
    
    # List generated files
    print(f"\nGenerated files:")
    for f in sorted(os.listdir(output_dir))[:5]:
        fpath = os.path.join(output_dir, f)
        fsize = os.path.getsize(fpath) / (1024 * 1024)  # MB
        print(f"  {f} ({fsize:.2f} MB)")
    if len(os.listdir(output_dir)) > 5:
        print(f"  ... and {len(os.listdir(output_dir)) - 5} more files")


if __name__ == "__main__":
    main()
