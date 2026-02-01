import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import time

from stream_pipeline_offline import StreamSDK


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


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):
    # 전체 추론 시간 측정 시작
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_start = time.perf_counter()

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source_path, output_path, **setup_kwargs)

    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    SDK.close()
    
    # Inference 완료까지 대기 (CUDA 동기화)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - inference_start

    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)

    print(output_path)
    
    # SDK의 timing_stats에서 시간 정보 가져오기
    timing_stats = getattr(SDK, "timing_stats", {}) if hasattr(SDK, "timing_stats") else {}
    
    # Motion DiT 시간
    dit_total_ms = timing_stats.get("dit_total_ms", None)
    motion_dit_time = (dit_total_ms / 1000.0) if dit_total_ms is not None else 0.0
    
    # Face Rendering 시간 (실제 경과 시간 추정)
    # Inference 시간에서 Motion DiT 시간을 빼면 Rendering 시간의 근사치
    rendering_time = inference_time - motion_dit_time if motion_dit_time > 0 else inference_time
    
    # Model type 확인
    use_meanflow_flag = getattr(SDK, 'use_meanflow', None)
    if use_meanflow_flag is None and hasattr(SDK, 'audio2motion'):
        if hasattr(SDK.audio2motion, 'lmdm'):
            use_meanflow_flag = getattr(SDK.audio2motion.lmdm, 'use_meanflow', False)
    if use_meanflow_flag is None:
        use_meanflow_flag = False
    
    sampling_timesteps_val = getattr(SDK, 'sampling_timesteps', None)
    if sampling_timesteps_val is None and hasattr(SDK, 'audio2motion'):
        sampling_timesteps_val = getattr(SDK.audio2motion, 'sampling_timesteps', None)
    
    # 간단한 요약 출력 (사용자 요청 형식)
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    if use_meanflow_flag:
        print(f"MF (1 step) = motion generation ({motion_dit_time:.3f}s) + rendering ({rendering_time:.3f}s)")
    else:
        steps_str = f"{sampling_timesteps_val} steps" if sampling_timesteps_val else "N steps"
        print(f"Ditto ({steps_str}) = motion generation ({motion_dit_time:.3f}s) + rendering ({rendering_time:.3f}s)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_root", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch", help="path to trt data_root")          # pytorch model
    parser.add_argument("--data_root", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_trt_Ampere_Plus", help="path to trt data_root")    # tensorrt model
    # parser.add_argument("--cfg_pkl", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", help="path to cfg_pkl")          # pytorch model
    parser.add_argument("--cfg_pkl", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", help="path to cfg_pkl")    # ditto tensorrt model
    # parser.add_argument("--cfg_pkl", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_meanflow.pkl", help="path to cfg_pkl")    # meanflow tensorrt model
    # parser.add_argument("--checkpoint_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/experiments/ditto_original_hdtf_20251221_234237/weights/train_99.pt", help="path to trained checkpoint (overrides pkl model_path)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to trained checkpoint (overrides pkl model_path)")



    parser.add_argument("--audio_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/example/audio.wav")
    parser.add_argument("--source_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/example/image.png")
    # parser.add_argument("--audio_path", type=str, default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/audio/Shaheen_10s.wav")
    # parser.add_argument("--source_path", type=str, default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/ref/Shaheen.png")
    parser.add_argument("--output_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/ditto_original_output/ditto15s_trt.mp4") # iMF meanflow original / mf15s_trt imf15s_trt ditto15s_trt
    args = parser.parse_args()

    # init sdk
    data_root = args.data_root   # model dir
    cfg_pkl = args.cfg_pkl     # cfg pkl
    # SDK = StreamSDK(cfg_pkl, data_root)
    # checkpoint_path가 지정되면 새로 학습한 가중치로 대체
    use_meanflow = False  # True: MeanFlow (1-step), False: DDIM diffusion
    if args.checkpoint_path:
        SDK = StreamSDK(cfg_pkl, data_root, checkpoint_path=args.checkpoint_path, use_meanflow=use_meanflow)
    else:
        SDK = StreamSDK(cfg_pkl, data_root, use_meanflow=use_meanflow)

    # input args
    audio_path = args.audio_path    # .wav
    source_path = args.source_path   # video|image
    output_path = args.output_path   # .mp4

    # run
    # seed_everything(1024)
    run(SDK, audio_path, source_path, output_path)
