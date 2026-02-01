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
    # 전체 영상 생성 시간 측정 시작
    total_start = time.perf_counter()
    
    # CUDA 동기화 (정확한 시간 측정을 위해)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    # Setup 시간 측정
    setup_start = time.perf_counter() #
    SDK.setup(source_path, output_path, **setup_kwargs)
    if torch.cuda.is_available(): #
        torch.cuda.synchronize()
    setup_time = time.perf_counter() - setup_start

    audio, sr = librosa.core.load(audio_path, sr=16000)
    audio_duration = len(audio) / sr #
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    # Inference 시간 측정 시작
    inference_start = time.perf_counter()
    
    # Audio2Feat 시간 측정 (HuBERT 특징 추출)
    audio2feat_start = time.perf_counter()
    
    # GPU 메모리 상태 확인 (디버깅용)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_before_a2f = torch.cuda.memory_allocated(0) / 1024**3  # GB
    else:
        gpu_mem_before_a2f = None

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
        # Offline 모드: Audio2Feat 시간 직접 측정
        if torch.cuda.is_available(): #
            torch.cuda.synchronize()
        aud_feat = SDK.wav2feat.wav2feat(audio)
        if torch.cuda.is_available(): #
            torch.cuda.synchronize()
        audio2feat_time = time.perf_counter() - audio2feat_start  # 초 단위
        
        SDK.audio2motion_queue.put(aud_feat)
    
    # GPU 메모리 상태 확인 (디버깅용)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_after_a2f = torch.cuda.memory_allocated(0) / 1024**3  # GB
    else:
        gpu_mem_after_a2f = None
    SDK.close()
    
    # Inference 완료까지 대기 (CUDA 동기화)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - inference_start
    
    # SDK의 timing_stats에서 상세 시간 정보 가져오기
    timing_stats = getattr(SDK, "timing_stats", {}) if hasattr(SDK, "timing_stats") else {}
    
    # Audio2Feat 시간 (offline에서는 직접 측정, online에서는 timing_stats 사용)
    if online_mode:
        audio2feat_total_ms = timing_stats.get("audio2feat_total_ms", 0.0)
        audio2feat_time = audio2feat_total_ms / 1000.0  # ms -> s
    else:
        # offline 모드에서는 이미 측정됨
        pass
    
    # Motion DiT 시간
    dit_total_ms = timing_stats.get("dit_total_ms", None)
    motion_dit_time = (dit_total_ms / 1000.0) if dit_total_ms is not None else None
    
    # Face Rendering 시간 (Warp + Decode + Stitch + Putback + Writer)
    # 주의: 이들은 파이프라인으로 병렬 처리되므로 합계는 실제 경과 시간보다 큼
    warp_total_ms = timing_stats.get("warp_total_ms", 0.0) or 0.0
    decode_total_ms = timing_stats.get("decode_total_ms", 0.0) or 0.0
    stitch_total_ms = timing_stats.get("stitch_total_ms", 0.0) or 0.0
    putback_total_ms = timing_stats.get("putback_total_ms", 0.0) or 0.0
    writer_total_ms = timing_stats.get("writer_total_ms", 0.0) or 0.0
    
    # Face Rendering 실제 경과 시간은 Inference Time - Audio2Feat - DiT 시간
    # (단, 일부 오버랩이 있을 수 있음)
    if motion_dit_time is not None:
        face_rendering_wallclock_time = inference_time - audio2feat_time - motion_dit_time
    else:
        face_rendering_wallclock_time = inference_time - audio2feat_time
    
    # 각 모듈별 순수 작업 시간 합계 (병렬 처리 고려 X, 참고용)
    face_rendering_sum_time = (warp_total_ms + decode_total_ms + stitch_total_ms + putback_total_ms + writer_total_ms) / 1000.0

    # ffmpeg muxing 시간 측정
    mux_start = time.perf_counter()
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)
    mux_time = time.perf_counter() - mux_start

    # 전체 시간 측정 종료
    total_time = time.perf_counter() - total_start

    # print(output_path)

    # 시간 출력
    print("\n" + "=" * 80)
    print("⏱️  TIMING BREAKDOWN")
    print("=" * 80)
    print(f"  Setup Time (SDK 초기화 시간)                                  : {setup_time:.3f}s")
    print()
    print("  📊 Inference 단계별 시간:")
    print(f"    🎵 Audio2Feat (HuBERT 특징 추출)                            : {audio2feat_time:.3f}s")
    # print(f"      └─ Audio length: {audio_duration:.3f}s ({len(audio)} samples @ 16kHz)")
    # if gpu_mem_before_a2f is not None:
    #     print(f"      └─ GPU mem before: {gpu_mem_before_a2f:.2f}GB | after: {gpu_mem_after_a2f:.2f}GB")
    if motion_dit_time is not None:
        print(f"    🧠 Motion DiT (모션 생성)                                  : {motion_dit_time:.3f}s")
        # 각 클립 처리 시간 분석
        dit_per_chunk_ms = timing_stats.get('dit_per_chunk_ms', [])
        if dit_per_chunk_ms:
            dit_per_chunk_ms_array = np.array(dit_per_chunk_ms)
            # sampling_timesteps 가져오기
            sampling_timesteps = getattr(SDK, 'sampling_timesteps', None)
            if sampling_timesteps is None and hasattr(SDK, 'audio2motion'):
                sampling_timesteps = getattr(SDK.audio2motion, 'sampling_timesteps', None)
            
            # MeanFlow인지 확인 (use_meanflow 확인)
            use_meanflow = getattr(SDK, 'use_meanflow', None)
            if use_meanflow is None and hasattr(SDK, 'audio2motion'):
                if hasattr(SDK.audio2motion, 'lmdm'):
                    use_meanflow = getattr(SDK.audio2motion.lmdm, 'use_meanflow', False)
            
            # use_meanflow가 여전히 None이면 False로 설정 (안전장치)
            if use_meanflow is None:
                use_meanflow = False
            
            print(f"      ├─ 클립 수                                               : {len(dit_per_chunk_ms)}개")
            print(f"      ├─ 평균 클립 처리 시간                                   : {dit_per_chunk_ms_array.mean():.2f}ms/clip")
            # print(f"      ├─ 최소 클립 처리 시간                                   : {dit_per_chunk_ms_array.min():.2f}ms/clip")
            # print(f"      ├─ 최대 클립 처리 시간                                   : {dit_per_chunk_ms_array.max():.2f}ms/clip")
            
            # MeanFlow는 실제로 1-step만 수행하므로, 클립 시간 = 1-step 시간
            # Ditto는 여러 step을 수행하므로, 클립 시간을 sampling_timesteps로 나누어야 함
            if use_meanflow:
                # MeanFlow: 1-step이므로 클립 시간 그대로 사용
                avg_clip_time_ms = dit_per_chunk_ms_array.mean()
                print(f"      ├─ Model type                                          : MeanFlow (1-step)")
                print(f"      └─ 평균 1 step 시간 (MeanFlow는 1-step만 수행)          : {avg_clip_time_ms:.2f}ms/step")
            elif sampling_timesteps is not None and sampling_timesteps > 1:
                # Ditto (Diffusion): 클립 시간을 sampling_timesteps로 나누어 1 step 시간 계산
                avg_clip_time_ms = dit_per_chunk_ms_array.mean()
                avg_step_time_ms = avg_clip_time_ms / sampling_timesteps
                print(f"      ├─ Model type                                          : Ditto (Diffusion)")
                print(f"      ├─ Sampling timesteps                                  : {sampling_timesteps} step")
                print(f"      └─ 평균 1 diffusion step 시간                          : {avg_step_time_ms:.2f}ms/step")
            else:
                # sampling_timesteps가 None이거나 1인 경우 (예외 상황)
                avg_clip_time_ms = dit_per_chunk_ms_array.mean()
                print(f"      ├─ Model type                                          : Unknown")
                print(f"      └─ 평균 클립 처리 시간 (step 정보 없음)                 : {avg_clip_time_ms:.2f}ms/clip")
            # print(f"      └─ 각 클립 시간 상세                                     : {[f'{t:.2f}ms' for t in dit_per_chunk_ms]}")
    else:
        print(f"    🧠 Motion DiT (모션 생성)                                  : N/A")
    print(f"    🎨 Face Rendering (실제 경과 시간)                          : {face_rendering_wallclock_time:.3f}s")
    
    if motion_dit_time is not None:
        print(f"      ├─ Warp     : {warp_total_ms/1000.0:.3f}s")
        print(f"      ├─ Decode   : {decode_total_ms/1000.0:.3f}s")
        print(f"      ├─ Stitch   : {stitch_total_ms/1000.0:.3f}s")
        print(f"      ├─ Putback  : {putback_total_ms/1000.0:.3f}s")
        print(f"      └─ Writer   : {writer_total_ms/1000.0:.3f}s (순수 작업 시간, 진행 표시줄은 대기 시간 포함)")        
    print(f"    📝 Inference Total                                            : {inference_time:.3f}s")
    print()
    print(f"  🔧 Muxing Time (ffmpeg 오디오/비디오 합성)                    : {mux_time:.3f}s")
    print("-" * 80)
    print(f"  ⏱️  Total Time                                                : {total_time:.3f}s")
    print(f"  🎵 Audio Duration                                            : {audio_duration:.3f}s")
    print(f"  📊 RTF (Real-Time Factor)                                     : {total_time / audio_duration:.4f}")
    if total_time / audio_duration < 1.0:
        print(f"  ✅ Status                                                     : Real-time capable ({1.0 / (total_time / audio_duration):.2f}x speed)")
    else:
        print(f"  ❌ Status                                                     : Not real-time ({audio_duration / total_time:.2f}x speed)")
    print("=" * 80)
    print(f"Output: {output_path}")
    
    # 간단한 요약 출력 (사용자 요청 형식)
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    
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
    
    # Motion Generation 시간 (Motion DiT 시간)
    motion_gen_time = motion_dit_time if motion_dit_time is not None else 0.0
    
    # Rendering 시간 (Face Rendering 실제 경과 시간)
    rendering_time = face_rendering_wallclock_time
    
    # 출력 형식
    if use_meanflow_flag:
        print(f"MF (1 step) = motion generation ({motion_gen_time:.3f}s) + rendering ({rendering_time:.3f}s)")
    else:
        steps_str = f"{sampling_timesteps_val} steps" if sampling_timesteps_val else "N steps"
        print(f"Ditto ({steps_str}) = motion generation ({motion_gen_time:.3f}s) + rendering ({rendering_time:.3f}s)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_root", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch", help="path to trt data_root")          # pytorch model
    parser.add_argument("--data_root", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_trt_Ampere_Plus", help="path to trt data_root")    # tensorrt model

    # parser.add_argument("--cfg_pkl", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", help="path to cfg_pkl")          # pytorch model
    parser.add_argument("--cfg_pkl", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", help="path to cfg_pkl")    # ditto tensorrt model
    # parser.add_argument("--cfg_pkl", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_meanflow.pkl", help="path to cfg_pkl")    # meanflow tensorrt model
    # parser.add_argument("--cfg_pkl", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_iMF.pkl", help="path to cfg_pkl")    # meanflow tensorrt model


    # parser.add_argument("--checkpoint_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/experiments/ditto_meanflow_hdtf_20251229_225128/samples/train_100.pt", help="path to trained checkpoint (overrides pkl model_path)")
    # parser.add_argument("--checkpoint_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/experiments/ditto_original_hdtf_20251221_234237/weights/train_99.pt", help="path to trained checkpoint (overrides pkl model_path)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to trained checkpoint (overrides pkl model_path)")  # tensorrt 사용시 None


    parser.add_argument("--audio_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/example/audio.wav")
    parser.add_argument("--source_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/example/image.png")
    # parser.add_argument("--audio_path", type=str, default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/audio/obama_10s.wav")
    # parser.add_argument("--source_path", type=str, default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/ref/obama.png")
    parser.add_argument("--output_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/sample_output/ditto50_15s_trt.mp4") # tensorrt/ ditto_meanflow_output/ditto_original_output
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
