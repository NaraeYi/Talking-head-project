import argparse
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import json
import math
import librosa

# 경로 설정: metrics 폴더 및 현재 폴더 인식
sys.path.append(os.getcwd())
from metrics.rtf import RTFCalculator
from metrics.ffd import FFDCalculator


def run_batch_performance(args):
    # ---------------------------------------------------------
    # 1. SDK 초기화 (모델 로딩은 여기서 딱 한 번만 수행,모드에 따라 online/offline 선택)
    # ---------------------------------------------------------
    print("="*60)
    print(f"🚀 Initializing StreamSDK")
    print(f"   Mode: {args.mode.upper()}")
    print(f"   Meanflow: {args.use_meanflow}")
    print(f"   Save Videos: {args.save_videos}")
    print("="*60)
    
    try:
        if args.mode == "online":
            from stream_pipeline_online import StreamSDK
            print("✅ Loaded 'stream_pipeline_online.py' (ONLINE mode)")
        else:
            from stream_pipeline_offline import StreamSDK
            print("✅ Loaded 'stream_pipeline_offline.py' (OFFLINE mode)")
        
        sdk = StreamSDK(
            args.cfg_pkl, 
            args.data_root, 
            checkpoint_path=args.checkpoint_path, 
            use_meanflow=args.use_meanflow
        )
        
        # 온라인 모드 플래그 설정
        if args.mode == "online":
            sdk.online_mode = True
            
        print("✅ SDK Initialized successfully.")
        
    except ImportError as e:
        print(f"Error: Pipeline module not found. {e}")
        return
    except Exception as e:
        print(f"Error initializing SDK: {e}")
        return

    # ---------------------------------------------------------
    # 2. 테스트셋 파일 목록 가져오기
    # ---------------------------------------------------------
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        return

    clips = []
    for clip_name in sorted(os.listdir(args.test_dir)):
        clip_path = os.path.join(args.test_dir, clip_name)
        
        if os.path.isdir(clip_path):
            audio_path = os.path.join(clip_path, "audio.wav")
            image_path = os.path.join(clip_path, "image.png")
            if not os.path.exists(image_path):
                image_path = os.path.join(clip_path, "image.jpg")
            
            if os.path.exists(audio_path) and os.path.exists(image_path):
                clips.append({
                    "name": clip_name,
                    "audio": audio_path,
                    "source": image_path
                })
    
    total_clips = len(clips)
    print(f"📂 Found {total_clips} valid test clips in {args.test_dir}")
    
    if total_clips == 0:
        print("No clips found. Check your directory structure.")
        return

    # ---------------------------------------------------------
    # 3. 결과 저장 폴더 생성
    # ---------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 비디오 저장 폴더 (save_videos가 True일 때)
    if args.save_videos:
        video_save_dir = args.results_dir
        os.makedirs(video_save_dir, exist_ok=True)
        print(f"📁 Videos will be saved to: {video_save_dir}")

    # ---------------------------------------------------------
    # 4. 측정 시작 (Loop)
    # ---------------------------------------------------------
    rtf_calc = RTFCalculator()
    ffd_calc = FFDCalculator()
    
    all_rtf = []
    all_ffd = []
    
    # 상세 로그를 저장할 리스트
    detailed_logs = []
    
    print("\n⚡ Starting Batch Performance Measurement...")
    
    # Tqdm 사용 (desc 제거하여 출력 깔끔하게)
    for i, clip in enumerate(tqdm(clips)):
        # [요청하신 기능] 현재 진행률 명시적 출력
        # 예: [ 1 / 40 ] Processing: RD_Radio12_000 ...
        current_idx = i + 1
        tqdm.write(f"\n[{current_idx:02d} / {total_clips:02d}] Processing: {clip['name']} ...")
        
        # 임시 출력 파일 (메트릭 측정용)
        temp_output = os.path.join(args.output_dir, f"temp_perf_{clip['name']}.mp4")
        
        # 최종 저장 파일 (save_videos가 True일 때)
        if args.save_videos:
            final_output = os.path.join(video_save_dir, f"{clip['name']}.mp4")
        
        clip_result = {
            "name": clip['name'],
            "rtf": None,
            "ffd_ms": None,
            "dit_total_ms": None,    # total time for audio2motion
            "dit_avg_chunk_ms": None,
            "writer_total_ms": None,  # total time for writer
            "writer_avg_frame_ms": None,  # per frame time for writer
            "warp_total_ms": None,  # total time for warp
            "decode_total_ms": None,  # total time for decode
            "stitch_total_ms": None,  # total time for stitch
            "putback_total_ms": None,  # total time for putback
            "status": "failed"
        }
        
        try:
            # === RTF 측정 ===
            # 첫 번째 파일일 때만 Warmup 수행
            current_warmup = 1 if i == 0 else 0
            
            rtf_res = rtf_calc.calculate_with_sdk(
                sdk, clip['audio'], clip['source'], temp_output,
                num_runs=1, warmup_runs=current_warmup,
                max_seconds=args.max_seconds
            )
            all_rtf.append(rtf_res['rtf_mean'])
            clip_result["rtf"] = rtf_res['rtf_mean']
            
            # === 타이밍 정보 수집 (RTF 측정 후) ===
            if hasattr(sdk, 'timing_stats'):
                ts = sdk.timing_stats
                clip_result["dit_total_ms"] = ts.get('dit_total_ms', 0)
                clip_result["writer_total_ms"] = ts.get('writer_total_ms', 0)
                clip_result["warp_total_ms"] = ts.get('warp_total_ms', 0)
                clip_result["decode_total_ms"] = ts.get('decode_total_ms', 0)
                clip_result["stitch_total_ms"] = ts.get('stitch_total_ms', 0)
                clip_result["putback_total_ms"] = ts.get('putback_total_ms', 0)
                
                # 평균 계산
                dit_chunks = ts.get('dit_per_chunk_ms', [])
                writer_frames = ts.get('writer_per_frame_ms', [])
                clip_result["dit_avg_chunk_ms"] = np.mean(dit_chunks) if dit_chunks else 0
                clip_result["writer_avg_frame_ms"] = np.mean(writer_frames) if writer_frames else 0

            # === FFD 측정 ===
            ffd_res = ffd_calc.calculate_with_sdk(
                sdk, clip['audio'], clip['source'], temp_output,
                num_runs=1, warmup_runs=0,
                max_seconds=args.max_seconds
            )
            all_ffd.append(ffd_res['ffd_mean_ms'])
            clip_result["ffd_ms"] = ffd_res['ffd_mean_ms']
            
            clip_result["status"] = "success"
            
            # 타이밍 정보 출력 (상세)
            tqdm.write(f"  ⏱️  RTF: {clip_result['rtf']:.4f} | FFD: {clip_result['ffd_ms']:.1f}ms")
            tqdm.write(f"      🧠 DiT: {clip_result['dit_total_ms']:.1f}ms (chunk avg: {clip_result['dit_avg_chunk_ms']:.1f}ms)")
            tqdm.write(f"      🔄 Warp: {clip_result['warp_total_ms']:.1f}ms | 🎨 Decode: {clip_result['decode_total_ms']:.1f}ms")
            tqdm.write(f"      🧵 Stitch: {clip_result['stitch_total_ms']:.1f}ms | 📦 Putback: {clip_result['putback_total_ms']:.1f}ms")
            tqdm.write(f"      📝 Writer: {clip_result['writer_total_ms']:.1f}ms (frame avg: {clip_result['writer_avg_frame_ms']:.2f}ms)")
            
            # === 비디오 저장 (오디오 합성 포함) ===
            if args.save_videos and os.path.exists(temp_output):
                # 오디오 길이 제한 적용
                audio_path = clip['audio']
                if args.max_seconds and args.max_seconds > 0:
                    # 임시 오디오 파일 생성 (max_seconds 길이로 자름)
                    audio, sr = librosa.core.load(audio_path, sr=16000)
                    max_samples = int(args.max_seconds * 16000)
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]
                        temp_audio = os.path.join(args.output_dir, f"temp_audio_{clip['name']}.wav")
                        import soundfile as sf
                        sf.write(temp_audio, audio, 16000)
                        audio_path = temp_audio
                
                # 오디오와 비디오 합성
                cmd = f'ffmpeg -loglevel error -y -i "{temp_output}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{final_output}"'
                os.system(cmd)
                
                # 임시 오디오 파일 삭제
                if args.max_seconds and args.max_seconds > 0:
                    temp_audio = os.path.join(args.output_dir, f"temp_audio_{clip['name']}.wav")
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
                
                tqdm.write(f"  💾 Saved: {final_output}")
            
        except Exception as e:
            err_msg = str(e)
            tqdm.write(f"  ❌ Error: {err_msg}")
            clip_result["error"] = err_msg
            continue # 에러 나면 다음 파일로 넘어감
        finally:
            # save_videos가 False이거나 저장 완료 후 임시 파일 삭제
            if os.path.exists(temp_output) and not args.save_videos:
                os.remove(temp_output)
            elif os.path.exists(temp_output) and args.save_videos:
                # 저장 후 임시 파일 삭제
                os.remove(temp_output)
        
        # 상세 로그에 추가
        detailed_logs.append(clip_result)

    # ---------------------------------------------------------
    # 5. 결과 종합 및 저장
    # ---------------------------------------------------------
    if not all_rtf:
        print("No successful measurements.")
        return

    # 통계 계산
    mean_rtf = np.mean(all_rtf)
    std_rtf = np.std(all_rtf)
    mean_ffd = np.mean(all_ffd)
    std_ffd = np.std(all_ffd)
    min_ffd = np.min(all_ffd)
    max_ffd = np.max(all_ffd)
    
    # 타이밍 통계 계산
    all_dit_total = [d["dit_total_ms"] for d in detailed_logs if d.get("dit_total_ms")]
    all_dit_avg = [d["dit_avg_chunk_ms"] for d in detailed_logs if d.get("dit_avg_chunk_ms")]
    all_writer_total = [d["writer_total_ms"] for d in detailed_logs if d.get("writer_total_ms")]
    all_writer_avg = [d["writer_avg_frame_ms"] for d in detailed_logs if d.get("writer_avg_frame_ms")]
    all_warp = [d["warp_total_ms"] for d in detailed_logs if d.get("warp_total_ms")]
    all_decode = [d["decode_total_ms"] for d in detailed_logs if d.get("decode_total_ms")]
    all_stitch = [d["stitch_total_ms"] for d in detailed_logs if d.get("stitch_total_ms")]
    all_putback = [d["putback_total_ms"] for d in detailed_logs if d.get("putback_total_ms")]

    # JSON 로그 구조 생성
    mf_tag = "meanflow" if args.use_meanflow else "baseline"
    performance_log = {
        "config": {
            "mode": args.mode,
            "model_path": args.data_root,
            "checkpoint_path": args.checkpoint_path,
            "use_meanflow": args.use_meanflow,
            "max_seconds": args.max_seconds,
            "save_videos": args.save_videos,
            "results_dir": args.results_dir if args.save_videos else None,
            "total_samples": total_clips,
            "processed_samples": len(all_rtf)
        },
        "summary": {
            "rtf_mean": float(mean_rtf),
            "rtf_std": float(std_rtf),
            "ffd_mean_ms": float(mean_ffd),
            "ffd_std_ms": float(std_ffd),
            "ffd_min_ms": float(min_ffd),
            "ffd_max_ms": float(max_ffd),
            "timing": {
                "dit_total_mean_ms": float(np.mean(all_dit_total)) if all_dit_total else 0,
                "dit_chunk_mean_ms": float(np.mean(all_dit_avg)) if all_dit_avg else 0,
                "writer_total_mean_ms": float(np.mean(all_writer_total)) if all_writer_total else 0,
                "writer_frame_mean_ms": float(np.mean(all_writer_avg)) if all_writer_avg else 0,
                "warp_mean_ms": float(np.mean(all_warp)) if all_warp else 0,
                "decode_mean_ms": float(np.mean(all_decode)) if all_decode else 0,
                "stitch_mean_ms": float(np.mean(all_stitch)) if all_stitch else 0,
                "putback_mean_ms": float(np.mean(all_putback)) if all_putback else 0,
            }
        },
        "details": detailed_logs  # 개별 파일 기록 포함
    }

    # JSON 파일로 저장
    log_filename = f"performance_{args.mode}_{mf_tag}.json"
    save_path = os.path.join(args.output_dir, log_filename)
    
    with open(save_path, "w") as f:
        json.dump(performance_log, f, indent=4)

    # 화면 출력
    print("\n" + "="*60)
    print("📊 BATCH PERFORMANCE REPORT")
    print("="*60)
    print(f"Mode    : {args.mode.upper()}")
    print(f"Samples : {len(all_rtf)} / {total_clips}")
    print(f"Meanflow: {args.use_meanflow}")
    print(f"Max Sec : {args.max_seconds if args.max_seconds else 'Full length'}")
    print("-" * 60)
    
    print(f"⏱️  Avg RTF : {mean_rtf:.4f} (± {std_rtf:.4f})")
    if mean_rtf < 1.0:
        print(f"   Status: ✅ Real-time Capable ({1/mean_rtf:.2f}x speed)")
    else:
        print(f"   Status: ⚠️  Slower than Real-time")

    print(f"🚀 Avg FFD : {mean_ffd:.2f} ms (± {std_ffd:.2f})")
    print(f"⚡ Min FFD : {min_ffd:.2f} ms")
    print(f"🐌 Max FFD : {max_ffd:.2f} ms")
    
    # 타이밍 상세 출력
    print("-" * 60)
    print("📌 TIMING BREAKDOWN (Average per video):")
    if all_dit_total:
        print(f"   🧠 DiT (Diffusion) Total : {np.mean(all_dit_total):.2f} ms")
        print(f"      └─ Per Chunk Avg     : {np.mean(all_dit_avg):.2f} ms")
    if all_writer_total:
        print(f"   📝 Writer Total          : {np.mean(all_writer_total):.2f} ms")
        print(f"      └─ Per Frame Avg     : {np.mean(all_writer_avg):.2f} ms")
    if all_warp:
        print(f"   🔄 Warp Total            : {np.mean(all_warp):.2f} ms")
    if all_decode:
        print(f"   🎨 Decode Total          : {np.mean(all_decode):.2f} ms")
    if all_stitch:
        print(f"   🧵 Stitch Total          : {np.mean(all_stitch):.2f} ms")
    if all_putback:
        print(f"   📦 Putback Total         : {np.mean(all_putback):.2f} ms")
    
    # Latency 등급 판정 (online 모드일 때만)
    if args.mode == "online":
        print("-" * 60)
        if mean_ffd < 300:
            print("✅ Latency Status: Ultra Low (Great for Live)")
        elif mean_ffd < 500:
            print("✅ Latency Status: Low (Good for Chat)")
        elif mean_ffd < 1000:
            print("⚠️ Latency Status: Medium (Noticeable Delay)")
        else:
            print("❌ Latency Status: High (Laggy)")
    
    print("-" * 60)
    print(f"📝 Log saved to: {save_path}")
    
    if args.save_videos:
        video_count = len([f for f in os.listdir(video_save_dir) if f.endswith('.mp4')])
        print(f"🎬 Videos saved to: {video_save_dir} ({video_count} files)")
        print(f"\n💡 Tip: Run quality metrics with:")
        print(f"   python metrics/run_all_metrics.py --batch \\")
        print(f"       --results_dir {video_save_dir} \\")
        print(f"       --real_videos_dir /path/to/ground_truth_videos")
    
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Performance Benchmarking for Ditto (RTF + FFD)")
    
    # 필수 경로 인자
    # parser.add_argument("--data_root", type=str, 
    #                     default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch", 
    #                     help="Path to model directory")
    # parser.add_argument("--cfg_pkl", type=str, 
    #                     default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", 
    #                     help="Path to config pickle")

    parser.add_argument("--data_root", type=str, 
                        default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_trt_Ampere_Plus", 
                        help="Path to model directory")
    parser.add_argument("--cfg_pkl", type=str, 
                        default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", 
                        help="Path to config pickle")


    parser.add_argument("--test_dir", type=str, 
                        default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/testset", 
                        help="Directory containing test set folders")
    
    # parser.add_argument("--checkpoint_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/experiments/ditto_meanflow_hdtf_20251229_225128/samples/train_100.pt", 
                        # help="Specific checkpoint path")  # meanflow 100epoch
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Specific checkpoint path")  # original 500epoch
    parser.add_argument("--output_dir", type=str, 
                        default="/tmp/ditto_perf_test", 
                        help="Directory for metric logs and temp files")
    
    # 모드 선택
    parser.add_argument("--mode", type=str, default="online", choices=["offline", "online"],   # 모드 선택 (offline: 배치 처리, online: 스트리밍/실시간)   
                        help="SDK mode: 'offline' (batch processing) or 'online' (streaming/realtime)")
    
    # Meanflow 사용 여부
    parser.add_argument("--use_meanflow", action="store_true", default=False, 
                        help="Enable meanflow option in SDK (default: True)")
    
    # 오디오 길이 제한
    parser.add_argument("--max_seconds", type=float, default=10.0,               # 오디오 최대 길이 10.0초(0이면 전체)
                        help="Maximum audio length in seconds (default: 10.0, use 0 for full length)")
    
    # 비디오 저장 옵션
    parser.add_argument("--save_videos", action="store_true", default=False,    # 생성된 영상 저장 여부
                        help="Save generated videos for quality metrics")
    parser.add_argument("--results_dir", type=str,                              # 영상 저장 경로
                        default="/workspace/ditto/ditto-talkinghead-train/testset_results/realtime_metric_original10steps_tensorRT_Talk8", # realtime_metric_original10steps_tensorRT_Talk8, realtime_metric_meanflow_100epoch_tensorRT_Talk8
                        help="Directory to save generated videos (only used with --save_videos)")

    args = parser.parse_args()
    
    # max_seconds가 0이면 None으로 변환
    if args.max_seconds <= 0:
        args.max_seconds = None
    
    run_batch_performance(args)
