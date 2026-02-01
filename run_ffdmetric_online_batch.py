import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import json
import torch

# 현재 경로를 sys.path에 추가 (metrics 폴더 인식용)
sys.path.append(os.getcwd())
from metrics.ffd import FFDCalculator

def run_online_ffd_batch(args):
    # ---------------------------------------------------------
    # 1. 온라인 SDK 초기화
    # ---------------------------------------------------------
    print("="*60)
    print(f"🚀 Initializing ONLINE SDK for FFD Measurement")
    print(f"   Meanflow: {args.use_meanflow}")
    print("="*60)
    
    try:
        # [핵심] 온라인 파이프라인 강제 로드
        from stream_pipeline_online import StreamSDK
        print("✅ Loaded 'stream_pipeline_online.py'")
        
        # SDK 초기화
        sdk = StreamSDK(
            args.cfg_pkl, 
            args.data_root, 
            checkpoint_path=args.checkpoint_path, 
            use_meanflow=args.use_meanflow
        )
        
        # FFD 계산기가 온라인 로직(첫 청크 측정)을 타도록 플래그 설정
        sdk.online_mode = True
        print("✅ SDK Initialized in ONLINE mode.")
        
    except ImportError:
        print("❌ Error: 'stream_pipeline_online.py' not found.")
        return
    except Exception as e:
        print(f"❌ Error initializing SDK: {e}")
        return

    # ---------------------------------------------------------
    # 2. 테스트셋 로드
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
    print(f"📂 Found {total_clips} clips in {args.test_dir}")
    
    # ---------------------------------------------------------
    # 3. FFD 측정 시작 (RTF 제외)
    # ---------------------------------------------------------
    ffd_calc = FFDCalculator()
    all_ffd = []
    detailed_logs = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n⚡ Starting Online Latency (FFD) Measurement...")
    
    for i, clip in enumerate(tqdm(clips)):
        temp_output = os.path.join(args.output_dir, f"temp_ffd_{clip['name']}.mp4")
        
        clip_result = {
            "name": clip['name'],
            "ffd_ms": None,
            "status": "failed"
        }
        
        try:
            # FFD 측정 (num_runs=1)
            # 온라인 모드이므로 '첫 번째 오디오 청크'가 처리되는 시간만 측정함
            ffd_res = ffd_calc.calculate_with_sdk(
                sdk, clip['audio'], clip['source'], temp_output,
                num_runs=1, warmup_runs=0
            )
            
            latency = ffd_res['ffd_mean_ms']
            all_ffd.append(latency)
            
            clip_result["ffd_ms"] = latency
            clip_result["status"] = "success"
            
        except Exception as e:
            # 에러 발생 시 로그만 남기고 계속 진행
            # tqdm.write(f"  ❌ Error {clip['name']}: {e}")
            clip_result["error"] = str(e)
        finally:
            if os.path.exists(temp_output):
                try: os.remove(temp_output)
                except: pass

        detailed_logs.append(clip_result)

    # ---------------------------------------------------------
    # 4. 결과 리포트
    # ---------------------------------------------------------
    if not all_ffd:
        print("No successful measurements.")
        return

    mean_ffd = np.mean(all_ffd)
    std_ffd = np.std(all_ffd)
    min_ffd = np.min(all_ffd)
    max_ffd = np.max(all_ffd)

    # JSON 저장
    performance_log = {
        "config": {
            "mode": "online_only",
            "use_meanflow": args.use_meanflow,
            "total_samples": total_clips
        },
        "summary": {
            "ffd_mean_ms": float(mean_ffd),
            "ffd_std_ms": float(std_ffd),
            "ffd_min_ms": float(min_ffd),
            "ffd_max_ms": float(max_ffd)
        },
        "details": detailed_logs
    }

    mf_tag = "meanflow" if args.use_meanflow else "baseline"
    save_path = os.path.join(args.output_dir, f"ffd_online_{mf_tag}.json")
    
    with open(save_path, "w") as f:
        json.dump(performance_log, f, indent=4)

    print("\n" + "="*60)
    print("🚀 ONLINE LATENCY (FFD) REPORT")
    print("="*60)
    print(f"Meanflow: {args.use_meanflow}")
    print(f"Samples : {len(all_ffd)} / {total_clips}")
    print("-" * 60)
    print(f"⏱️  Avg FFD : {mean_ffd:.2f} ms (± {std_ffd:.2f})")
    print(f"⚡ Min FFD : {min_ffd:.2f} ms")
    print(f"🐌 Max FFD : {max_ffd:.2f} ms")
    print("-" * 60)
    
    # Latency 등급 판정
    if mean_ffd < 300:
        print("✅ Status: Ultra Low Latency (Great for Live)")
    elif mean_ffd < 500:
        print("✅ Status: Low Latency (Good for Chat)")
    elif mean_ffd < 1000:
        print("⚠️ Status: Medium Latency (Noticeable Delay)")
    else:
        print("❌ Status: High Latency (Laggy)")
        
    print(f"\n📝 Detailed log saved to: {save_path}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFD Online Batch Measurement")
    
    # 필수 경로 인자
    parser.add_argument("--data_root", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_pytorch", help="Path to model directory")
    parser.add_argument("--cfg_pkl", type=str, default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", help="Path to config pickle")
    parser.add_argument("--test_dir", type=str, default="/workspace/ditto/ditto-talkinghead-train/example/testset", help="Directory containing test set folders")
    
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/tmp/ditto_ffd_test")
    parser.add_argument("--use_meanflow", default=False, action="store_true")

    args = parser.parse_args()
    run_online_ffd_batch(args)