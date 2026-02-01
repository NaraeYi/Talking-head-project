#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ditto Batch Performance Benchmarking (RTF + FFD + Module breakdown)

Key points (paper-aligned):
- Ditto paper Table 1/5 describes RTF/FFD for *streaming inference* (video stream output).
- OFFLINE batch mode has a different definition (chunk overlap/stitching/output pipeline), so:
  - We still report OFFLINE batch RTF, but we explicitly warn it's not directly comparable.
  - We hide Table 5-style module breakdown in OFFLINE mode.

RTF definitions used:
- rtf_audio  = elapsed / audio_seconds_used
- rtf_valid  = elapsed / (generated_frames / fps)   (frame-based "valid duration")
- rtf_paper  = elapsed / expected_duration, where expected_duration = round(audio_seconds_used * fps)/fps
             (helps avoid under/over-counting if SDK pads/overlaps internally)

Table 5-style module breakdown (ONLINE only):
- step duration = effective_frames * frame_ms (e.g., 5 frames @25fps => 200ms)
- frame duration = frame_ms (e.g., 40ms @25fps)
- Module RTF:
    Audio2Feat RTF = (ms/step) / step_ms
    Motion DiT RTF = (ms/step) / step_ms
    Face Rendering RTF = (ms/frame) / frame_ms
"""

import os
import sys
import json
import time
import math
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# Optional torch (for CUDA sync if available)
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def cuda_sync_if_needed():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def pretty_ms(x: float) -> str:
    return f"{x:,.2f} ms"


def pretty_sec(x: float) -> str:
    return f"{x:,.3f} s"


def load_audio_duration_seconds(audio_path: str, max_seconds: float) -> float:
    """
    Returns audio duration (seconds) after applying max_seconds truncation.
    Uses soundfile if available; otherwise falls back to ffprobe if available.
    """
    dur = None

    # Try soundfile
    try:
        import soundfile as sf
        with sf.SoundFile(audio_path) as f:
            dur = len(f) / float(f.samplerate)
    except Exception:
        dur = None

    if dur is None:
        # Try ffprobe
        try:
            import subprocess
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()
            dur = float(out)
        except Exception:
            dur = None

    if dur is None:
        # As a last resort, assume max_seconds if given, else 0
        dur = float(max_seconds) if max_seconds and max_seconds > 0 else 0.0

    if max_seconds and max_seconds > 0:
        dur = min(dur, float(max_seconds))
    return float(dur)


def list_test_samples(test_dir: str) -> List[Dict[str, str]]:
    """
    Expects test_dir containing subfolders. Each subfolder should include:
      - audio.wav (or .mp3) and
      - source.mp4 (or reference assets)
    Your project might already have a convention; adjust patterns if needed.
    """
    test_dir = Path(test_dir)
    samples = []

    for sub in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
        # try common file names
        audio = None
        for cand in ["audio.wav", "audio.mp3", "audio.flac", "audio.m4a", "audio.aac"]:
            if (sub / cand).exists():
                audio = str(sub / cand)
                break
        if audio is None:
            # try any audio
            for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.aac"]:
                hits = list(sub.glob(ext))
                if hits:
                    audio = str(hits[0])
                    break

        source = None
        for cand in ["source.mp4", "source.mov", "video.mp4", "input.mp4", "ref.mp4"]:
            if (sub / cand).exists():
                source = str(sub / cand)
                break
        if source is None:
            # any video
            for ext in ["*.mp4", "*.mov", "*.mkv"]:
                hits = list(sub.glob(ext))
                if hits:
                    source = str(hits[0])
                    break

        if audio and source:
            samples.append({
                "name": sub.name,
                "audio": audio,
                "source": source,
                "folder": str(sub),
            })

    return samples


def ffmpeg_mux(video_path: str, audio_path: str, out_path: str) -> None:
    """
    Mux audio into a silent video, if needed.
    """
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


# ---------------------------------------------------------------------
# Core: inference measurement
# ---------------------------------------------------------------------
def run_inference_measurement(
    args,
    sample: Dict[str, str],
    sdk_mode: str,
    effective_frames: int,
) -> Tuple[float, float, int, Dict[str, Any]]:
    """
    Returns:
      elapsed_sec, ffd_ms, generated_frames, timing_stats
    """
    source_path = sample["source"]
    audio_path = sample["audio"]

    # Import StreamSDK depending on mode
    if sdk_mode == "online":
        from stream_pipeline_online import StreamSDK
        print("✅ Loaded 'stream_pipeline_online.py' (ONLINE mode)")
    else:
        from stream_pipeline_offline import StreamSDK
        print("✅ Loaded 'stream_pipeline_offline.py' (OFFLINE mode)")

    # Create output directory
    ensure_dir(args.output_dir)

    temp_output = os.path.join(args.output_dir, f"{sample['name']}_temp.mp4")
    final_output = None
    if args.save_videos:
        ensure_dir(args.results_dir)
        final_output = os.path.join(args.results_dir, f"{sample['name']}.mp4")

    # Load audio seconds (truncated by max_seconds)
    trimmed_audio_duration = load_audio_duration_seconds(audio_path, args.max_seconds)

    # Init SDK
    sdk = StreamSDK(
        data_root=args.data_root,
        cfg_pkl=args.cfg_pkl,
        checkpoint_path=args.checkpoint_path,
        use_meanflow=args.use_meanflow,
    )

    # Setup: pass key configs explicitly (many codebases merge kwargs; we prefer explicit to reduce "not applied" risk)
    sdk.setup(
        source_path=source_path,
        output_path=temp_output,
        online_mode=(sdk_mode == "online"),
        sampling_timesteps=args.sampling_timesteps,
        overlap_v2=args.overlap_v2,
        max_size=args.max_size,
        # NOTE: add other fixed knobs here if you want them forced for reproducibility
    )

    cuda_sync_if_needed()
    t0 = time.perf_counter()

    # Run
    sdk.run(
        source_path=source_path,
        output_path=temp_output,
        audio_path=audio_path,
        max_seconds=args.max_seconds,
    )

    # Close
    sdk.close()

    cuda_sync_if_needed()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # Stats
    timing_stats = getattr(sdk, "timing_stats", {}) or {}

    # generated frames: prefer SDK reported frames, else fallback to timing_stats arrays
    gen_frames = 0
    if "writer_per_frame_ms" in timing_stats and isinstance(timing_stats["writer_per_frame_ms"], (list, tuple)):
        gen_frames = len(timing_stats["writer_per_frame_ms"])
    elif "num_frames" in timing_stats:
        gen_frames = int(timing_stats["num_frames"])
    else:
        # fallback: expected frames from audio
        gen_frames = int(round(trimmed_audio_duration * args.fps))

    # FFD: first frame delay (ms), prefer SDK if available
    ffd_ms = None
    if "ffd_ms" in timing_stats:
        ffd_ms = safe_float(timing_stats["ffd_ms"], default=None)
    elif "first_frame_ms" in timing_stats:
        ffd_ms = safe_float(timing_stats["first_frame_ms"], default=None)

    if ffd_ms is None:
        # conservative: approximate as time to produce first "effective_frames" chunk
        ffd_ms = elapsed * 1000.0

    # If saving videos, mux audio after the measured window
    if args.save_videos and final_output is not None:
        try:
            ffmpeg_mux(temp_output, audio_path, final_output)
        except Exception:
            # Even if mux fails, keep going.
            pass

    # Cleanup temp output if not saving videos
    try:
        if (not args.save_videos) and os.path.exists(temp_output):
            os.remove(temp_output)
    except Exception:
        pass

    return elapsed, float(ffd_ms), int(gen_frames), timing_stats


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch Performance Benchmarking for Ditto (RTF + FFD)")

    parser.add_argument("--data_root", type=str,
                        default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_trt_Ampere_Plus",
                        help="Path to model directory")
    parser.add_argument("--cfg_pkl", type=str,
                        default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
                        help="Path to config pickle")

    parser.add_argument("--test_dir", type=str,
                        default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/testset",
                        help="Directory containing test set folders")

    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Specific checkpoint path (optional)")

    parser.add_argument("--output_dir", type=str,
                        default="/tmp/ditto_perf_test",
                        help="Directory for metric logs and temp files")

    parser.add_argument("--mode", type=str, default="offline", choices=["offline", "online"],
                        help="SDK mode: 'offline' (batch processing) or 'online' (streaming/realtime)")

    parser.add_argument("--use_meanflow", action="store_true", default=False,
                        help="Enable meanflow option in SDK (default: False)")

    parser.add_argument("--max_seconds", type=float, default=10.0,
                        help="Maximum audio length in seconds (default: 10.0, use 0 for full length)")

    parser.add_argument("--save_videos", action="store_true", default=True,
                        help="Save generated videos for quality metrics")
    parser.add_argument("--results_dir", type=str,
                        default="/workspace/ditto/ditto-talkinghead-train/rtf_results/original50steps_tensorRT_Talk8",
                        help="Directory to save generated videos (only used with --save_videos)")

    # Controls you wanted fixed
    parser.add_argument("--sampling_timesteps", type=int, default=50,
                        help="Number of denoising steps (e.g., 10 for Ours-s10, 50 for Ours-s50)")
    parser.add_argument("--overlap_v2", type=int, default=10,
                        help="Overlap frames (offline chunk overlap / online fusion overlap depending on pipeline)")
    parser.add_argument("--max_size", type=int, default=512,
                        help="Max resolution for processing (e.g., 512). If your pipeline expects 1920 default, adjust here.")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="FPS used for RTF duration conversion (default: 25.0)")

    args = parser.parse_args()

    # Effective frames: online produces "current" frames per step; offline produces sequence frames with overlap trimming.
    # Keep your project convention:
    #   online: chunksize=(3,5,2) -> current=5 frames -> 0.2s @25fps
    #   offline: seq_frames=80, overlap=10 -> valid=70 frames
    if args.mode == "online":
        effective_frames = 5
        chunksize = (3, 5, 2)
    else:
        effective_frames = 70
        chunksize = None

    # Banner
    print("\n" + "=" * 60)
    print("🚀 Batch Performance Benchmark")
    print("=" * 60)
    print(f"Mode             : {args.mode.upper()}")
    print(f"Meanflow          : {args.use_meanflow}")
    print(f"Sampling timesteps: {args.sampling_timesteps}")
    if chunksize is not None:
        print(f"Chunksize         : {chunksize} (pre,cur,post)")
    print(f"Overlap_v2        : {args.overlap_v2}")
    print(f"Max size          : {args.max_size}")
    print(f"Max seconds       : {args.max_seconds}")
    print(f"FPS               : {args.fps}")
    print("=" * 60 + "\n")

    samples = list_test_samples(args.test_dir)
    if not samples:
        print(f"❌ No valid samples found in: {args.test_dir}")
        sys.exit(1)

    ensure_dir(args.output_dir)

    # Accumulators
    rtf_audio_list = []
    rtf_valid_list = []
    rtf_paper_list = []  # paper-style RTF (streaming definition; see comments below)
    ffd_list = []

    # Module timing accumulators (optional)
    audio2feat_list = []
    dit_total_list = []
    face_total_list = []   # face pipeline total (warp+decode+stitch+putback if present)

    # Per-sample logs
    per_sample = []

    # Main loop
    for idx, sample in enumerate(samples, 1):
        print(f"\n[{idx}/{len(samples)}] ▶ {sample['name']}")
        elapsed, ffd_ms, gen_frames, timing_stats = run_inference_measurement(
            args=args,
            sample=sample,
            sdk_mode=args.mode,
            effective_frames=effective_frames,
        )

        fps = float(args.fps)
        trimmed_audio_duration = load_audio_duration_seconds(sample["audio"], args.max_seconds)

        # Durations
        valid_duration = gen_frames / fps if gen_frames > 0 else 0.0

        # RTF(audio)
        rtf_audio = elapsed / trimmed_audio_duration if trimmed_audio_duration > 0 else float("inf")

        # RTF(valid frames)
        rtf_valid = elapsed / valid_duration if valid_duration > 0 else float("inf")

        # Paper-style RTF: use the *expected output duration* (= audio seconds at given FPS).
        # This avoids under/over-counting when the SDK internally pads/overlaps frames.
        expected_frames = int(round(trimmed_audio_duration * float(fps)))
        expected_duration = (expected_frames / float(fps)) if expected_frames > 0 else 0.0
        rtf_paper = (elapsed / expected_duration) if expected_duration > 0 else float('inf')

        # Timing breakdown extraction (robust keys)
        audio2feat_ms = safe_float(timing_stats.get("audio2feat_ms", np.nan))
        dit_total_ms = safe_float(timing_stats.get("dit_total_ms", np.nan))
        dit_per_chunk_ms = safe_float(timing_stats.get("dit_per_chunk_ms", np.nan))

        warp_ms = safe_float(timing_stats.get("warp_total_ms", np.nan))
        decode_ms = safe_float(timing_stats.get("decode_total_ms", np.nan))
        stitch_ms = safe_float(timing_stats.get("stitch_total_ms", np.nan))
        putback_ms = safe_float(timing_stats.get("putback_total_ms", np.nan))
        writer_total_ms = safe_float(timing_stats.get("writer_total_ms", np.nan))

        face_total_ms = np.nan
        if not np.isnan(warp_ms) or not np.isnan(decode_ms) or not np.isnan(stitch_ms) or not np.isnan(putback_ms):
            face_total_ms = float(np.nansum([warp_ms, decode_ms, stitch_ms, putback_ms]))

        # Save clip result
        clip_result = {
            "name": sample["name"],
            "elapsed_sec": elapsed,
            "ffd_ms": ffd_ms,
            "gen_frames": gen_frames,
            "fps": fps,
            "audio_sec": trimmed_audio_duration,
            "valid_sec": valid_duration,
            "expected_frames": expected_frames,
            "expected_duration": expected_duration,
            "rtf_audio": rtf_audio,
            "rtf_valid": rtf_valid,
            "rtf_paper": rtf_paper,
            "timing": {
                "audio2feat_ms": audio2feat_ms,
                "dit_total_ms": dit_total_ms,
                "dit_per_chunk_ms": dit_per_chunk_ms,
                "writer_total_ms": writer_total_ms,
                "warp_total_ms": warp_ms,
                "decode_total_ms": decode_ms,
                "stitch_total_ms": stitch_ms,
                "putback_total_ms": putback_ms,
                "face_total_ms": face_total_ms,
            },
        }
        per_sample.append(clip_result)

        # Console per-sample brief
        print(f"  ⏱️  RTF(audio): {rtf_audio:.4f} | RTF(valid): {rtf_valid:.4f} | RTF(paper): {rtf_paper:.4f} | FFD: {ffd_ms:.1f}ms")
        if not np.isnan(dit_total_ms):
            print(f"      🧠 DiT: {pretty_ms(dit_total_ms)} (chunk avg: {pretty_ms(dit_per_chunk_ms)}) | 📝 Writer: {pretty_ms(writer_total_ms)}")
        if not np.isnan(face_total_ms):
            print(f"      🔄 Warp: {pretty_ms(warp_ms)} | 🎨 Decode: {pretty_ms(decode_ms)} | 🧵 Stitch: {pretty_ms(stitch_ms)} | 📦 Putback: {pretty_ms(putback_ms)}")

        # Accumulate
        rtf_audio_list.append(rtf_audio)
        rtf_valid_list.append(rtf_valid)
        rtf_paper_list.append(rtf_paper)
        ffd_list.append(ffd_ms)

        if not np.isnan(audio2feat_ms):
            audio2feat_list.append(audio2feat_ms)
        if not np.isnan(dit_total_ms):
            dit_total_list.append(dit_total_ms)
        if not np.isnan(face_total_ms):
            face_total_list.append(face_total_ms)

    # Summary
    mean_rtf_audio = float(np.mean(rtf_audio_list)) if len(rtf_audio_list) else float("nan")
    std_rtf_audio  = float(np.std(rtf_audio_list))  if len(rtf_audio_list) else float("nan")
    mean_rtf_paper = float(np.mean(rtf_paper_list)) if len(rtf_paper_list) else float('nan')
    std_rtf_paper  = float(np.std(rtf_paper_list))  if len(rtf_paper_list) else float('nan')
    mean_rtf_valid = float(np.mean(rtf_valid_list)) if len(rtf_valid_list) else float("nan")
    std_rtf_valid  = float(np.std(rtf_valid_list))  if len(rtf_valid_list) else float("nan")

    mean_ffd = float(np.mean(ffd_list)) if len(ffd_list) else float("nan")
    std_ffd  = float(np.std(ffd_list))  if len(ffd_list) else float("nan")
    min_ffd  = float(np.min(ffd_list))  if len(ffd_list) else float("nan")
    max_ffd  = float(np.max(ffd_list))  if len(ffd_list) else float("nan")

    # Avg timing breakdown
    mean_audio2feat = float(np.mean(audio2feat_list)) if len(audio2feat_list) else None
    mean_dit_total  = float(np.mean(dit_total_list)) if len(dit_total_list) else None
    mean_face_total = float(np.mean(face_total_list)) if len(face_total_list) else None

    # Prepare log
    log_obj = {
        "mode": args.mode,
        "use_meanflow": args.use_meanflow,
        "sampling_timesteps": args.sampling_timesteps,
        "overlap_v2": args.overlap_v2,
        "max_size": args.max_size,
        "max_seconds": args.max_seconds,
        "fps": args.fps,
        "samples": len(samples),
        "results": per_sample,
        "summary": {
            "mean_rtf_audio": mean_rtf_audio,
            "std_rtf_audio": std_rtf_audio,
            "mean_rtf_valid": mean_rtf_valid,
            "std_rtf_valid": std_rtf_valid,
            "mean_rtf_paper": mean_rtf_paper,
            "std_rtf_paper": std_rtf_paper,
            "mean_ffd_ms": mean_ffd,
            "std_ffd_ms": std_ffd,
            "min_ffd_ms": min_ffd,
            "max_ffd_ms": max_ffd,
            "mean_audio2feat_ms": mean_audio2feat,
            "mean_dit_total_ms": mean_dit_total,
            "mean_face_total_ms": mean_face_total,
        }
    }

    log_path = os.path.join(args.output_dir, f"performance_{args.mode}_baseline.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_obj, f, indent=2, ensure_ascii=False)

    # Print report
    print("\n" + "=" * 60)
    print("📊 BATCH PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Mode    : {args.mode.upper()}")
    print(f"sampling_timesteps : {args.sampling_timesteps}steps")
    print(f"max_size : {args.max_size}")
    print(f"overlap_v2 : {args.overlap_v2}")
    print(f"Samples : {len(samples)} / {len(samples)}")
    print(f"Meanflow: {args.use_meanflow}")
    print(f"Max Sec : {args.max_seconds}")
    print("-" * 60)

    # RTF summary
    if args.mode == "online":
        # Ditto paper reports RTF for streaming inference (video stream output).
        print(f"⏱️  Avg RTF(paper/stream): {mean_rtf_paper:.4f} (± {std_rtf_paper:.4f})")
        if mean_rtf_paper < 1.0:
            print(f"   Status: ✅ Real-time Capable ({1/mean_rtf_paper:.2f}x speed)")
        else:
            print("   Status: ⚠️  Slower than Real-time")
        # (debug) also show audio-based and generated-video-based RTF
        print(f"⏱️  Avg RTF(audio): {mean_rtf_audio:.4f} (± {std_rtf_audio:.4f})")
        print(f"⏱️  Avg RTF(valid): {mean_rtf_valid:.4f} (± {std_rtf_valid:.4f})")
    else:
        # OFFLINE batch mode uses a different definition than the paper (not step-wise streaming).
        print(f"⏱️  Avg RTF(batch/audio): {mean_rtf_audio:.4f} (± {std_rtf_audio:.4f})")
        if mean_rtf_audio < 1.0:
            print(f"   Status: ✅ Real-time Capable ({1/mean_rtf_audio:.2f}x speed)")
        else:
            print("   Status: ⚠️  Slower than Real-time")
        print("   ℹ️  Note: Ditto paper Table 1/5 RTF is measured in ONLINE streaming mode (video stream output).")
        print("      OFFLINE batch RTF uses a different definition, so do not compare the numbers directly.")

    print(f"🚀 Avg FFD : {mean_ffd:.2f} ms (± {std_ffd:.2f})")
    print(f"⚡ Min FFD : {min_ffd:.2f} ms")
    print(f"🐌 Max FFD : {max_ffd:.2f} ms")
    print("-" * 60)

    # Timing breakdown from per-sample averages (if available)
    print("📌 TIMING BREAKDOWN (Average per video):")
    if mean_dit_total is not None:
        print(f"   🧠 DiT (Diffusion) Total : {pretty_ms(mean_dit_total)}")
        # try to infer per-step avg (using ONLINE effective_frames)
        if args.mode == "online":
            # estimate steps from expected frames / effective_frames
            step_frames = 5
            step_count = max(1, int(round((args.max_seconds * args.fps) / step_frames))) if args.max_seconds > 0 else None
            if step_count:
                print(f"      └─ Per Step Avg      : {pretty_ms(mean_dit_total / step_count)}")
        else:
            # offline: per "valid step" is ambiguous; keep total only
            pass

    # Writer / face totals from sample logs (we didn't keep mean of each component robustly here)
    # But we can reconstruct from summary mean_face_total_ms if present
    if mean_face_total is not None:
        print(f"   🎨 Face Pipeline Total   : {pretty_ms(mean_face_total)} (warp+decode+stitch+putback)")
    if mean_audio2feat is not None:
        print(f"   🔊 Audio2Feat Total      : {pretty_ms(mean_audio2feat)}")
    print("-" * 60)

    # Table 5-style (ONLINE only)
    if args.mode == "online" and mean_audio2feat is not None and mean_dit_total is not None and mean_face_total is not None:
        # Use expected frames to compute step count
        # step duration = effective_frames * frame_ms (paper uses ms/step for audio2feat & motion, ms/frame for rendering)
        fps = float(args.fps)
        frame_ms = 1000.0 / fps
        effective_frames = 5  # online current frames
        step_ms = (effective_frames * frame_ms)

        # Steps inferred from expected output frames (per video)
        # (Use per_sample to compute average steps more accurately)
        step_counts = []
        for r in per_sample:
            exp_frames = int(r.get("expected_frames", 0))
            if exp_frames > 0:
                step_counts.append(max(1, int(math.ceil(exp_frames / float(effective_frames)))))
        mean_steps = float(np.mean(step_counts)) if step_counts else 1.0

        audio2feat_ms_per_step = mean_audio2feat / mean_steps
        dit_ms_per_step = mean_dit_total / mean_steps

        # face rendering per frame: face_total / expected_frames
        exp_frames_list = [int(r.get("expected_frames", 0)) for r in per_sample if int(r.get("expected_frames", 0)) > 0]
        mean_exp_frames = float(np.mean(exp_frames_list)) if exp_frames_list else (args.max_seconds * args.fps if args.max_seconds > 0 else 1.0)
        face_ms_per_frame = mean_face_total / mean_exp_frames

        print("📋 Table 5-style (Module time + Module RTF) — ONLINE streaming only")
        print(f"   step = {effective_frames} frames ({step_ms:.1f} ms), frame = {frame_ms:.1f} ms @ {fps:.1f} fps")
        # Paper Table 5 uses: step_RTF = (ms/step) / (effective_frames * frame_ms), frame_RTF = (ms/frame) / frame_ms
        step_ms = (effective_frames * frame_ms)
        audio2feat_rtf = (audio2feat_ms_per_step / step_ms) if step_ms > 0 else float("nan")
        motion_dit_rtf  = (dit_ms_per_step / step_ms) if step_ms > 0 else float("nan")
        face_rtf        = (face_ms_per_frame / frame_ms) if frame_ms > 0 else float("nan")
        print(f"   Audio2Feat     : {audio2feat_ms_per_step:7.2f} ms/step | RTF {audio2feat_rtf:6.3f}")
        print(f"   Motion DiT     : {dit_ms_per_step:7.2f} ms/step | RTF {motion_dit_rtf:6.3f}")
        print(f"   Face Rendering : {face_ms_per_frame:7.2f} ms/frame| RTF {face_rtf:6.3f}")
        print("-" * 60)
    elif args.mode == "offline":
        print("ℹ️  Table 5-style module breakdown/RTF is defined for ONLINE streaming inference.")
        print("   (OFFLINE mode definition differs, so this section is hidden.)")
        print("-" * 60)

    print(f"📝 Log saved to: {log_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
