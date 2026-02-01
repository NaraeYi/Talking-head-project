#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rtf_ditto_batch.py
==================
Batch performance benchmark for Ditto (RTF + FFD + module timing breakdown)
- Supports ONLINE (streaming) and OFFLINE (batch) StreamSDK pipelines.
- Prints:
  * Avg RTF(audio), Avg RTF(valid), Avg FFD
  * TIMING BREAKDOWN (Average per video)
  * Table 5-style module time + module RTF (Average per video)
- Saves a JSON log under output_dir.

NOTE
- ONLINE 모드에서는 run_chunk로 오디오를 frame 단위로 feed.
- OFFLINE 모드에서는 wav2feat.wav2feat(audio)로 전체 오디오 feature를 1번에 만들고 queue에 1번 push.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np
from torch._dynamo.decorators import F
from tqdm import tqdm

import torch
import librosa

# 로컬 파일(stream_pipeline_online/offline.py) import 보장
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ----------------------------
# Audio / file helpers
# ----------------------------
def load_audio_16k_mono(audio_path: str, max_seconds: float) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio.astype(np.float32)
    if max_seconds and max_seconds > 0:
        max_samples = int(max_seconds * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
    return audio


def find_first_existing(folder: str, names: List[str]) -> Optional[str]:
    for n in names:
        p = os.path.join(folder, n)
        if os.path.isfile(p):
            return p
    return None


def find_clip_folders(test_dir: str) -> List[str]:
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"test_dir not found: {test_dir}")
    subdirs = []
    for name in sorted(os.listdir(test_dir)):
        p = os.path.join(test_dir, name)
        if os.path.isdir(p):
            subdirs.append(p)
    return subdirs


# ----------------------------
# Metrics structs
# ----------------------------
@dataclass
class PerVideoResult:
    clip: str
    mode: str
    meanflow: bool
    video_path: Optional[str]
    audio_sec: float
    valid_sec: float
    elapsed_sec: float
    rtf_audio: float
    rtf_valid: float
    ffd_ms: Optional[float]

    # step/frame definitions (for Table5-style)
    fps: float
    step_frames: int
    step_ms: float
    frame_ms: float

    # module times (avg)
    audio2feat_ms_step: Optional[float]
    motion_dit_ms_step: Optional[float]
    face_ms_frame: Optional[float]

    # module rtf
    audio2feat_rtf: Optional[float]
    motion_dit_rtf: Optional[float]
    face_rtf: Optional[float]

    # raw breakdown (totals)
    dit_total_ms: Optional[float]
    writer_total_ms: Optional[float]
    warp_total_ms: Optional[float]
    decode_total_ms: Optional[float]
    stitch_total_ms: Optional[float]
    putback_total_ms: Optional[float]


def safe_mean(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
        return float(np.mean(x))
    return None


# ----------------------------
# Core single inference
# ----------------------------
def perform_single_infer(
    sdk,
    audio: np.ndarray,
    audio_duration: float,
    fps: float,
    source_path: str,
    output_path: str,
    audio_path: str,
    mode: str,
    chunksize: Tuple[int, int, int],
    sampling_timesteps: int,
    overlap_v2: int,
    max_size: int,
    ffd_timeout_sec: float,
) -> tuple[float, Optional[float], Optional[int], int]:
    """
    Returns:
      elapsed_sec, ffd_ms, generated_frames, step_frames
    """

    # ✅ 논문 셋업처럼 "setup에 전달" (중요)
    sdk.setup(
        source_path,
        output_path,
        online_mode=(mode == "offline"),
        sampling_timesteps=sampling_timesteps,
        overlap_v2=overlap_v2,
        max_size=max_size,
    )

    got_ffd = False
    ffd_ms = None

    def try_mark_ffd(t0):
        nonlocal got_ffd, ffd_ms
        if got_ffd:
            return
        if hasattr(sdk, "timing_stats") and isinstance(sdk.timing_stats, dict):
            w = sdk.timing_stats.get("writer_per_frame_ms", None)
            if isinstance(w, (list, tuple)) and len(w) > 0:
                got_ffd = True
                ffd_ms = (time.perf_counter() - t0) * 1000.0

    # CUDA sync before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    step_frames = None

    if mode == "online":
        pre, cur, post = chunksize
        step_frames = int(cur)

        # 25fps, 16kHz => 1 frame = 0.04s = 640 samples
        hop = 640

        # pre-context padding
        audio_padded = np.concatenate([np.zeros((pre * hop,), dtype=np.float32), audio], axis=0)

        # window length: (pre+cur+post) frames (+ small pad)
        split_len = int((pre + cur + post) * hop) + 80

        # advance by cur frames each step
        for i in range(0, len(audio_padded), cur * hop):
            audio_chunk = audio_padded[i : i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            sdk.run_chunk(audio_chunk, chunksize)
            try_mark_ffd(t0)

        sdk.close()

    else:
        # OFFLINE: full audio -> wav2feat.wav2feat(audio) once
        # (pipeline 쪽에 audio2feat timer가 없을 수 있으니 여기서도 기록)
        if hasattr(sdk, "timing_stats") and isinstance(sdk.timing_stats, dict):
            sdk.timing_stats.setdefault("audio2feat_total_ms", 0.0)
            sdk.timing_stats.setdefault("audio2feat_per_step_ms", [])

        t_a2f0 = time.perf_counter()
        aud_feat = sdk.wav2feat.wav2feat(audio)
        t_a2f_ms = (time.perf_counter() - t_a2f0) * 1000.0

        if hasattr(sdk, "timing_stats") and isinstance(sdk.timing_stats, dict):
            sdk.timing_stats["audio2feat_total_ms"] += t_a2f_ms
            sdk.timing_stats["audio2feat_per_step_ms"].append(t_a2f_ms)

        sdk.audio2motion_queue.put(aud_feat)
        sdk.close()

        vf = getattr(getattr(sdk, "audio2motion", None), "valid_clip_len", None)
        step_frames = int(vf) if vf is not None else int(round(audio_duration * fps))

    # ----------------------------
    # FINALIZE: mux audio into final mp4 (same idea as inference.py)
    # - output: output_path (final mp4 with audio)
    # - input video: sdk.tmp_output_path (writer temp)
    # NOTE: batch script can truncate audio by --max_seconds, so we use "-shortest"
    #       to avoid producing longer audio than video.
    # ----------------------------
    if hasattr(sdk, "tmp_output_path") and isinstance(sdk.tmp_output_path, str):
        if os.path.exists(sdk.tmp_output_path) and os.path.exists(audio_path):
            cmd = (
                f'ffmpeg -loglevel error -y '
                f'-i "{sdk.tmp_output_path}" -i "{audio_path}" '
                f'-map 0:v -map 1:a -c:v copy -c:a aac -shortest "{output_path}"'
            )
            ret = os.system(cmd)
            if ret != 0:
                raise RuntimeError(f"ffmpeg mux failed (exit={ret}). cmd={cmd}")
            # cleanup temp file after successful mux
            try:
                os.remove(sdk.tmp_output_path)
            except Exception:
                pass


    # wait a bit for ffd best-effort
    if (not got_ffd) and ffd_timeout_sec and ffd_timeout_sec > 0:
        t_deadline = t0 + ffd_timeout_sec
        while time.perf_counter() < t_deadline:
            try_mark_ffd(t0)
            if got_ffd:
                break
            time.sleep(0.005)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated_frames = None
    if hasattr(sdk, "timing_stats") and isinstance(sdk.timing_stats, dict):
        w = sdk.timing_stats.get("writer_per_frame_ms", None)
        if isinstance(w, (list, tuple)) and len(w) > 0:
            generated_frames = len(w)

    return elapsed, ffd_ms, generated_frames, int(step_frames)


# ----------------------------
# Batch runner + reporting
# ----------------------------
def summarize_list(vals: List[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))

# ----------------------------
# path helpers
# ----------------------------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def sanitize_filename(name: str) -> str:
    # keep simple + safe
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in name)
    
def make_run_name(mode: str, use_meanflow: bool, sampling_timesteps: int) -> str:
    if use_meanflow:
        tag = "mf1step"
    else:
        tag = f"dit{int(sampling_timesteps)}step"
    return f"{mode}_{tag}"



def run_batch(args):
    # import pipeline
    if args.mode == "online":
        from stream_pipeline_online import StreamSDK
        print("✅ Loaded 'stream_pipeline_online.py' (ONLINE mode)")
    else:
        from stream_pipeline_offline import StreamSDK
        print("✅ Loaded 'stream_pipeline_offline.py' (OFFLINE mode)")

    fps = float(args.fps)
    chunksize = tuple(int(x) for x in args.chunksize.split(","))  # e.g. "3,5,2"
    assert len(chunksize) == 3, "--chunksize must be 'pre,cur,post'"

    # Collect clips
    clip_dirs = find_clip_folders(args.test_dir)
    if len(clip_dirs) == 0:
        raise RuntimeError(f"No clip folders found in {args.test_dir}")

    # Map clip folders -> media paths
    clips = []
    for d in clip_dirs:
        audio_path = find_first_existing(d, ["audio.wav", "audio.flac", "audio.mp3"])
        src_path = find_first_existing(d, ["image.png", "image.jpg", "image.jpeg", "source.png", "source.jpg", "source.jpeg"])
        if audio_path is None or src_path is None:
            print(f"⚠️ Skip (missing files): {d}")
            continue
        clips.append({"name": os.path.basename(d), "dir": d, "audio": audio_path, "source": src_path})

    if args.num_samples and args.num_samples > 0:
        clips = clips[: args.num_samples]

    print("=" * 60)
    print("🚀 Batch Performance Benchmark")
    print("=" * 60)
    print(f"Mode             : {args.mode.upper()}")
    print(f"Meanflow          : {args.use_meanflow}")
    print(f"Sampling timesteps: {args.sampling_timesteps}")
    print(f"Chunksize         : {chunksize} (pre,cur,post)")
    print(f"Overlap_v2        : {args.overlap_v2}")
    print(f"Max seconds       : {args.max_seconds}")
    print(f"FPS               : {fps}")
    print(f"Samples           : {len(clips)}")
    print("=" * 60)

    # #os.makedirs(args.output_dir, exist_ok=True)
    ensure_dir(args.output_dir)
    # Always write videos under a dedicated subfolder (easier to inspect)
    run_name = args.run_name if args.run_name else make_run_name(
    args.mode, bool(args.use_meanflow), args.sampling_timesteps
    )
    run_dir = ensure_dir(os.path.join(args.output_dir, run_name))
    video_dir = ensure_dir(os.path.join(run_dir, "videos"))

    print(f"🗂️  Run dir         : {run_dir}")
    print(f"🎞️  Video dir       : {video_dir}")
    print(f"🎞️  Save videos     : {bool(args.save_videos)}")

    # Init SDK once (load model once)
    # (checkpoint_path는 SDK에서 안 쓰면 무시될 가능성이 높지만, 호환 위해 전달)
    sdk = StreamSDK(
        args.cfg_pkl,
        args.data_root,
        use_meanflow=bool(args.use_meanflow),
        checkpoint_path=args.checkpoint_path,
    )

    # Warmup
    if args.warmup and len(clips) > 0:
        w = clips[0]
        try:
            w_audio = load_audio_16k_mono(w["audio"], max_seconds=min(args.max_seconds, 3.0) if args.max_seconds > 0 else 3.0)
            w_dur = float(len(w_audio) / 16000.0)
            # #tmp_out = os.path.join(args.output_dir, "_warmup_tmp.mp4")
            # warmup video goes to output_dir root (and removed by default)
            tmp_out = os.path.join(video_dir, "_warmup_tmp.mp4")

            try:
                perform_single_infer(
                    sdk=sdk,
                    audio=w_audio,
                    audio_duration=w_dur,
                    fps=fps,
                    source_path=w["source"],
                    output_path=tmp_out,
                    audio_path=w["audio"],
                    mode=args.mode,
                    chunksize=chunksize,
                    sampling_timesteps=args.sampling_timesteps,
                    overlap_v2=args.overlap_v2,
                    max_size=args.max_size,
                    ffd_timeout_sec=args.ffd_timeout,
                )
            finally:
                # #if os.path.exists(tmp_out):
                # keep warmup only if explicitly requested
                if (not args.save_warmup_video) and os.path.exists(tmp_out):
                    os.remove(tmp_out)
        except Exception as e:
            print(f"⚠️ Warmup failed (continuing): {e}")

    results: List[PerVideoResult] = []

    pbar = tqdm(clips, desc="writer", ncols=120)
    for clip in pbar:
        name = clip["name"]
        audio = load_audio_16k_mono(clip["audio"], args.max_seconds)
        audio_sec = float(len(audio) / 16000.0)

        # #out_path = os.path.join(args.output_dir, f"{name}.mp4")
        # save video into videos/ with informative filename
        safe_name = sanitize_filename(name)
        tag = "meanflow" if args.use_meanflow else f"diff{args.sampling_timesteps}"
        out_path = os.path.join(
            video_dir,
            f"{safe_name}_{args.mode}_{tag}.mp4"
        )

        try:
            elapsed, ffd_ms, gen_frames, step_frames = perform_single_infer(
                sdk=sdk,
                audio=audio,
                audio_duration=audio_sec,
                fps=fps,
                source_path=clip["source"],
                output_path=out_path,
                audio_path=clip["audio"],
                mode=args.mode,
                chunksize=chunksize,
                sampling_timesteps=args.sampling_timesteps,
                overlap_v2=args.overlap_v2,
                max_size=args.max_size,
                ffd_timeout_sec=args.ffd_timeout,
                # video_path=out_path if os.path.exists(out_path) else None, #
            )

            # valid duration (generated frames / fps)
            valid_sec = audio_sec if gen_frames is None else float(gen_frames / fps)

            rtf_audio = elapsed / max(audio_sec, 1e-8)
            rtf_valid = elapsed / max(valid_sec, 1e-8)

            # --- timing breakdown from sdk.timing_stats ---
            ts: Dict = getattr(sdk, "timing_stats", {}) if hasattr(sdk, "timing_stats") else {}
            dit_total_ms = ts.get("dit_total_ms", None)
            dit_per_chunk = ts.get("dit_per_chunk_ms", None)
            writer_total_ms = ts.get("writer_total_ms", None)
            writer_per_frame = ts.get("writer_per_frame_ms", None)
            warp_total_ms = ts.get("warp_total_ms", None)
            decode_total_ms = ts.get("decode_total_ms", None)
            stitch_total_ms = ts.get("stitch_total_ms", None)
            putback_total_ms = ts.get("putback_total_ms", None)

            # audio2feat (pipeline이 제공하면 그걸 사용, 아니면 offline에서 여기서 기록됨)
            a2f_per_step = ts.get("audio2feat_per_step_ms", None)
            a2f_total_ms = ts.get("audio2feat_total_ms", None)

            # step/frame definitions
            step_ms = float(step_frames * (1000.0 / fps))
            frame_ms = float(1000.0 / fps)

            # Table5-style: module times
            audio2feat_ms_step = safe_mean(a2f_per_step)
            if audio2feat_ms_step is None and a2f_total_ms is not None:
                steps = max(1, int(math.ceil(valid_sec / (step_frames / fps))))
                audio2feat_ms_step = float(a2f_total_ms) / steps

            # motion_dit_ms_step: MeanFlow는 1-step이므로 클립 시간 그대로, Ditto는 sampling_timesteps로 나누기
            motion_dit_ms_step = safe_mean(dit_per_chunk)
            if motion_dit_ms_step is not None and not args.use_meanflow and args.sampling_timesteps > 1:
                # Ditto (Diffusion): 클립 시간을 sampling_timesteps로 나누어 1 step 시간 계산
                motion_dit_ms_step = motion_dit_ms_step / args.sampling_timesteps
            # MeanFlow는 1-step이므로 나누지 않음 (motion_dit_ms_step = 클립 처리 시간)

            # Face Rendering: per-frame average of (writer+warp+decode+stitch+putback)
            face_ms_frame = None
            if gen_frames and gen_frames > 0:
                face_total = 0.0
                for v in [writer_total_ms, warp_total_ms, decode_total_ms, stitch_total_ms, putback_total_ms]:
                    if v is not None:
                        face_total += float(v)
                face_ms_frame = face_total / float(gen_frames)
            else:
                face_ms_frame = safe_mean(writer_per_frame)

            # Table5-style: module RTF
            audio2feat_rtf = (audio2feat_ms_step / step_ms) if (audio2feat_ms_step is not None and step_ms > 0) else None
            motion_dit_rtf = (motion_dit_ms_step / step_ms) if (motion_dit_ms_step is not None and step_ms > 0) else None
            face_rtf = (face_ms_frame / frame_ms) if (face_ms_frame is not None and frame_ms > 0) else None

            results.append(
                PerVideoResult(
                    meanflow=bool(args.use_meanflow),
                    video_path=out_path if (args.save_videos and os.path.exists(out_path)) else None,

                    clip=name,
                    mode=args.mode,
                    # meanflow=bool(args.use_meanflow),
                    audio_sec=audio_sec,
                    valid_sec=valid_sec,
                    elapsed_sec=elapsed,
                    rtf_audio=rtf_audio,
                    rtf_valid=rtf_valid,
                    ffd_ms=ffd_ms,
                    fps=fps,
                    step_frames=int(step_frames),
                    step_ms=step_ms,
                    frame_ms=frame_ms,
                    audio2feat_ms_step=audio2feat_ms_step,
                    motion_dit_ms_step=motion_dit_ms_step,
                    face_ms_frame=face_ms_frame,
                    audio2feat_rtf=audio2feat_rtf,
                    motion_dit_rtf=motion_dit_rtf,
                    face_rtf=face_rtf,
                    dit_total_ms=float(dit_total_ms) if dit_total_ms is not None else None,
                    writer_total_ms=float(writer_total_ms) if writer_total_ms is not None else None,
                    warp_total_ms=float(warp_total_ms) if warp_total_ms is not None else None,
                    decode_total_ms=float(decode_total_ms) if decode_total_ms is not None else None,
                    stitch_total_ms=float(stitch_total_ms) if stitch_total_ms is not None else None,
                    putback_total_ms=float(putback_total_ms) if putback_total_ms is not None else None,
                )
            )

            pbar.set_postfix_str(
                f"RTF(audio)={rtf_audio:.4f} | RTF(valid)={rtf_valid:.4f} | FFD={ffd_ms:.1f}ms"
                if ffd_ms is not None else f"RTF(audio)={rtf_audio:.4f}"
            )

        except Exception as e:
            print(f"\n❌ Error on clip {name}: {e}")
            traceback.print_exc()

        finally:
            # remove output unless asked to keep
            # if (not args.save_videos) and os.path.exists(out_path):
            #     try:
            #         os.remove(out_path)
            #     except:
            #         pass
            if (not args.save_videos):
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except:
                        pass

    # ----------------------------
    # Aggregate report
    # ----------------------------
    if len(results) == 0:
        raise RuntimeError("No successful results.")

    rtf_audio_vals = [r.rtf_audio for r in results]
    rtf_valid_vals = [r.rtf_valid for r in results]
    ffd_vals = [r.ffd_ms for r in results if r.ffd_ms is not None]

    avg_rtf_audio, std_rtf_audio = summarize_list(rtf_audio_vals)
    avg_rtf_valid, std_rtf_valid = summarize_list(rtf_valid_vals)
    avg_ffd = float(np.mean(ffd_vals)) if len(ffd_vals) else None
    std_ffd = float(np.std(ffd_vals, ddof=0)) if len(ffd_vals) else None
    min_ffd = float(np.min(ffd_vals)) if len(ffd_vals) else None
    max_ffd = float(np.max(ffd_vals)) if len(ffd_vals) else None

    def avg_field(field: str) -> Optional[float]:
        vals = [getattr(r, field) for r in results if getattr(r, field) is not None]
        return float(np.mean(vals)) if vals else None

    # Per-step/per-frame stats
    dit_ms_step_vals = [r.motion_dit_ms_step for r in results if r.motion_dit_ms_step is not None]
    dit_ms_step = float(np.mean(dit_ms_step_vals)) if dit_ms_step_vals else None

    face_ms_frame_vals = [r.face_ms_frame for r in results if r.face_ms_frame is not None]
    face_ms_frame = float(np.mean(face_ms_frame_vals)) if face_ms_frame_vals else None

    a2f_ms_step_vals = [r.audio2feat_ms_step for r in results if r.audio2feat_ms_step is not None]
    a2f_ms_step = float(np.mean(a2f_ms_step_vals)) if a2f_ms_step_vals else None

    a2f_rtf_vals = [r.audio2feat_rtf for r in results if r.audio2feat_rtf is not None]
    dit_rtf_vals = [r.motion_dit_rtf for r in results if r.motion_dit_rtf is not None]
    face_rtf_vals = [r.face_rtf for r in results if r.face_rtf is not None]
    a2f_rtf = float(np.mean(a2f_rtf_vals)) if a2f_rtf_vals else None
    dit_rtf = float(np.mean(dit_rtf_vals)) if dit_rtf_vals else None
    face_rtf = float(np.mean(face_rtf_vals)) if face_rtf_vals else None

    step_frames = results[0].step_frames
    step_ms = results[0].step_ms
    frame_ms = results[0].frame_ms

    print("\n" + "=" * 60)
    print("📊 BATCH PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Mode    : {args.mode.upper()}")
    ###
    print(f"sampling_timesteps : {args.sampling_timesteps}steps")
    print(f"max_size : {args.max_size}")
    print(f"overlap_v2 : {args.overlap_v2}")
    ###
    print(f"Samples : {len(results)} / {len(clips)}")
    print(f"Meanflow: {bool(args.use_meanflow)}")
    print(f"Max Sec : {args.max_seconds}")
    print("-" * 60)
    print(f"⏱️  Avg RTF(audio): {avg_rtf_audio:.4f} (± {std_rtf_audio:.4f})")
    if avg_rtf_audio > 0:
        status = "✅ Real-time Capable" if avg_rtf_audio < 1.0 else "❌ Not Real-time"
        print(f"   Status: {status} ({1.0/avg_rtf_audio:.2f}x speed)")
    print(f"⏱️  Avg RTF(valid): {avg_rtf_valid:.4f} (± {std_rtf_valid:.4f})")
    if avg_ffd is not None:
        print(f"🚀 Avg FFD : {avg_ffd:.2f} ms (± {std_ffd:.2f})")
        print(f"⚡ Min FFD : {min_ffd:.2f} ms")
        print(f"🐌 Max FFD : {max_ffd:.2f} ms")
    print("-" * 60)

    print("📌 TIMING BREAKDOWN (Average per video):")
    dit_total = avg_field("dit_total_ms")
    writer_total = avg_field("writer_total_ms")
    warp_total = avg_field("warp_total_ms")
    decode_total = avg_field("decode_total_ms")
    stitch_total = avg_field("stitch_total_ms")
    putback_total = avg_field("putback_total_ms")

    if dit_total is not None:
        print(f"   🧠 DiT (Diffusion) Total : {dit_total:,.2f} ms")
        if dit_ms_step is not None:
            print(f"      └─ Per Step Avg      : {dit_ms_step:,.2f} ms (step={step_frames} frames, {step_ms:.1f}ms)")
    if writer_total is not None:
        print(f"   📝 Writer Total          : {writer_total:,.2f} ms")
    if warp_total is not None:
        print(f"   🔄 Warp Total            : {warp_total:,.2f} ms")
    if decode_total is not None:
        print(f"   🎨 Decode Total          : {decode_total:,.2f} ms")
    if stitch_total is not None:
        print(f"   🧵 Stitch Total          : {stitch_total:,.2f} ms")
    if putback_total is not None:
        print(f"   📦 Putback Total         : {putback_total:,.2f} ms")

    ## ✅ 요청한 위치: 📌 아래, 📝 위에 Table5-style 출력
    print("-" * 60)
    print("📋 Table 5-style (Time + Module RTF) — Average per video")
    print(f"   step = {step_frames} frames ({step_ms:.1f} ms), frame = {frame_ms:.1f} ms @ {fps:.1f} fps")
    if a2f_ms_step is not None:
        if a2f_rtf is not None:
            print(f"   Audio2Feat     : {a2f_ms_step:>8.2f} ms/step | RTF {a2f_rtf:>6.3f}")
        else:
            print(f"   Audio2Feat     : {a2f_ms_step:>8.2f} ms/step")
    else:
        print("   Audio2Feat     : (no timing found)  ← stream_pipeline_*에 timer가 없으면 추가 필요")

    if dit_ms_step is not None:
        if dit_rtf is not None:
            print(f"   Motion DiT     : {dit_ms_step:>8.2f} ms/step | RTF {dit_rtf:>6.3f}")
        else:
            print(f"   Motion DiT     : {dit_ms_step:>8.2f} ms/step")
    else:
        print("   Motion DiT     : (no timing found)")

    if face_ms_frame is not None:
        if face_rtf is not None:
            print(f"   Face Rendering : {face_ms_frame:>8.2f} ms/frame| RTF {face_rtf:>6.3f}")
        else:
            print(f"   Face Rendering : {face_ms_frame:>8.2f} ms/frame")
    else:
        print("   Face Rendering : (no timing found)")

    print("-" * 60)


    # Save log
    log = {
        "args": vars(args),
        "summary": {
            "avg_rtf_audio": avg_rtf_audio,
            "std_rtf_audio": std_rtf_audio,
            "avg_rtf_valid": avg_rtf_valid,
            "std_rtf_valid": std_rtf_valid,
            "avg_ffd_ms": avg_ffd,
            "std_ffd_ms": std_ffd,
            "table5_style": {
                "step_frames": step_frames,
                "step_ms": step_ms,
                "frame_ms": frame_ms,
                "audio2feat_ms_step": a2f_ms_step,
                "motion_dit_ms_step": dit_ms_step,
                "face_ms_frame": face_ms_frame,
                "audio2feat_rtf": a2f_rtf,
                "motion_dit_rtf": dit_rtf,
                "face_rtf": face_rtf,
            },
        },
        "per_video": [asdict(r) for r in results],
    }

    out_name = f"performance_{args.mode}_{'meanflow' if args.use_meanflow else 'baseline'}.json"
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"📝 Log saved to: {out_path}")
    print("=" * 60)


def build_argparser():
    p = argparse.ArgumentParser(description="Ditto Batch Performance Benchmark (RTF + Table5-style module breakdown)")

    # Paths
    p.add_argument("--data_root", type=str,
                   default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_trt_Ampere_Plus",
                   help="Path to model directory")

    # p.add_argument("--cfg_pkl", type=str,
    #                default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
    #                help="Path to config pickle")
    # p.add_argument("--cfg_pkl", type=str,
    #                default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_meanflow.pkl",
    #                help="Path to config pickle")
    p.add_argument("--cfg_pkl", type=str,
                   default="/workspace/ditto/ditto-talkinghead-train/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_iMF.pkl",
                   help="Path to config pickle")

    p.add_argument("--checkpoint_path", type=str, default=None,
                    help="Optional checkpoint path (used by some configs; ignored if not supported)")               
    # p.add_argument("--checkpoint_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/experiments/ditto_original_hdtf_20251221_234237/weights/train_99.pt",
    #                help="Optional checkpoint path (used by some configs; ignored if not supported)")
    # p.add_argument("--checkpoint_path", type=str, default="/workspace/ditto/ditto-talkinghead-train/experiments/ditto_meanflow_hdtf_20251229_225128/samples/train_100.pt",
    #                help="Optional checkpoint path (used by some configs; ignored if not supported)")
    

    p.add_argument("--test_dir", type=str,
                   default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/testset",
                   help="Directory containing test set folders")
    p.add_argument("--output_dir", type=str,
                   default="/workspace/ditto/ditto-talkinghead-train/testset_results",
                   help="Directory to save logs (and videos if --save_videos)")

    # Mode
    p.add_argument("--mode", type=str, default="offline", choices=["offline", "online"],
                   help="SDK mode: 'offline' (batch) or 'online' (streaming)")
    p.add_argument("--use_meanflow", action="store_true", default=True,
                   help="Enable meanflow option in SDK")

    # RTF setup params (✅ 논문 셋업 고정하려면 여기 값을 맞추고, setup에 전달됨)
    p.add_argument("--sampling_timesteps", type=int, default=1,
                   help="Diffusion sampling steps (e.g., 10 or 50). Ignored if meanflow is used.")
    p.add_argument("--chunksize", type=str, default="3,5,2",
                   help="ONLINE only: chunksize as 'pre,cur,post' frames (default: 3,5,2)")
    p.add_argument("--overlap_v2", type=int, default=10,
                   help="Overlap frames for internal buffering (default: 10)")
    p.add_argument("--max_size", type=int, default=512,
                   help="Max image side for avatar registrar resize (default: 1920)")

    # Runtime
    p.add_argument("--fps", type=float, default=25.0, help="FPS assumed by pipeline (default: 25)")
    p.add_argument("--max_seconds", type=float, default=10.0,
                   help="Maximum audio length in seconds (0 = full length)")
    p.add_argument("--ffd_timeout", type=float, default=0.0,
                   help="Optional timeout (sec) to keep checking for first frame after inference starts")
    p.add_argument("--save_videos", action="store_true", default=True,
                   help="Keep generated mp4 files (default: off; will delete)")
    p.add_argument("--num_samples", type=int, default=0,
                   help="If >0, run only first N samples")
    p.add_argument("--warmup", action="store_true", default=False,
                   help="Run 1 short warmup clip before benchmark")
    p.add_argument("--save_warmup_video", action="store_true", default=False,
                   help="Keep warmup mp4 (default: off)")

    p.add_argument("--run_name", type=str, default="imf1step_trt",   # ditto10step_trt mf1step_trt imf1step_trt
               help="Optional run folder name (default: auto like 'offline_dit10step')")


    return p


def main():
    args = build_argparser().parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
