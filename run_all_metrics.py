"""
Run All Metrics
================
Comprehensive evaluation script that runs all metrics on generated videos.

사용법:
    # 단일 비디오 평가
    python run_all_metrics.py -g output.mp4 -s source.png
    
    # 배치 평가 (test set 전체)
    python run_all_metrics.py --batch \
        --results_dir /workspace/ditto/ditto-talkinghead-train/results/model_name \
        --test_dir /workspace/ditto/ditto-talkinghead-train/example/test

생성된 파일 구조
/workspace/ditto/ditto-talkinghead-train/metrics/
├── __init__.py          # 패키지 초기화
├── fid.py               # FID (Fréchet Inception Distance)
├── fvd.py               # FVD (Fréchet Video Distance)
├── csim.py              # CSIM (Cosine Similarity)
├── sync.py              # Sync-C, Sync-D (Lip Sync 메트릭)
├── rtf.py               # RTF (Real-Time Factor)
├── ffd.py               # FFD (First-Frame Delay)
├── run_all_metrics.py   # 모든 메트릭 통합 실행
└── requirements.txt     # 의존성 패키지
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.fid import FIDCalculator
from metrics.fvd import FVDCalculator
from metrics.csim import CSIMCalculator
from metrics.sync import SyncCalculator
from metrics.rtf import RTFCalculator
from metrics.ffd import FFDCalculator


class MetricsEvaluator:
    """
    Comprehensive metrics evaluator for talking head videos.
    
    Runs all available metrics and generates a report.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = {}
        
        # Initialize calculators
        self._init_calculators()
    
    def _init_calculators(self):
        """Initialize all metric calculators."""
        print("Initializing metric calculators...")
        
        try:
            self.fid_calc = FIDCalculator(device=self.device)
            print("  ✓ FID Calculator")
        except Exception as e:
            print(f"  ✗ FID Calculator: {e}")
            self.fid_calc = None
        
        try:
            self.fvd_calc = FVDCalculator(device=self.device)
            print("  ✓ FVD Calculator")
        except Exception as e:
            print(f"  ✗ FVD Calculator: {e}")
            self.fvd_calc = None
        
        try:
            self.csim_calc = CSIMCalculator(device=self.device)
            print("  ✓ CSIM Calculator")
        except Exception as e:
            print(f"  ✗ CSIM Calculator: {e}")
            self.csim_calc = None
        
        try:
            self.sync_calc = SyncCalculator(device=self.device)
            print("  ✓ Sync Calculator")
        except Exception as e:
            print(f"  ✗ Sync Calculator: {e}")
            self.sync_calc = None
        
        self.rtf_calc = RTFCalculator()
        print("  ✓ RTF Calculator")
        
        self.ffd_calc = FFDCalculator()
        print("  ✓ FFD Calculator")
    
    def evaluate_quality(self, generated_video: str, real_video: str,
                         max_frames: int = None, skip_fvd: bool = False) -> Dict[str, Any]:
        """
        Evaluate video quality metrics (FID, FVD).
        
        Args:
            generated_video: Path to generated video
            real_video: Path to real/ground-truth video
            max_frames: Maximum frames to use
            skip_fvd: Skip FVD (calculated at dataset level in batch mode)
            
        Returns:
            dict with FID and FVD scores
        """
        quality_results = {}
        
        # FID (per-video is fine)
        if self.fid_calc is not None:
            try:
                print("\n[FID] Calculating Fréchet Inception Distance...")
                fid_score = self.fid_calc.calculate(
                    generated_video, real_video, max_frames=max_frames
                )
                quality_results['fid'] = fid_score
                print(f"[FID] Score: {fid_score:.4f}")
            except Exception as e:
                print(f"[FID] Error: {e}")
                quality_results['fid_error'] = str(e)
        
        # FVD (skip for batch mode - calculated at dataset level)
        if not skip_fvd and self.fvd_calc is not None:
            try:
                print("\n[FVD] Calculating Fréchet Video Distance (per-video)...")
                fvd_score = self.fvd_calc.calculate(generated_video, real_video)
                quality_results['fvd'] = fvd_score
                print(f"[FVD] Score: {fvd_score:.4f}")
            except Exception as e:
                print(f"[FVD] Error: {e}")
                quality_results['fvd_error'] = str(e)
        
        return quality_results
    
    def evaluate_identity(self, generated_video: str, source: str,
                          max_frames: int = None) -> Dict[str, Any]:
        """
        Evaluate identity preservation (CSIM).
        
        Args:
            generated_video: Path to generated video
            source: Path to source image/video
            max_frames: Maximum frames to use
            
        Returns:
            dict with CSIM metrics
        """
        identity_results = {}
        
        if self.csim_calc is not None:
            try:
                print("\n[CSIM] Calculating Cosine Similarity...")
                csim_result = self.csim_calc.calculate(
                    generated_video, source, max_frames=max_frames
                )
                identity_results['csim_mean'] = csim_result['mean_csim']
                identity_results['csim_std'] = csim_result['std_csim']
                identity_results['csim_min'] = csim_result['min_csim']
                identity_results['csim_max'] = csim_result['max_csim']
                print(f"[CSIM] Mean: {csim_result['mean_csim']:.4f} (± {csim_result['std_csim']:.4f})")
            except Exception as e:
                print(f"[CSIM] Error: {e}")
                identity_results['csim_error'] = str(e)
        
        return identity_results
    
    def evaluate_sync(self, generated_video: str,
                      max_windows: int = None) -> Dict[str, Any]:
        """
        Evaluate lip synchronization (Sync-C, Sync-D).
        
        Args:
            generated_video: Path to generated video with audio
            max_windows: Maximum windows to process
            
        Returns:
            dict with sync metrics
        """
        sync_results = {}
        
        if self.sync_calc is not None:
            try:
                print("\n[Sync] Calculating Lip Sync metrics...")
                sync_result = self.sync_calc.calculate(
                    generated_video, max_windows=max_windows
                )
                sync_results['sync_c'] = sync_result['sync_c']
                sync_results['sync_c_std'] = sync_result['sync_c_std']
                sync_results['sync_d'] = sync_result['sync_d']
                sync_results['sync_d_std'] = sync_result['sync_d_std']
                print(f"[Sync-C] Score: {sync_result['sync_c']:.4f} (± {sync_result['sync_c_std']:.4f})")
                print(f"[Sync-D] Score: {sync_result['sync_d']:.4f} (± {sync_result['sync_d_std']:.4f})")
            except Exception as e:
                print(f"[Sync] Error: {e}")
                sync_results['sync_error'] = str(e)
        
        return sync_results
    
    def evaluate_all(self, generated_video: str,
                     source: str = None,
                     real_video: str = None,
                     audio: str = None,
                     max_frames: int = None,
                     max_windows: int = None,
                     skip_fvd: bool = False) -> Dict[str, Any]:
        """
        Run all applicable metrics.
        
        Args:
            generated_video: Path to generated video
            source: Path to source image (for CSIM)
            real_video: Path to real video (for FID/FVD)
            audio: Path to audio file (optional, uses video audio if not provided)
            max_frames: Maximum frames for frame-based metrics
            max_windows: Maximum windows for sync metrics
            skip_fvd: Skip FVD calculation (for batch mode, dataset-level FVD is computed separately)
            
        Returns:
            Comprehensive results dict
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'generated_video': generated_video,
            'source': source,
            'real_video': real_video,
            'audio': audio,
            'device': self.device,
            'metrics': {}
        }
        
        # Quality metrics (need real video)
        if real_video and os.path.exists(real_video):
            quality = self.evaluate_quality(generated_video, real_video, max_frames, skip_fvd=skip_fvd)
            results['metrics'].update(quality)
        else:
            if not skip_fvd:
                print("\n[Quality] Skipping FID/FVD (no real video provided)")
        
        # Identity metrics (need source)
        if source and os.path.exists(source):
            identity = self.evaluate_identity(generated_video, source, max_frames)
            results['metrics'].update(identity)
        else:
            print("\n[Identity] Skipping CSIM (no source provided)")
        
        # Sync metrics
        sync = self.evaluate_sync(generated_video, max_windows)
        results['metrics'].update(sync)
        
        # Video info
        try:
            import cv2
            cap = cv2.VideoCapture(generated_video)
            results['video_info'] = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
            cap.release()
        except:
            pass
        
        self.results = results
        return results
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self):
        """Print summary of results."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 60)
        print("                  EVALUATION SUMMARY")
        print("=" * 60)
        
        metrics = self.results.get('metrics', {})
        
        # Quality metrics
        print("\n📊 Quality Metrics:")
        if 'fid' in metrics:
            print(f"   FID  : {metrics['fid']:.4f} (lower is better)")
        if 'fvd' in metrics:
            print(f"   FVD  : {metrics['fvd']:.4f} (lower is better)")
        
        # Identity metrics
        print("\n👤 Identity Preservation:")
        if 'csim_mean' in metrics:
            print(f"   CSIM : {metrics['csim_mean']:.4f} ± {metrics['csim_std']:.4f} (higher is better)")
        
        # Sync metrics
        print("\n🔊 Lip Synchronization:")
        if 'sync_c' in metrics:
            print(f"   Sync-C: {metrics['sync_c']:.4f} ± {metrics['sync_c_std']:.4f} (higher is better)")
        if 'sync_d' in metrics:
            print(f"   Sync-D: {metrics['sync_d']:.4f} ± {metrics['sync_d_std']:.4f} (lower is better)")
        
        # Performance metrics
        print("\n⚡ Performance:")
        if 'rtf_mean' in metrics:
            print(f"   RTF  : {metrics['rtf_mean']:.4f} (< 1.0 = real-time)")
        if 'ffd_mean_ms' in metrics:
            print(f"   FFD  : {metrics['ffd_mean_ms']:.2f}ms")
        
        print("\n" + "=" * 60)


class BatchMetricsEvaluator:
    """
    Batch evaluation for multiple test samples.
    
    Evaluates all clips in a test directory and computes aggregate statistics.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.evaluator = MetricsEvaluator(device=device)
        self.all_results = []
        self.aggregate_results = {}
    
    def get_test_clips(self, test_dir: str) -> List[Dict[str, str]]:
        """Get list of test clips with their audio and image paths."""
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
    
    def evaluate_batch(self, results_dir: str, test_dir: str,
                       real_videos_dir: str = None,
                       max_frames: int = None,
                       max_windows: int = None) -> Dict[str, Any]:
        """
        Evaluate all clips in batch.
        
        Args:
            results_dir: Directory containing generated videos (clip_name.mp4)
            test_dir: Directory containing test clips (clip_name/audio.wav, image.png)
            real_videos_dir: Optional directory containing real videos for FID/FVD
            max_frames: Maximum frames for frame-based metrics
            max_windows: Maximum windows for sync metrics
            
        Returns:
            Aggregate results dict
        """
        # Get test clips
        test_clips = self.get_test_clips(test_dir)
        print(f"\nFound {len(test_clips)} test clips")
        
        if len(test_clips) == 0:
            print("[ERROR] No test clips found!")
            return {}
        
        # Collect video paths for dataset-level FVD
        generated_videos = []
        real_videos = []
        
        # Process each clip for per-video metrics (CSIM, Sync, FID)
        self.all_results = []
        failed_clips = []
        
        for clip in tqdm(test_clips, desc="Evaluating per-clip metrics"):
            clip_name = clip["name"]
            generated_video = os.path.join(results_dir, f"{clip_name}.mp4")
            
            # Check if generated video exists
            if not os.path.exists(generated_video):
                print(f"\n[SKIP] {clip_name}: Generated video not found")
                failed_clips.append(clip_name)
                continue
            
            generated_videos.append(generated_video)
            
            # Optional: real video for FVD
            real_video = None
            if real_videos_dir:
                # Try exact match first (clip_name.mp4)
                real_video_path = os.path.join(real_videos_dir, f"{clip_name}.mp4")
                if not os.path.exists(real_video_path):
                    # If not found, try removing suffix (e.g., Jae-in_000 -> Jae-in)
                    # This handles cases where generated videos have _000 suffix but real videos don't
                    base_name = clip_name.rsplit('_', 1)[0] if '_' in clip_name else clip_name
                    real_video_path = os.path.join(real_videos_dir, f"{base_name}.mp4")
                
                if os.path.exists(real_video_path):
                    real_video = real_video_path
                    real_videos.append(real_video_path)
            
            try:
                # Run per-clip evaluation (CSIM, Sync, FID - skip FVD here)
                result = self.evaluator.evaluate_all(
                    generated_video=generated_video,
                    source=clip["source_path"],
                    real_video=real_video,
                    audio=clip["audio_path"],
                    max_frames=max_frames,
                    max_windows=max_windows,
                    skip_fvd=True  # Skip per-video FVD, will calculate dataset-level
                )
                result['clip_name'] = clip_name
                self.all_results.append(result)
            except Exception as e:
                print(f"\n[ERROR] {clip_name}: {e}")
                failed_clips.append(clip_name)
        
        # Compute aggregate statistics for per-clip metrics
        self._compute_aggregate_stats()
        
        # Calculate dataset-level FVD (standard method)
        if real_videos_dir and len(generated_videos) > 0 and len(real_videos) > 0:
            try:
                print("\n" + "="*50)
                print("Calculating Dataset-level FVD (standard method)...")
                print("="*50)
                fvd_score = self.evaluator.fvd_calc.calculate_dataset_fvd(
                    generated_videos=generated_videos,
                    real_videos=real_videos,
                    max_frames=max_frames
                )
                self.aggregate_results['dataset_fvd'] = fvd_score
                print(f"[FVD] Dataset-level FVD: {fvd_score:.2f}")
            except Exception as e:
                print(f"[FVD] Error calculating dataset-level FVD: {e}")
                self.aggregate_results['dataset_fvd_error'] = str(e)
        else:
            print("\n[FVD] Skipping dataset-level FVD (no real videos directory provided)")
        
        # Add metadata
        self.aggregate_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'results_dir': results_dir,
            'test_dir': test_dir,
            'real_videos_dir': real_videos_dir,
            'total_clips': len(test_clips),
            'successful_clips': len(self.all_results),
            'failed_clips': failed_clips,
            'device': self.device,
        }
        
        return self.aggregate_results
    
    def _compute_aggregate_stats(self):
        """Compute mean and std of all metrics across clips."""
        if not self.all_results:
            return
        
        # Collect all metric values
        metric_values = {}
        for result in self.all_results:
            metrics = result.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not key.endswith('_error'):
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(value)
        
        # Compute mean and std
        aggregate = {}
        for key, values in metric_values.items():
            if len(values) > 0:
                arr = np.array(values)
                aggregate[key] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'count': len(values)
                }
        
        self.aggregate_results['aggregate_metrics'] = aggregate
        self.aggregate_results['per_clip_results'] = self.all_results
    
    def print_summary(self):
        """Print summary of aggregate results."""
        if not self.aggregate_results:
            print("No results to display")
            return
        
        metadata = self.aggregate_results.get('metadata', {})
        aggregate = self.aggregate_results.get('aggregate_metrics', {})
        
        print("\n" + "=" * 70)
        print("                    BATCH EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\n📁 Results Directory: {metadata.get('results_dir', 'N/A')}")
        print(f"📂 Test Directory: {metadata.get('test_dir', 'N/A')}")
        print(f"✅ Successful: {metadata.get('successful_clips', 0)}/{metadata.get('total_clips', 0)} clips")
        
        if metadata.get('failed_clips'):
            print(f"❌ Failed: {len(metadata['failed_clips'])} clips")
        
        # Quality metrics
        print("\n📊 Quality Metrics:")
        if 'fid' in aggregate:
            m = aggregate['fid']
            print(f"   FID   : {m['mean']:.4f} ± {m['std']:.4f} (lower is better, per-video avg)")
        
        # Dataset-level FVD (standard method)
        dataset_fvd = self.aggregate_results.get('dataset_fvd')
        if dataset_fvd is not None:
            print(f"   FVD   : {dataset_fvd:.2f} (lower is better, dataset-level)")
        elif 'dataset_fvd_error' in self.aggregate_results:
            print(f"   FVD   : Error - {self.aggregate_results['dataset_fvd_error']}")
        
        # Identity metrics
        print("\n👤 Identity Preservation (mean ± std):")
        if 'csim_mean' in aggregate:
            m = aggregate['csim_mean']
            print(f"   CSIM  : {m['mean']:.4f} ± {m['std']:.4f} (higher is better)")
        
        # Sync metrics
        print("\n🔊 Lip Synchronization (mean ± std):")
        if 'sync_c' in aggregate:
            m = aggregate['sync_c']
            print(f"   Sync-C: {m['mean']:.4f} ± {m['std']:.4f} (higher is better)")
        if 'sync_d' in aggregate:
            m = aggregate['sync_d']
            print(f"   Sync-D: {m['mean']:.4f} ± {m['std']:.4f} (lower is better)")
        
        print("\n" + "=" * 70)
        
        # Per-clip summary table (first 10)
        if self.all_results:
            print("\n📋 Per-Clip Results (first 10):")
            print("-" * 70)
            print(f"{'Clip Name':<30} {'CSIM':>10} {'Sync-C':>10} {'Sync-D':>10}")
            print("-" * 70)
            
            for result in self.all_results[:10]:
                clip_name = result.get('clip_name', 'Unknown')
                metrics = result.get('metrics', {})
                csim = metrics.get('csim_mean', float('nan'))
                sync_c = metrics.get('sync_c', float('nan'))
                sync_d = metrics.get('sync_d', float('nan'))
                print(f"{clip_name:<30} {csim:>10.4f} {sync_c:>10.4f} {sync_d:>10.4f}")
            
            if len(self.all_results) > 10:
                print(f"... and {len(self.all_results) - 10} more clips")
            print("-" * 70)
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.aggregate_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all evaluation metrics on generated talking head video(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video evaluation
  python run_all_metrics.py -g output.mp4 -s source.png

  # Batch evaluation (test set)
  python run_all_metrics.py --batch \\
      --results_dir /workspace/ditto/ditto-talkinghead-train/results/model_name \\
      --test_dir /workspace/ditto/ditto-talkinghead-train/example/test
      
  # Batch evaluation with real videos (for FID/FVD)
  python run_all_metrics.py --batch \\
      --results_dir results/model_name \\
      --test_dir example/test \\
      --real_videos_dir /path/to/real_videos
        """
    )
    
    # Mode selection
    parser.add_argument("--batch", action="store_true", default=True,
                       help="Run batch evaluation on test set")
    
    # Single video mode arguments
    parser.add_argument("--generated", "-g", type=str, default=None,
                       help="Path to generated video (single mode)")
    parser.add_argument("--source", "-s", type=str, default=None,
                       help="Path to source image (for CSIM, single mode)")
    parser.add_argument("--real", "-r", type=str, default=None,
                       help="Path to real/ground-truth video (for FID/FVD, single mode)")
    parser.add_argument("--audio", "-a", type=str, default=None,
                       help="Path to audio file (optional, single mode)")
    
    # Batch mode arguments
    parser.add_argument("--results_dir", type=str, default="/workspace/ditto/ditto-talkinghead-train/testset_results/mf_100epoch",
                       help="Directory containing generated(inference_metric.py) videos (batch mode)")
    parser.add_argument("--test_dir", type=str, 
                       default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/testset",
                       help="Directory containing test clips (batch mode)")
    parser.add_argument("--real_videos_dir", type=str, default="/workspace/ditto/datasets/Talk8/SUBSET_Talk8/512x512",
                       help="Directory containing Ground-truth real videos for FID/FVD (batch mode)")
    
    # Common arguments
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Path to save results JSON")
    parser.add_argument("--max_seconds", type=float, default=10.0,
                       help="Maximum seconds to process (default: 10.0 for faster evaluation)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process (overrides max_seconds if set)")
    parser.add_argument("--max_windows", type=int, default=None,
                       help="Maximum windows for sync metrics")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--fps", type=int, default=25,
                       help="Video FPS for max_seconds calculation (default: 25)")
    
    args = parser.parse_args()
    
    # Convert max_seconds to max_frames and max_windows if not explicitly set
    if args.max_seconds is not None:
        if args.max_frames is None:
            args.max_frames = int(args.max_seconds * args.fps)
        if args.max_windows is None:
            # Sync uses window_size=5 frames, so max_windows = max_frames / 5
            args.max_windows = int(args.max_frames / 5)
        print(f"[INFO] Using first {args.max_seconds}s: {args.max_frames} frames, {args.max_windows} windows @ {args.fps}fps")
    
    if args.batch:
        # ============ Batch Mode ============
        if not args.results_dir:
            print("Error: --results_dir is required for batch mode")
            sys.exit(1)
        
        if not os.path.exists(args.results_dir):
            print(f"Error: Results directory not found: {args.results_dir}")
            sys.exit(1)
        
        if not os.path.exists(args.test_dir):
            print(f"Error: Test directory not found: {args.test_dir}")
            sys.exit(1)
        
        print("=" * 70)
        print("              BATCH METRICS EVALUATION")
        print("=" * 70)
        print(f"Results Dir: {args.results_dir}")
        print(f"Test Dir: {args.test_dir}")
        if args.real_videos_dir:
            print(f"Real Videos Dir: {args.real_videos_dir}")
        if args.max_seconds:
            print(f"Max Duration: {args.max_seconds}s ({args.max_frames} frames)")
        print("=" * 70)
        
        # Run batch evaluation
        evaluator = BatchMetricsEvaluator(device=args.device)
        results = evaluator.evaluate_batch(
            results_dir=args.results_dir,
            test_dir=args.test_dir,
            real_videos_dir=args.real_videos_dir,
            max_frames=args.max_frames,
            max_windows=args.max_windows
        )
        
        # Print summary
        evaluator.print_summary()
        
        # Save results
        if args.output:
            output_path = args.output
        else:
            # Auto-generate output path in results directory
            results_path = Path(args.results_dir)
            output_path = results_path / "batch_metrics.json"
        
        evaluator.save_results(str(output_path))
        
    else:
        # ============ Single Mode ============
        if not args.generated:
            print("Error: --generated is required for single mode (or use --batch for batch mode)")
            sys.exit(1)
        
        # Check input file
        if not os.path.exists(args.generated):
            print(f"Error: Generated video not found: {args.generated}")
            sys.exit(1)
        
        # Run evaluation
        evaluator = MetricsEvaluator(device=args.device)
        results = evaluator.evaluate_all(
            generated_video=args.generated,
            source=args.source,
            real_video=args.real,
            audio=args.audio,
            max_frames=args.max_frames,
            max_windows=args.max_windows
        )
        
        # Print summary
        evaluator.print_summary()
        
        # Save results
        if args.output:
            evaluator.save_results(args.output)
        else:
            # Auto-generate output path
            gen_path = Path(args.generated)
            output_path = gen_path.parent / f"{gen_path.stem}_metrics.json"
            evaluator.save_results(str(output_path))


if __name__ == "__main__":
    main()

