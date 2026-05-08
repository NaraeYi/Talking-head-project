"""
Run All Metrics
================
Comprehensive evaluation script that runs all metrics on generated talking-head videos.

Key evaluation notes
--------------------
- FID is reported as a per-video average over frame features.
- FVD is reported as a dataset-level metric over matched generated/real video sets.
- E-FID is reported as a dataset-level metric over expression-coefficient distributions.
- HeadPose is a generated-video diversity proxy: higher pitch/yaw/roll std/range
  means more pose variation, not necessarily better visual quality.
- For paper-style talking-head comparisons, recent works commonly use FVD72 on MEAD
  and FVD128 on HDTF. Use `--metric_preset mead_paper` or `--metric_preset hdtf_paper`
  to reproduce those truncation conventions for FVD / E-FID.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.fid import FIDCalculator
from metrics.fvd import FVDCalculator
from metrics.efid import EFIDCalculator
from metrics.csim import CSIMCalculator
from metrics.sync import SyncCalculator
from metrics.headpose import HeadPoseDiversityCalculator
from metrics.rtf import RTFCalculator
from metrics.ffd import FFDCalculator


def resolve_metric_preset(metric_preset: str, fvd_num_frames: Optional[int], efid_num_frames: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if metric_preset == 'mead_paper':
        return fvd_num_frames if fvd_num_frames is not None else 72, efid_num_frames if efid_num_frames is not None else 72
    if metric_preset == 'hdtf_paper':
        return fvd_num_frames if fvd_num_frames is not None else 128, efid_num_frames if efid_num_frames is not None else 128
    return fvd_num_frames, efid_num_frames


def resolve_headpose_gpu_id(device: str, explicit_gpu_id: Optional[int]) -> Optional[int]:
    if explicit_gpu_id is not None:
        return explicit_gpu_id
    if not device:
        return None
    if device.startswith('cpu'):
        return -1
    if device.startswith('cuda'):
        if ':' in device:
            try:
                return int(device.split(':', 1)[1])
            except ValueError:
                return 0
        return 0
    return None


class MetricsEvaluator:
    """Comprehensive metrics evaluator for talking-head videos."""

    def __init__(self, device: str = 'cuda', fvd_kwargs: Optional[Dict[str, Any]] = None, efid_kwargs: Optional[Dict[str, Any]] = None, headpose_kwargs: Optional[Dict[str, Any]] = None, enable_fvd: bool = True, enable_headpose: bool = True, enable_efid: bool = True):
        self.device = device
        self.results: Dict[str, Any] = {}
        self.fvd_kwargs = fvd_kwargs or {}
        self.efid_kwargs = efid_kwargs or {}
        self.headpose_kwargs = headpose_kwargs or {}
        self.enable_fvd = enable_fvd
        self.enable_headpose = enable_headpose
        self.enable_efid = enable_efid
        self._init_calculators()

    def _init_calculators(self):
        print('Initializing metric calculators...')

        try:
            self.fid_calc = FIDCalculator(device=self.device)
            print('  ✓ FID Calculator')
        except Exception as exc:
            print(f'  ✗ FID Calculator: {exc}')
            self.fid_calc = None

        if self.enable_fvd:
            try:
                self.fvd_calc = FVDCalculator(device=self.device, **self.fvd_kwargs)
                print('  ✓ FVD Calculator')
            except Exception as exc:
                print(f'  ✗ FVD Calculator: {exc}')
                self.fvd_calc = None
        else:
            self.fvd_calc = None
            print('  - FVD Calculator skipped')

        if self.enable_efid:
            try:
                self.efid_calc = EFIDCalculator(device=self.device, **self.efid_kwargs)
                print('  ✓ E-FID Calculator')
            except Exception as exc:
                print(f'  ✗ E-FID Calculator: {exc}')
                self.efid_calc = None
        else:
            self.efid_calc = None
            print('  - E-FID Calculator skipped')

        try:
            self.csim_calc = CSIMCalculator(device=self.device)
            print('  ✓ CSIM Calculator')
        except Exception as exc:
            print(f'  ✗ CSIM Calculator: {exc}')
            self.csim_calc = None

        try:
            self.sync_calc = SyncCalculator(device=self.device)
            print('  ✓ Sync Calculator')
        except Exception as exc:
            print(f'  ✗ Sync Calculator: {exc}')
            self.sync_calc = None

        if self.enable_headpose:
            try:
                self.headpose_calc = HeadPoseDiversityCalculator(**self.headpose_kwargs)
                print('  ✓ HeadPose Diversity Calculator')
            except Exception as exc:
                print(f'  ✗ HeadPose Diversity Calculator: {exc}')
                self.headpose_calc = None
        else:
            self.headpose_calc = None
            print('  - HeadPose Diversity Calculator skipped')

        self.rtf_calc = RTFCalculator()
        print('  ✓ RTF Calculator')
        self.ffd_calc = FFDCalculator()
        print('  ✓ FFD Calculator')

    def evaluate_quality(self, generated_video: str, real_video: str, max_frames: Optional[int] = None, skip_fvd: bool = False, skip_efid: bool = False, fvd_num_frames: Optional[int] = None, efid_num_frames: Optional[int] = None) -> Dict[str, Any]:
        quality_results: Dict[str, Any] = {}

        if self.fid_calc is not None:
            try:
                print('\n[FID] Calculating Fréchet Inception Distance...')
                fid_score = self.fid_calc.calculate(generated_video, real_video, max_frames=max_frames)
                quality_results['fid'] = fid_score
                print(f'[FID] Score: {fid_score:.4f}')
            except Exception as exc:
                print(f'[FID] Error: {exc}')
                quality_results['fid_error'] = str(exc)

        if not skip_fvd and self.fvd_calc is not None:
            try:
                print('\n[FVD] Calculating dataset-style single-pair FVD...')
                fvd_frames = fvd_num_frames if fvd_num_frames is not None else max_frames
                fvd_score = self.fvd_calc.calculate(generated_video, real_video, max_frames=fvd_frames)
                quality_results['fvd'] = fvd_score
                print(f'[FVD] Score: {fvd_score:.4f}')
            except Exception as exc:
                print(f'[FVD] Error: {exc}')
                quality_results['fvd_error'] = str(exc)

        if not skip_efid and self.efid_calc is not None:
            try:
                print('\n[E-FID] Calculating single-pair E-FID...')
                efid_frames = efid_num_frames if efid_num_frames is not None else max_frames
                efid_score = self.efid_calc.calculate(generated_video, real_video, max_frames=efid_frames)
                quality_results['efid'] = efid_score
                print(f'[E-FID] Score: {efid_score:.4f}')
            except Exception as exc:
                print(f'[E-FID] Error: {exc}')
                quality_results['efid_error'] = str(exc)

        return quality_results

    def evaluate_identity(self, generated_video: str, source: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
        identity_results: Dict[str, Any] = {}
        if self.csim_calc is not None:
            try:
                print('\n[CSIM] Calculating Cosine Similarity...')
                csim_result = self.csim_calc.calculate(generated_video, source, max_frames=max_frames)
                identity_results['csim_mean'] = csim_result['mean_csim']
                identity_results['csim_std'] = csim_result['std_csim']
                identity_results['csim_min'] = csim_result['min_csim']
                identity_results['csim_max'] = csim_result['max_csim']
                print(f"[CSIM] Mean: {csim_result['mean_csim']:.4f} (± {csim_result['std_csim']:.4f})")
            except Exception as exc:
                print(f'[CSIM] Error: {exc}')
                identity_results['csim_error'] = str(exc)
        return identity_results

    def evaluate_sync(self, generated_video: str, max_windows: Optional[int] = None) -> Dict[str, Any]:
        sync_results: Dict[str, Any] = {}
        if self.sync_calc is not None:
            try:
                print('\n[Sync] Calculating Lip Sync metrics...')
                sync_result = self.sync_calc.calculate(generated_video, max_windows=max_windows)
                sync_results['sync_c'] = sync_result['sync_c']
                sync_results['sync_c_std'] = sync_result['sync_c_std']
                sync_results['sync_d'] = sync_result['sync_d']
                sync_results['sync_d_std'] = sync_result['sync_d_std']
                print(f"[Sync-C] Score: {sync_result['sync_c']:.4f} (± {sync_result['sync_c_std']:.4f})")
                print(f"[Sync-D] Score: {sync_result['sync_d']:.4f} (± {sync_result['sync_d_std']:.4f})")
            except Exception as exc:
                print(f'[Sync] Error: {exc}')
                sync_results['sync_error'] = str(exc)
        return sync_results

    def evaluate_headpose(self, generated_video: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
        headpose_results: Dict[str, Any] = {}
        if self.headpose_calc is not None:
            try:
                print('\n[HeadPose] Calculating pitch/yaw/roll diversity...')
                headpose_result = self.headpose_calc.calculate(generated_video, max_frames=max_frames)
                headpose_results.update(headpose_result)
                print(
                    '[HeadPose] Std mean: '
                    f"{headpose_result['headpose_std_mean']:.4f} "
                    f"(pitch={headpose_result['headpose_pitch_std']:.4f}, "
                    f"yaw={headpose_result['headpose_yaw_std']:.4f}, "
                    f"roll={headpose_result['headpose_roll_std']:.4f})"
                )
            except Exception as exc:
                print(f'[HeadPose] Error: {exc}')
                headpose_results['headpose_error'] = str(exc)
        return headpose_results

    def evaluate_all(self, generated_video: str, source: Optional[str] = None, real_video: Optional[str] = None, audio: Optional[str] = None, max_frames: Optional[int] = None, max_windows: Optional[int] = None, skip_fvd: bool = False, skip_efid: bool = False, fvd_num_frames: Optional[int] = None, efid_num_frames: Optional[int] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {'timestamp': datetime.now().isoformat(), 'generated_video': generated_video, 'source': source, 'real_video': real_video, 'audio': audio, 'device': self.device, 'metrics': {}}

        if real_video and os.path.exists(real_video):
            quality = self.evaluate_quality(generated_video, real_video, max_frames=max_frames, skip_fvd=skip_fvd, skip_efid=skip_efid, fvd_num_frames=fvd_num_frames, efid_num_frames=efid_num_frames)
            results['metrics'].update(quality)
        else:
            if not skip_fvd or not skip_efid:
                print('\n[Quality] Skipping FID/FVD/E-FID (no real video provided)')

        if source and os.path.exists(source):
            identity = self.evaluate_identity(generated_video, source, max_frames=max_frames)
            results['metrics'].update(identity)
        else:
            print('\n[Identity] Skipping CSIM (no source provided)')

        sync = self.evaluate_sync(generated_video, max_windows=max_windows)
        results['metrics'].update(sync)

        headpose = self.evaluate_headpose(generated_video, max_frames=max_frames)
        results['metrics'].update(headpose)

        try:
            import cv2
            cap = cv2.VideoCapture(generated_video)
            results['video_info'] = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            cap.release()
        except Exception:
            pass

        self.results = results
        return results

    def save_results(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f'\nResults saved to: {output_path}')

    def print_summary(self):
        if not self.results:
            print('No results to display')
            return

        print('\n' + '=' * 60)
        print('                  EVALUATION SUMMARY')
        print('=' * 60)
        metrics = self.results.get('metrics', {})
        print('\n📊 Quality Metrics:')
        if 'fid' in metrics:
            print(f"   FID   : {metrics['fid']:.4f} (lower is better)")
        if 'fvd' in metrics:
            print(f"   FVD   : {metrics['fvd']:.4f} (lower is better)")
        if 'efid' in metrics:
            print(f"   E-FID : {metrics['efid']:.4f} (lower is better)")
        print('\n👤 Identity Preservation:')
        if 'csim_mean' in metrics:
            print(f"   CSIM  : {metrics['csim_mean']:.4f} ± {metrics['csim_std']:.4f} (higher is better)")
        print('\n🔊 Lip Synchronization:')
        if 'sync_c' in metrics:
            print(f"   Sync-C: {metrics['sync_c']:.4f} ± {metrics['sync_c_std']:.4f} (higher is better)")
        if 'sync_d' in metrics:
            print(f"   Sync-D: {metrics['sync_d']:.4f} ± {metrics['sync_d_std']:.4f} (lower is better)")
        print('\n🧭 Head Pose Diversity:')
        if 'headpose_std_mean' in metrics:
            print(f"   HP-Std: {metrics['headpose_std_mean']:.4f} deg avg (higher = more pose diversity)")
        print('\n' + '=' * 60)


class BatchMetricsEvaluator:
    """Batch evaluation for a test directory."""

    def __init__(self, device: str = 'cuda', fvd_kwargs: Optional[Dict[str, Any]] = None, efid_kwargs: Optional[Dict[str, Any]] = None, headpose_kwargs: Optional[Dict[str, Any]] = None, enable_fvd: bool = True, enable_headpose: bool = True, enable_efid: bool = True):
        self.device = device
        self.evaluator = MetricsEvaluator(device=device, fvd_kwargs=fvd_kwargs, efid_kwargs=efid_kwargs, headpose_kwargs=headpose_kwargs, enable_fvd=enable_fvd, enable_headpose=enable_headpose, enable_efid=enable_efid)
        self.all_results: List[Dict[str, Any]] = []
        self.aggregate_results: Dict[str, Any] = {}

    def get_test_clips(self, test_dir: str) -> List[Dict[str, str]]:
        clips: List[Dict[str, str]] = []
        for clip_name in sorted(os.listdir(test_dir)):
            clip_path = os.path.join(test_dir, clip_name)
            if os.path.isdir(clip_path):
                audio_path = os.path.join(clip_path, 'audio.wav')
                image_path = os.path.join(clip_path, 'image.png')
                if os.path.exists(audio_path) and os.path.exists(image_path):
                    clips.append({'name': clip_name, 'audio_path': audio_path, 'source_path': image_path})
        return clips

    @staticmethod
    def _match_real_video(real_videos_dir: str, clip_name: str) -> Optional[str]:
        real_video_path = os.path.join(real_videos_dir, f'{clip_name}.mp4')
        if os.path.exists(real_video_path):
            return real_video_path
        base_name = clip_name.rsplit('_', 1)[0] if '_' in clip_name else clip_name
        real_video_path = os.path.join(real_videos_dir, f'{base_name}.mp4')
        if os.path.exists(real_video_path):
            return real_video_path
        return None

    def evaluate_batch(self, results_dir: str, test_dir: str, real_videos_dir: Optional[str] = None, max_frames: Optional[int] = None, max_windows: Optional[int] = None, fvd_num_frames: Optional[int] = None, efid_num_frames: Optional[int] = None, skip_dataset_fid: bool = False) -> Dict[str, Any]:
        test_clips = self.get_test_clips(test_dir)
        print(f'\nFound {len(test_clips)} test clips')
        if len(test_clips) == 0:
            print('[ERROR] No test clips found!')
            return {}

        paired_generated_videos: List[str] = []
        paired_real_videos: List[str] = []
        self.all_results = []
        failed_clips: List[str] = []

        for clip in tqdm(test_clips, desc='Evaluating per-clip metrics'):
            clip_name = clip['name']
            generated_video = os.path.join(results_dir, f'{clip_name}.mp4')
            if not os.path.exists(generated_video):
                print(f'\n[SKIP] {clip_name}: Generated video not found')
                failed_clips.append(clip_name)
                continue

            real_video = None
            if real_videos_dir:
                real_video = self._match_real_video(real_videos_dir, clip_name)
                if real_video is not None:
                    paired_generated_videos.append(generated_video)
                    paired_real_videos.append(real_video)

            try:
                result = self.evaluator.evaluate_all(generated_video=generated_video, source=clip['source_path'], real_video=real_video, audio=clip['audio_path'], max_frames=max_frames, max_windows=max_windows, skip_fvd=True, skip_efid=True, fvd_num_frames=fvd_num_frames, efid_num_frames=efid_num_frames)
                result['clip_name'] = clip_name
                self.all_results.append(result)
            except Exception as exc:
                print(f'\n[ERROR] {clip_name}: {exc}')
                failed_clips.append(clip_name)

        self._compute_aggregate_stats()

        if real_videos_dir and len(paired_generated_videos) > 1 and len(paired_real_videos) > 1:
            print('\n' + '=' * 50)
            print('Calculating dataset-level metrics on matched generated/real sets...')
            print('=' * 50)
            print(f'[PAIRING] Using {len(paired_generated_videos)} matched video pairs')

            if not skip_dataset_fid and self.evaluator.fid_calc is not None:
                try:
                    fid_score = self.evaluator.fid_calc.calculate_dataset_fid(generated_videos=paired_generated_videos, real_videos=paired_real_videos, max_frames=max_frames)
                    self.aggregate_results['dataset_fid'] = fid_score
                    print(f'[FID] Dataset-level FID: {fid_score:.4f}')
                except Exception as exc:
                    print(f'[FID] Error calculating dataset-level FID: {exc}')
                    self.aggregate_results['dataset_fid_error'] = str(exc)

            if self.evaluator.fvd_calc is not None:
                try:
                    fvd_score = self.evaluator.fvd_calc.calculate_dataset_fvd(generated_videos=paired_generated_videos, real_videos=paired_real_videos, max_frames=fvd_num_frames)
                    self.aggregate_results['dataset_fvd'] = fvd_score
                    print(f'[FVD] Dataset-level FVD: {fvd_score:.2f}')
                except Exception as exc:
                    print(f'[FVD] Error calculating dataset-level FVD: {exc}')
                    self.aggregate_results['dataset_fvd_error'] = str(exc)

            if self.evaluator.efid_calc is not None:
                try:
                    efid_score = self.evaluator.efid_calc.calculate_dataset_efid(generated_videos=paired_generated_videos, real_videos=paired_real_videos, max_frames=efid_num_frames)
                    self.aggregate_results['dataset_efid'] = efid_score
                    print(f'[E-FID] Dataset-level E-FID: {efid_score:.4f}')
                except Exception as exc:
                    print(f'[E-FID] Error calculating dataset-level E-FID: {exc}')
                    self.aggregate_results['dataset_efid_error'] = str(exc)
        else:
            print('\n[FVD/E-FID] Skipping dataset-level metrics (need >=2 matched generated/real videos)')

        self.aggregate_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'results_dir': results_dir,
            'test_dir': test_dir,
            'real_videos_dir': real_videos_dir,
            'total_clips': len(test_clips),
            'successful_clips': len(self.all_results),
            'failed_clips': failed_clips,
            'paired_videos_for_set_metrics': len(paired_generated_videos),
            'device': self.device,
            'fvd_num_frames': fvd_num_frames,
            'fvd_frame_sampling': self.evaluator.fvd_kwargs.get('frame_sampling'),
            'efid_num_frames': efid_num_frames,
            'dataset_fid_enabled': not skip_dataset_fid,
        }
        return self.aggregate_results

    def _compute_aggregate_stats(self):
        if not self.all_results:
            return

        metric_values: Dict[str, List[float]] = {}
        for result in self.all_results:
            metrics = result.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not key.endswith('_error'):
                    metric_values.setdefault(key, []).append(float(value))

        aggregate: Dict[str, Dict[str, float]] = {}
        for key, values in metric_values.items():
            if values:
                arr = np.asarray(values, dtype=np.float64)
                aggregate[key] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr)), 'min': float(np.min(arr)), 'max': float(np.max(arr)), 'count': int(arr.size)}

        self.aggregate_results['aggregate_metrics'] = aggregate
        self.aggregate_results['per_clip_results'] = self.all_results

    def print_summary(self):
        if not self.aggregate_results:
            print('No results to display')
            return

        metadata = self.aggregate_results.get('metadata', {})
        aggregate = self.aggregate_results.get('aggregate_metrics', {})
        print('\n' + '=' * 70)
        print('                    BATCH EVALUATION SUMMARY')
        print('=' * 70)
        print(f"\n📁 Results Directory: {metadata.get('results_dir', 'N/A')}")
        print(f"📂 Test Directory: {metadata.get('test_dir', 'N/A')}")
        print(f"✅ Successful: {metadata.get('successful_clips', 0)}/{metadata.get('total_clips', 0)} clips")
        if metadata.get('failed_clips'):
            print(f"❌ Failed: {len(metadata['failed_clips'])} clips")

        print('\n📊 Quality Metrics:')
        if 'fid' in aggregate:
            m = aggregate['fid']
            print(f"   FID    : {m['mean']:.4f} ± {m['std']:.4f} (lower is better, per-video avg)")
        dataset_fid = self.aggregate_results.get('dataset_fid')
        if dataset_fid is not None:
            print(f'   FID-set: {dataset_fid:.4f} (lower is better, dataset-level)')
        elif 'dataset_fid_error' in self.aggregate_results:
            print(f"   FID-set: Error - {self.aggregate_results['dataset_fid_error']}")
        dataset_fvd = self.aggregate_results.get('dataset_fvd')
        if dataset_fvd is not None:
            print(f'   FVD    : {dataset_fvd:.2f} (lower is better, dataset-level)')
        elif 'dataset_fvd_error' in self.aggregate_results:
            print(f"   FVD    : Error - {self.aggregate_results['dataset_fvd_error']}")
        dataset_efid = self.aggregate_results.get('dataset_efid')
        if dataset_efid is not None:
            print(f'   E-FID  : {dataset_efid:.4f} (lower is better, dataset-level)')
        elif 'dataset_efid_error' in self.aggregate_results:
            print(f"   E-FID  : Error - {self.aggregate_results['dataset_efid_error']}")

        print('\n👤 Identity Preservation (mean ± std):')
        if 'csim_mean' in aggregate:
            m = aggregate['csim_mean']
            print(f"   CSIM   : {m['mean']:.4f} ± {m['std']:.4f} (higher is better)")

        print('\n🔊 Lip Synchronization (mean ± std):')
        if 'sync_c' in aggregate:
            m = aggregate['sync_c']
            print(f"   Sync-C : {m['mean']:.4f} ± {m['std']:.4f} (higher is better)")
        if 'sync_d' in aggregate:
            m = aggregate['sync_d']
            print(f"   Sync-D : {m['mean']:.4f} ± {m['std']:.4f} (lower is better)")

        print('\n🧭 Head Pose Diversity (mean ± std):')
        if 'headpose_std_mean' in aggregate:
            m = aggregate['headpose_std_mean']
            print(f"   HP-Std : {m['mean']:.4f} ± {m['std']:.4f} deg avg (higher = more pose diversity)")
        if 'headpose_yaw_std' in aggregate:
            pitch = aggregate.get('headpose_pitch_std', {}).get('mean', float('nan'))
            yaw = aggregate.get('headpose_yaw_std', {}).get('mean', float('nan'))
            roll = aggregate.get('headpose_roll_std', {}).get('mean', float('nan'))
            print(f"   Axis   : pitch={pitch:.4f}, yaw={yaw:.4f}, roll={roll:.4f}")

        print('\n' + '=' * 70)
        if self.all_results:
            print('\n📋 Per-Clip Results (first 10):')
            print('-' * 70)
            print(f"{'Clip Name':<30} {'FID':>10} {'CSIM':>10} {'Sync-C':>10} {'Sync-D':>10} {'HP-Std':>10}")
            print('-' * 70)
            for result in self.all_results[:10]:
                clip_name = result.get('clip_name', 'Unknown')
                metrics = result.get('metrics', {})
                fid = metrics.get('fid', float('nan'))
                csim = metrics.get('csim_mean', float('nan'))
                sync_c = metrics.get('sync_c', float('nan'))
                sync_d = metrics.get('sync_d', float('nan'))
                hp_std = metrics.get('headpose_std_mean', float('nan'))
                print(f'{clip_name:<30} {fid:>10.4f} {csim:>10.4f} {sync_c:>10.4f} {sync_d:>10.4f} {hp_std:>10.4f}')
            if len(self.all_results) > 10:
                print(f'... and {len(self.all_results) - 10} more clips')
            print('-' * 70)

    def save_results(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.aggregate_results, f, indent=2)
        print(f'\nResults saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Run evaluation metrics on generated talking-head video(s)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_metrics.py --batch \
      --results_dir /workspace/ditto/ditto-talkinghead-train/testset_results/mf_100epoch \
      --test_dir /workspace/ditto/datasets/MEAD/SUBSET_MEAD_FRONT_TEST_W024_M022_M025/testset \
      --real_videos_dir /workspace/ditto/datasets/MEAD/SUBSET_MEAD_FRONT_TEST_W024_M022_M025/real_videos \
      --metric_preset mead_paper
        """,
    )

    parser.add_argument('--batch', action='store_true', default=True, help='Run batch evaluation on a test set')
    parser.add_argument('--single', dest='batch', action='store_false', help='Run single-video evaluation')
    parser.add_argument('--generated', '-g', type=str, default=None, help='Generated video path (single mode)')
    parser.add_argument('--source', '-s', type=str, default=None, help='Source image path (single mode)')
    parser.add_argument('--real', '-r', type=str, default=None, help='Ground-truth video path (single mode)')
    parser.add_argument('--audio', '-a', type=str, default=None, help='Audio path (single mode)')
    # parser.add_argument('--results_dir', type=str, default='/workspace/ditto/ditto-talkinghead-train/dynpose_talk8/ditto_original_posebranch_freeze_Lv75Ls05_spyrt', help='Directory containing generated videos in batch mode')  # /workspace/ditto/ditto-talkinghead-train/testset_results/mf_100epoch
    parser.add_argument('--results_dir', type=str, default='/workspace/ditto/ditto-talkinghead-train/dynpose_talk8/posebranch_freeze_Lv75Ls05_spyrt', help='Directory containing generated videos in batch mode')  # /workspace/ditto/ditto-talkinghead-train/testset_results/mf_100epoch
    parser.add_argument('--test_dir', type=str, default='/workspace/ditto/datasets/Talk8/SUBSET_Talk8/testset', help='Directory containing Ditto-format test clips')    # /workspace/ditto/datasets/Talk8/SUBSET_Talk8/testset
    parser.add_argument('--real_videos_dir', type=str, default='/workspace/ditto/datasets/Talk8/SUBSET_Talk8/512x512', help='Directory containing ground-truth videos')  # /workspace/ditto/datasets/Talk8/SUBSET_Talk8/512x512
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to save result JSON')
    parser.add_argument('--max_seconds', type=float, default=10.0, help='Maximum seconds for per-frame metrics if max_frames is not set')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum frames for per-video/frame-based metrics')
    parser.add_argument('--max_windows', type=int, default=None, help='Maximum windows for sync metrics')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--fps', type=int, default=25, help='Target FPS for frame-limit calculation')
    parser.add_argument('--metric_preset',  default='none', help='Apply paper-style truncation presets for dataset-level FVD/E-FID')    # choices=['none', 'mead_paper', 'hdtf_paper']
    parser.add_argument('--fvd_num_frames', type=int, default=128, help='Number of frames used for FVD (default: 128 center frames)')
    parser.add_argument('--efid_num_frames', type=int, default=None, help='Number of frames used for E-FID; if unset, preset may fill it')
    parser.add_argument('--fvd_feature_layer', choices=['logits', 'avg_pool'], default='logits', help='I3D feature layer used for FVD; logits matches the original paper best')
    parser.add_argument('--fvd_target_fps', type=int, default=25, help='FPS used when sampling videos for FVD')
    parser.add_argument('--fvd_frame_sampling', choices=['head', 'center'], default='center', help='How to choose the FVD frame window when fvd_num_frames is smaller than the video length')
    parser.add_argument('--efid_target_fps', type=int, default=25, help='FPS used when sampling videos for E-FID')
    parser.add_argument('--headpose_target_fps', type=int, default=25, help='FPS used when sampling videos for head pose diversity')
    parser.add_argument('--headpose_gpu_id', type=int, default=None, help='GPU id for 6DRepNet head pose estimator (-1 for CPU, default: infer from --device)')
    parser.add_argument('--headpose_model_path', type=str, default='', help='Optional local 6DRepNet weight path; empty uses package auto-download')
    parser.add_argument('--headpose_min_detection_confidence', type=float, default=0.5, help='MediaPipe face detection confidence for 6DRepNet face crops')
    parser.add_argument('--headpose_bbox_scale', type=float, default=1.25, help='Face bbox expansion factor before 6DRepNet pose estimation')
    parser.add_argument('--skip_fvd', action='store_true', help='Skip FVD metric entirely, including dataset-level FVD in batch mode')
    parser.add_argument('--skip_headpose', action='store_true', help='Skip head pose diversity metric')
    parser.add_argument('--skip_efid', action='store_true', help='Skip E-FID metric')
    parser.add_argument('--skip_dataset_fid', action='store_true', help='Skip extra dataset-level FID in batch mode')

    args = parser.parse_args()

    if args.max_seconds is not None:
        if args.max_frames is None:
            args.max_frames = int(args.max_seconds * args.fps)
        if args.max_windows is None:
            args.max_windows = int(args.max_frames / 5)
        print(f'[INFO] Using first {args.max_seconds}s for per-frame metrics: {args.max_frames} frames, {args.max_windows} windows @ {args.fps}fps')

    args.fvd_num_frames, args.efid_num_frames = resolve_metric_preset(args.metric_preset, args.fvd_num_frames, args.efid_num_frames)
    if args.metric_preset != 'none':
        print(f'[INFO] Applied {args.metric_preset}: fvd_num_frames={args.fvd_num_frames}, efid_num_frames={args.efid_num_frames}')

    fvd_kwargs = {'target_fps': args.fvd_target_fps, 'feature_layer': args.fvd_feature_layer, 'frame_sampling': args.fvd_frame_sampling}
    efid_kwargs = {'target_fps': args.efid_target_fps}
    headpose_kwargs = {
        'target_fps': args.headpose_target_fps,
        'gpu_id': resolve_headpose_gpu_id(args.device, args.headpose_gpu_id),
        'model_path': args.headpose_model_path,
        'min_detection_confidence': args.headpose_min_detection_confidence,
        'bbox_scale': args.headpose_bbox_scale,
    }

    if args.batch:
        if not args.results_dir:
            print('Error: --results_dir is required for batch mode')
            sys.exit(1)
        if not os.path.exists(args.results_dir):
            print(f'Error: Results directory not found: {args.results_dir}')
            sys.exit(1)
        if not os.path.exists(args.test_dir):
            print(f'Error: Test directory not found: {args.test_dir}')
            sys.exit(1)

        print('=' * 70)
        print('              BATCH METRICS EVALUATION')
        print('=' * 70)
        print(f'Results Dir: {args.results_dir}')
        print(f'Test Dir: {args.test_dir}')
        if args.real_videos_dir:
            print(f'Real Videos Dir: {args.real_videos_dir}')
        print(f'Per-frame metric budget: {args.max_frames} frames')
        if args.fvd_num_frames is not None:
            print(f'FVD frames: {args.fvd_num_frames} ({args.fvd_frame_sampling})')
        if args.efid_num_frames is not None:
            print(f'E-FID frames: {args.efid_num_frames}')
        print('=' * 70)

        evaluator = BatchMetricsEvaluator(device=args.device, fvd_kwargs=fvd_kwargs, efid_kwargs=efid_kwargs, headpose_kwargs=headpose_kwargs, enable_fvd=not args.skip_fvd, enable_headpose=not args.skip_headpose, enable_efid=not args.skip_efid)
        evaluator.evaluate_batch(results_dir=args.results_dir, test_dir=args.test_dir, real_videos_dir=args.real_videos_dir, max_frames=args.max_frames, max_windows=args.max_windows, fvd_num_frames=args.fvd_num_frames, efid_num_frames=args.efid_num_frames, skip_dataset_fid=args.skip_dataset_fid)
        evaluator.print_summary()
        output_path = args.output or str(Path(args.results_dir) / 'batch_metrics.json')
        evaluator.save_results(output_path)
    else:
        if not args.generated:
            print('Error: --generated is required for single mode (or use --batch for batch mode)')
            sys.exit(1)
        if not os.path.exists(args.generated):
            print(f'Error: Generated video not found: {args.generated}')
            sys.exit(1)

        evaluator = MetricsEvaluator(device=args.device, fvd_kwargs=fvd_kwargs, efid_kwargs=efid_kwargs, headpose_kwargs=headpose_kwargs, enable_fvd=not args.skip_fvd, enable_headpose=not args.skip_headpose, enable_efid=not args.skip_efid)
        evaluator.evaluate_all(generated_video=args.generated, source=args.source, real_video=args.real, audio=args.audio, max_frames=args.max_frames, max_windows=args.max_windows, fvd_num_frames=args.fvd_num_frames, efid_num_frames=args.efid_num_frames)
        evaluator.print_summary()
        output_path = args.output or str(Path(args.generated).with_suffix('')) + '_metrics.json'
        evaluator.save_results(output_path)


if __name__ == '__main__':
    main()
