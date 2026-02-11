import threading
import queue
import numpy as np
import traceback
import time
from tqdm import tqdm

import joblib
import torch

import os
import os.path as osp
import yaml

import sys, os
LP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "prepare_data_train", "LivePortrait"))
if LP_ROOT not in sys.path:
    sys.path.insert(0, LP_ROOT)

from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from src.utils.helper import load_model, concat_feat

from src.config.crop_config import CropConfig
from src.utils.cropper import Cropper



# writer, dit 진행률 표시를 위한 커스텀 tqdm
class PreciseTqdm(tqdm):
    """Custom tqdm that shows elapsed time with decimal precision (seconds). tqdm을 상속받아 format_dict를 오버라이드"""
    def __init__(self, *args, **kwargs):
        # bar_format을 커스터마이징해서 elapsed_sec 사용
        default_bar_format = '{desc}: {n_fmt}it [{elapsed_sec}, {rate_fmt}]'
        if 'bar_format' not in kwargs:
            kwargs['bar_format'] = default_bar_format
        # super().__init__()가 format_dict를 호출할 수 있으므로, 먼저 start_time 설정
        self.start_time = time.perf_counter()
        super().__init__(*args, **kwargs)
    
    @property
    def format_dict(self):
        """Override format_dict to show elapsed time in seconds with decimal precision. 경과 시간을 초 단위로 소수점 3자리까지 표시"""
        d = super().format_dict
        elapsed_seconds = time.perf_counter() - self.start_time
        # elapsed는 숫자로 유지하고, 새로운 변수 elapsed_sec를 문자열로 추가
        d['elapsed_sec'] = f"{elapsed_seconds:.3f}s"
        return d

from core.atomic_components.avatar_registrar import AvatarRegistrar, smooth_x_s_info_lst
from core.atomic_components.condition_handler import ConditionHandler, _mirror_index
from core.atomic_components.audio2motion import Audio2Motion
from core.atomic_components.motion_stitch import MotionStitch
from core.atomic_components.warp_f3d import WarpF3D
from core.atomic_components.decode_f3d import DecodeF3D
from core.atomic_components.putback import PutBack
from core.atomic_components.writer import VideoWriterByImageIO
from core.atomic_components.wav2feat import Wav2Feat
from core.atomic_components.cfg import parse_cfg, print_cfg

# =========================
# LivePortrait-style Retargeting Adapter (official skeleton)
# =========================
# from retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
# from helper import load_model, concat_feat

class LPRetargetingAdapter:
    """
    - weights 로딩: helper.load_model(..., model_type='stitching_retargeting_module')와 동일한 방식
    - ratio 결합: LivePortraitWrapper.calc_combined_eye_ratio / calc_combined_lip_ratio와 동일한 방식
    - Δ 계산: LivePortraitWrapper.retarget_eye / retarget_lip과 동일한 방식
    """
    def __init__(self, checkpoint_S: str, models_yaml: str, device: str):
        assert osp.exists(checkpoint_S), f"checkpoint_S not found: {checkpoint_S}"
        assert osp.exists(models_yaml), f"models_yaml not found: {models_yaml}"

        self.device = torch.device(device)
        model_config = yaml.load(open(models_yaml, "r"), Loader=yaml.SafeLoader)

        # helper.load_model의 stitching_retargeting_module 분기 그대로: {'stitching','lip','eye'} dict 반환
        self.stitching_retargeting_module = load_model(
            checkpoint_S, model_config, self.device, model_type="stitching_retargeting_module"
        )

    def calc_combined_eye_ratio(self, target_eye_ratio: float, source_lmk: np.ndarray) -> torch.Tensor:
        # LivePortraitWrapper.calc_combined_eye_ratio 그대로
        c_s_eyes = calc_eye_close_ratio(source_lmk[None])  # 1x2
        c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(self.device)
        c_d_eyes_tensor = torch.Tensor([[float(target_eye_ratio)]]).to(self.device)  # 1x1
        combined = torch.cat([c_s_eyes_tensor, c_d_eyes_tensor], dim=1)  # 1x3
        return combined
    def calc_combined_eye_ratio_from_eye_open(self, target_eye_ratio: float, eye_open_1x2: np.ndarray) -> torch.Tensor:
        """
        eye_open_1x2: shape (1,2)  (left, right)
        return: (1,3) = [left, right, target]
        """
        c_s = np.asarray(eye_open_1x2, dtype=np.float32).reshape(1, 2)
        c_s_tensor = torch.from_numpy(c_s).float().to(self.device)
        c_d_tensor = torch.tensor([[float(target_eye_ratio)]], device=self.device, dtype=c_s_tensor.dtype)
        return torch.cat([c_s_tensor, c_d_tensor], dim=1)  # 1x3


    def calc_combined_lip_ratio(self, target_lip_ratio: float, source_lmk: np.ndarray) -> torch.Tensor:
        # LivePortraitWrapper.calc_combined_lip_ratio 그대로
        c_s_lip = calc_lip_close_ratio(source_lmk[None])  # 1x1
        c_s_lip_tensor = torch.from_numpy(c_s_lip).float().to(self.device)
        c_d_lip_tensor = torch.Tensor([[float(target_lip_ratio)]]).to(self.device)  # 1x1
        combined = torch.cat([c_s_lip_tensor, c_d_lip_tensor], dim=1)  # 1x2
        return combined

    def retarget_eye(self, kp_source: torch.Tensor, combined_eye_ratio: torch.Tensor) -> torch.Tensor:
        # LivePortraitWrapper.retarget_eye 그대로
        feat_eye = concat_feat(kp_source, combined_eye_ratio)
        with torch.no_grad():
            delta = self.stitching_retargeting_module["eye"](feat_eye)
        return delta.reshape(-1, kp_source.shape[1], 3)  # (B,K,3)

    def retarget_lip(self, kp_source: torch.Tensor, combined_lip_ratio: torch.Tensor) -> torch.Tensor:
        # LivePortraitWrapper.retarget_lip 그대로
        feat_lip = concat_feat(kp_source, combined_lip_ratio)
        with torch.no_grad():
            delta = self.stitching_retargeting_module["lip"](feat_lip)
        return delta.reshape(-1, kp_source.shape[1], 3)  # (B,K,3)


class StreamSDK:
    def __init__(self, cfg_pkl, data_root, use_meanflow, meanflow_mode="improved", **kwargs):
        # use_meanflow, meanflow_mode를 kwargs에 추가하여 parse_cfg에 전달
        kwargs["use_meanflow"] = use_meanflow
        kwargs["meanflow_mode"] = meanflow_mode

        [
            avatar_registrar_cfg,
            condition_handler_cfg,
            lmdm_cfg,
            stitch_network_cfg,
            warp_network_cfg,
            decoder_cfg,
            wav2feat_cfg,
            default_kwargs,
        ] = parse_cfg(cfg_pkl, data_root, kwargs)
        
        self.default_kwargs = default_kwargs
        
        self.avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
        self.condition_handler = ConditionHandler(**condition_handler_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_network_cfg)
        self.warp_f3d = WarpF3D(warp_network_cfg)
        self.decode_f3d = DecodeF3D(decoder_cfg)
        self.putback = PutBack()

        self.wav2feat = Wav2Feat(**wav2feat_cfg)
        
        # 시간 측정용 변수 초기화
        self.timing_stats = {
            'audio2feat_total_ms': 0.0,  # total time for wav2feat (HuBERT)
            'audio2feat_per_step_ms': [],
            'dit_total_ms': 0.0,    # total time for audio2motion
            'dit_per_chunk_ms': [],
            'writer_total_ms': 0.0,  # total time for writer
            'writer_per_frame_ms': [],  # per frame time for writer
            'warp_total_ms': 0.0,  # total time for warp
            'decode_total_ms': 0.0,  # total time for decode
            'putback_total_ms': 0.0,  # total time for putback
            'stitch_total_ms': 0.0,  # total time for stitch
        }
        

    def _merge_kwargs(self, default_kwargs, run_kwargs):
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs

    def setup_Nd(self, N_d, fade_in=-1, fade_out=-1, ctrl_info=None):
        # for eye open at video end
        self.motion_stitch.set_Nd(N_d)

        # for fade in/out alpha
        if ctrl_info is None:
            ctrl_info = self.ctrl_info
        if fade_in > 0:
            for i in range(fade_in):
                alpha = i / fade_in
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
        if fade_out > 0:
            ss = N_d - fade_out - 1
            ee = N_d - 1
            for i in range(ss, N_d):
                alpha = max((ee - i) / (ee - ss), 0)
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
        self.ctrl_info = ctrl_info

    def setup(self, source_path, output_path, **kwargs):

        # ======== Prepare Options ========
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        print("=" * 20, "setup kwargs", "=" * 20)
        print_cfg(**kwargs)
        print("=" * 50)

        # -- avatar_registrar: template cfg --
        self.max_size = 512 #kwargs.get("max_size", 1920)    # 기본값 1920는 full-body/ 512로 바꿔서 rtf metric 측정: 512 # (talking-head)에 맞는 크기
        self.template_n_frames = kwargs.get("template_n_frames", -1)

        # -- avatar_registrar: crop cfg --
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        
        # -- avatar_registrar: smo for video --
        self.smo_k_s = kwargs.get('smo_k_s', 13)

        # -- condition_handler: ECS --
        self.emo = kwargs.get("emo", 4)    # int | [int] | [[int]] | numpy
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)    # for video
        self.ch_info = kwargs.get("ch_info", None)    # dict of np.ndarray

        # -- audio2motion: setup --
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)  # [ds,de]
        self.sampling_timesteps = 10 # kwargs.get("sampling_timesteps", 50) # 10 # 
        self.online_mode = False #kwargs.get("online_mode", False)
        self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
        self.smo_k_d = kwargs.get("smo_k_d", 3)

        # -- motion_stitch: setup --
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        self.drive_eye = kwargs.get("drive_eye", None)    # None: true4image, false4video
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")    # "" | "d0" | "s"
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)

        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        """
        ctrl_info: list or dict
            {
                fid: ctrl_kwargs
            }

            ctrl_kwargs (see motion_stitch.py):
                fade_alpha
                fade_out_keys

                delta_pitch
                delta_yaw
                delta_roll
        """

        # only hubert support online mode
        assert self.wav2feat.support_streaming or not self.online_mode

        # ======== Register Avatar ========
        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        n_frames = self.template_n_frames if self.template_n_frames > 0 else self.N_d
        source_info = self.avatar_registrar(
            source_path, 
            max_dim=self.max_size, 
            n_frames=n_frames, 
            **crop_kwargs,
        )

        if len(source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            source_info["x_s_info_lst"] = smooth_x_s_info_lst(source_info["x_s_info_lst"], smo_k=self.smo_k_s)

        self.source_info = source_info
        # print("[DEBUG] source_info keys:", self.source_info.keys())
        # print("[DEBUG] eye_open_lst type:", type(self.source_info.get("eye_open_lst", None)))
        # if "eye_open_lst" in self.source_info:
        #     e0 = self.source_info["eye_open_lst"][0]
        #     try:
        #         import numpy as np
        #         print("[DEBUG] eye_open_lst[0] type:", type(e0))
        #         if hasattr(e0, "shape"):
        #             print("[DEBUG] eye_open_lst[0].shape:", e0.shape)
        #         else:
        #             print("[DEBUG] eye_open_lst[0] (no shape):", e0)
        #         # numpy로 변환 가능한지도 확인
        #         e0_np = np.array(e0)
        #         print("[DEBUG] np.array(e0).shape:", e0_np.shape, "value:", e0_np)
        #     except Exception as ex:
        #         print("[DEBUG] eye_open debug failed:", ex)
        # else:
        #     print("[DEBUG] eye_open_lst not found in source_info")
        self.source_info_frames = len(source_info["x_s_info_lst"])

        # =========================
        # LivePortrait retargeting options (normalize ref baseline)
        # =========================
        self.lp_retarget_enable = kwargs.get("lp_retarget_enable", False)

        # generation-order counter (motion_stitch_worker에서 1씩 증가)
        self.lp_retarget_fid = 0
        self.lp_retarget_delta = None  # (B,K,3) cached once

        if self.lp_retarget_enable:
            # 필수: LivePortrait weights + models.yaml
            self.lp_checkpoint_S = kwargs.get("lp_checkpoint_S", None)
            self.lp_models_yaml  = kwargs.get("lp_models_yaml", None)

            # 목표 ratio: (눈 뜨기 / 입 닫기)
            # - LivePortrait 코드에서 눈은 0.39 같은 값을 "최소 오픈"으로 쓰는 패턴이 있음. :contentReference[oaicite:9]{index=9}
            self.lp_target_eye_ratio = float(kwargs.get("lp_target_eye_ratio", 0.39))
            self.lp_target_lip_ratio = float(kwargs.get("lp_target_lip_ratio", 0.0))

            # 적용 프레임 스케줄: 초반 N프레임만 강하게, 이후 fade_n 동안 선형 감소
            self.lp_first_n = int(kwargs.get("lp_first_n", 25))   # 25fps 기준 1초
            self.lp_fade_n  = int(kwargs.get("lp_fade_n", 10))

            # 어디에 더할지: 'driving' 추천 (baseline만 올리고 깜빡임 변동은 유지)
            self.lp_apply_to = kwargs.get("lp_apply_to", "driving")  # 'driving' or 'source'

            # retargeting MLP는 매우 가벼워서 CPU로 돌려도 됨.
            # torch 텐서로 x_s/x_d가 GPU면 device를 cuda로 두는 게 복사 비용이 적음.
            if torch.cuda.is_available():
                lp_device = kwargs.get("lp_device", "cuda:0")
            else:
                lp_device = kwargs.get("lp_device", "cpu")

            if self.lp_checkpoint_S is None or self.lp_models_yaml is None:
                print("[LP-Retarget] lp_checkpoint_S / lp_models_yaml must be provided. Disable.")
                self.lp_retarget_enable = False
            else:
                self.lp_retargeter = LPRetargetingAdapter(
                    checkpoint_S=self.lp_checkpoint_S,
                    models_yaml=self.lp_models_yaml,
                    device=lp_device,
                )
                print(f"[LP-Retarget] enabled. apply_to={self.lp_apply_to}, "
                      f"eye_target={self.lp_target_eye_ratio}, lip_target={self.lp_target_lip_ratio}, "
                      f"first_n={self.lp_first_n}, fade_n={self.lp_fade_n}, device={lp_device}")

            # setup() 안, [LP-Retarget] enabled 로그 찍은 직후 추천
            if self.lp_retarget_enable:
                if "lmk_crop" not in source_info or source_info["lmk_crop"] is None:
                    t0 = time.perf_counter()
                    try:
                        crop_cfg = CropConfig()

                        # (선택) ditto crop 파라미터와 최대한 맞추고 싶으면 아래처럼 덮어쓰기
                        crop_cfg.dsize = 512
                        crop_cfg.scale = getattr(self, "crop_scale", crop_cfg.scale)
                        crop_cfg.vx_ratio = getattr(self, "crop_vx_ratio", crop_cfg.vx_ratio)
                        crop_cfg.vy_ratio = getattr(self, "crop_vy_ratio", crop_cfg.vy_ratio)
                        crop_cfg.flag_do_rot = getattr(self, "crop_flag_do_rot", crop_cfg.flag_do_rot)

                        # weight 경로 sanity check (없으면 여기서 바로 원인 잡힘)
                        if not osp.exists(crop_cfg.insightface_root) or not osp.exists(crop_cfg.landmark_ckpt_path):
                            print(f"[LP-LMK] Missing weights. insightface_root={crop_cfg.insightface_root}, "
                                f"landmark_ckpt_path={crop_cfg.landmark_ckpt_path}")
                        else:
                            # Cropper는 (얼굴검출+landmark runner) 준비
                            device_id = 0
                            self.lp_cropper = Cropper(crop_cfg=crop_cfg, device_id=device_id)

                            img0 = source_info["img_rgb_lst"][0]  # reference RGB
                            crop_ret = self.lp_cropper.crop_source_image(img0, crop_cfg)  # lmk_crop 생성 :contentReference[oaicite:12]{index=12}

                            if crop_ret is None:
                                print("[LP-LMK] crop_source_image returned None (face not detected?)")
                            else:
                                source_info["lmk_crop"] = crop_ret["lmk_crop"]
                                print(f"[LP-LMK] injected lmk_crop shape={source_info['lmk_crop'].shape}, "
                                    f"setup_time={(time.perf_counter()-t0)*1000:.1f}ms")
                    except Exception as e:
                        print("[LP-LMK] landmark setup failed:", e)
                        traceback.print_exc()

        # =========================
        

        # ======== Setup Condition Handler ========
        self.condition_handler.setup(source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info)

        # ======== Setup Audio2Motion (LMDM) ========
        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            x_s_info_0, 
            overlap_v2=self.overlap_v2,
            fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
        )

        # ======== Setup Motion Stitch ========
        is_image_flag = source_info["is_image_flag"]
        x_s_info = source_info['x_s_info_lst'][0]
        self.motion_stitch.setup(
            N_d=self.N_d,
            use_d_keys=self.use_d_keys,
            relative_d=self.relative_d,
            drive_eye=self.drive_eye,
            delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag,
            x_s_info=x_s_info,
            d0=None,
            ch_info=self.ch_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )

        # ======== Video Writer ========
        self.output_path = output_path
        self.tmp_output_path = output_path + ".tmp.mp4"
        self.writer = VideoWriterByImageIO(self.tmp_output_path)
        # Custom tqdm with precise elapsed time in seconds (decimal precision)
        # self.writer_pbar = tqdm(desc="writer")
        self.writer_pbar = PreciseTqdm(desc="writer")

        # ======== Audio Feat Buffer ========
        if self.online_mode:
            # buffer: seq_frames - valid_clip_len
            self.audio_feat = self.wav2feat.wav2feat(np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000)
            assert len(self.audio_feat) == self.overlap_v2, f"{len(self.audio_feat)}"
        else:
            self.audio_feat = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
        self.cond_idx_start = 0 - len(self.audio_feat)

        # ======== Setup Worker Threads ========
        QUEUE_MAX_SIZE = 100
        # self.QUEUE_TIMEOUT = None

        self.worker_exception = None
        self.stop_event = threading.Event()

        self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.putback_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.writer_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)

        self.thread_list = [
            threading.Thread(target=self.audio2motion_worker),
            threading.Thread(target=self.motion_stitch_worker),
            threading.Thread(target=self.warp_f3d_worker),
            threading.Thread(target=self.decode_f3d_worker),
            threading.Thread(target=self.putback_worker),
            threading.Thread(target=self.writer_worker),
        ]

        for thread in self.thread_list:
            thread.start()
        
        # 매 setup마다 타이밍 초기화
        self.timing_stats = {
            'audio2feat_total_ms': 0.0,  # total time for wav2feat (HuBERT)
            'audio2feat_per_step_ms': [],
            'dit_total_ms': 0.0,     # total time for audio2motion
            'dit_per_chunk_ms': [],
            'writer_total_ms': 0.0,  # total time for writer
            'writer_per_frame_ms': [],  # per frame time for writer
            'warp_total_ms': 0.0,  # total time for warp
            'decode_total_ms': 0.0,  # total time for decode
            'putback_total_ms': 0.0,  # total time for putback
            'stitch_total_ms': 0.0,  # total time for stitch
        }

    def _lp_get_source_lmk(self, frame_idx: int):
        """
        AvatarRegistrar가 반환한 source_info 안에서 landmarks를 찾아온다.
        (프로젝트마다 키 이름이 다를 수 있으니 가능한 후보를 폭넓게 체크)
        """
        si = self.source_info

        for key in ["lmk_crop_lst", "lmk_lst", "lmks", "landmarks_lst"]:
            if key in si:
                v = si[key]
                if isinstance(v, list):
                    return v[frame_idx]
                else:
                    return v[frame_idx]

        for key in ["lmk_crop", "lmk", "landmarks"]:
            if key in si:
                return si[key]

        return None

    def _lp_weight(self, fid: int) -> float:
        """
        초반 first_n 프레임에 1.0, 이후 fade_n 동안 선형 감소, 그 뒤 0.0
        """
        if fid < self.lp_first_n:
            return 1.0
        if self.lp_fade_n > 0 and fid < self.lp_first_n + self.lp_fade_n:
            return 1.0 - (fid - self.lp_first_n) / float(self.lp_fade_n)
        return 0.0

    def _get_ctrl_info(self, fid):
        try:
            if isinstance(self.ctrl_info, dict):
                return self.ctrl_info.get(fid, {})
            elif isinstance(self.ctrl_info, list):
                return self.ctrl_info[fid]
            else:
                return {}
        except Exception as e:
            traceback.print_exc()
            return {}

    def writer_worker(self):
        try:
            self._writer_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _writer_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break
            res_frame_rgb = item
            
            t_start = time.perf_counter()
            self.writer(res_frame_rgb, fmt="rgb")
            t_elapsed = (time.perf_counter() - t_start) * 1000
            self.timing_stats['writer_per_frame_ms'].append(t_elapsed)
            self.timing_stats['writer_total_ms'] += t_elapsed           # ← writer_total_ms: 모든 프레임 쓰기 시간
            
            self.writer_pbar.update()

    def putback_worker(self):
        try:
            self._putback_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _putback_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.putback_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.writer_queue.put(None)
                break
            frame_idx, render_img = item
            frame_rgb = self.source_info["img_rgb_lst"][frame_idx]
            M_c2o = self.source_info["M_c2o_lst"][frame_idx]
            
            t_start = time.perf_counter()
            res_frame_rgb = self.putback(frame_rgb, render_img, M_c2o)
            self.timing_stats['putback_total_ms'] += (time.perf_counter() - t_start) * 1000
            
            self.writer_queue.put(res_frame_rgb)

    def decode_f3d_worker(self):
        try:
            self._decode_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _decode_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.decode_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.putback_queue.put(None)
                break
            frame_idx, f_3d = item
            
            t_start = time.perf_counter()
            render_img = self.decode_f3d(f_3d)
            self.timing_stats['decode_total_ms'] += (time.perf_counter() - t_start) * 1000
            
            self.putback_queue.put([frame_idx, render_img])

    def warp_f3d_worker(self):
        try:
            self._warp_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _warp_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.warp_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.decode_f3d_queue.put(None)
                break
            frame_idx, x_s, x_d = item
            f_s = self.source_info["f_s_lst"][frame_idx]
            
            t_start = time.perf_counter()
            f_3d = self.warp_f3d(f_s, x_s, x_d)
            self.timing_stats['warp_total_ms'] += (time.perf_counter() - t_start) * 1000
            
            self.decode_f3d_queue.put([frame_idx, f_3d])

    def motion_stitch_worker(self):
        try:
            self._motion_stitch_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _motion_stitch_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.motion_stitch_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.warp_f3d_queue.put(None)
                break
            
            frame_idx, x_d_info, ctrl_kwargs = item
            x_s_info = self.source_info["x_s_info_lst"][frame_idx]
            
            t_start = time.perf_counter()
            x_s, x_d = self.motion_stitch(x_s_info, x_d_info, **ctrl_kwargs)    # 최종 keypoints
            self.timing_stats['stitch_total_ms'] += (time.perf_counter() - t_start) * 1000

            # =========================
            # LivePortrait retargeting apply (in implicit-keypoints space)
            # =========================
            if getattr(self, "lp_retarget_enable", False):
                fid = self.lp_retarget_fid
                self.lp_retarget_fid += 1

                w = self._lp_weight(fid)
                if w > 0:
                    # delta가 아직 없으면(첫 적용 순간) 1회만 계산해서 캐싱
                    if self.lp_retarget_delta is None:
                        # source_lmk = self._lp_get_source_lmk(frame_idx)
                        # if source_lmk is None:
                        #     print("[LP-Retarget] source landmarks not found in source_info. Disable retargeting.")
                        #     self.lp_retarget_enable = False
                        # else:
                        #     # x_s/x_d 타입이 torch일 때를 기본으로 (대부분 warp_f3d가 torch 기반)
                        #     if not isinstance(x_s, torch.Tensor):
                        #         # numpy라면 torch로 올려서 delta 계산 후 다시 numpy로 변환
                        #         x_s_t = torch.from_numpy(x_s).float().to(self.lp_retargeter.device)
                        #     else:
                        #         x_s_t = x_s.to(self.lp_retargeter.device)

                        #     # 공식 wrapper 흐름: combined ratio → retarget_eye/lip
                        #     ce = self.lp_retargeter.calc_combined_eye_ratio(self.lp_target_eye_ratio, source_lmk)
                        #     cl = self.lp_retargeter.calc_combined_lip_ratio(self.lp_target_lip_ratio, source_lmk)
                        #     de = self.lp_retargeter.retarget_eye(x_s_t, ce)  # (B,K,3)
                        #     dl = self.lp_retargeter.retarget_lip(x_s_t, cl)  # (B,K,3)
                        #     self.lp_retarget_delta = (de + dl)               # (B,K,3)
                        source_lmk = self._lp_get_source_lmk(frame_idx)

                        # x_s 텐서 준비
                        if not isinstance(x_s, torch.Tensor):
                            x_s_t = torch.from_numpy(x_s).float().to(self.lp_retargeter.device)
                        else:
                            x_s_t = x_s.to(self.lp_retargeter.device)

                        # --- Eye delta 계산 (landmark 있으면 landmark, 없으면 eye_open_lst fallback) ---
                        if source_lmk is not None:
                            ce = self.lp_retargeter.calc_combined_eye_ratio(self.lp_target_eye_ratio, source_lmk)
                        else:
                            if "eye_open_lst" not in self.source_info:
                                print("[LP-Retarget] no landmarks AND no eye_open_lst. Disable retargeting.")
                                self.lp_retarget_enable = False
                                ce = None
                            else:
                                eye_open = self.source_info["eye_open_lst"][frame_idx]  # (1,2)
                                ce = self.lp_retargeter.calc_combined_eye_ratio_from_eye_open(self.lp_target_eye_ratio, eye_open)
                                print(f"[LP-Retarget] landmark missing -> eye_open fallback used. eye_open={eye_open}, target={self.lp_target_eye_ratio}")

                        if ce is not None:
                            de = self.lp_retargeter.retarget_eye(x_s_t, ce)  # (B,K,3)

                            # --- Lip은 landmark 없으면 안전하게 skip(0) ---
                            if source_lmk is not None:
                                cl = self.lp_retargeter.calc_combined_lip_ratio(self.lp_target_lip_ratio, source_lmk)
                                dl = self.lp_retargeter.retarget_lip(x_s_t, cl)  # (B,K,3)
                            else:
                                dl = torch.zeros_like(de)

                            self.lp_retarget_delta = (de + dl)

                    if self.lp_retarget_delta is not None:
                        # delta를 x_s/x_d 타입에 맞춰 적용
                        if isinstance(x_d, torch.Tensor):
                            delta = self.lp_retarget_delta.to(device=x_d.device, dtype=x_d.dtype)
                            if self.lp_apply_to == "source":
                                x_s = x_s + w * delta
                            else:
                                x_d = x_d + w * delta
                        else:
                            # numpy일 경우
                            delta_np = self.lp_retarget_delta.detach().cpu().numpy().astype(np.float32)
                            if self.lp_apply_to == "source":
                                x_s = x_s + w * delta_np
                            else:
                                x_d = x_d + w * delta_np
            # =========================
            
            self.warp_f3d_queue.put([frame_idx, x_s, x_d])    # ← warping 단계로 넘김 / ⭐️ 여기서 x_s/x_d에 → retargeting Δ를 더해주기

    def audio2motion_worker(self):
        try:
            # self._audio2motion_worker()
            self._audio2motion_offline()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _audio2motion_offline(self):

        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)    # audio feat
            except queue.Empty:
                continue

            if item is None:
                break

            aud_feat = item

            aud_cond_all = self.condition_handler(aud_feat, 0)
            seq_frames = self.audio2motion.seq_frames
            valid_clip_len = self.audio2motion.valid_clip_len
            num_frames = len(aud_cond_all)
            idx = 0
            res_kp_seq = None
            # Custom tqdm with precise elapsed time in seconds (decimal precision)
            # pbar = tqdm(desc="dit")
            pbar = PreciseTqdm(desc="dit")
            dit_start_time = time.perf_counter()
            while idx < num_frames:
                pbar.update()
                aud_cond = aud_cond_all[idx:idx + seq_frames][None]
                if aud_cond.shape[1] < seq_frames:
                    pad = np.stack([aud_cond[:, -1]] * (seq_frames - aud_cond.shape[1]), 1)
                    aud_cond = np.concatenate([aud_cond, pad], 1)
                
                chunk_start = time.perf_counter()
                res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)        # ← 여기가 측정됨. LMDM 모델이 호출되는 곳.
                chunk_elapsed = (time.perf_counter() - chunk_start) * 1000
                self.timing_stats['dit_per_chunk_ms'].append(chunk_elapsed)
                
                idx += valid_clip_len
            
            dit_total = (time.perf_counter() - dit_start_time) * 1000       # ← dit_total: 모션 생성 시간(Diffusion loop)
            self.timing_stats['dit_total_ms'] = dit_total
            pbar.close()
            res_kp_seq = res_kp_seq[:, :num_frames]
            res_kp_seq = self.audio2motion._smo(res_kp_seq, 0, res_kp_seq.shape[1])     # 최종 스무딩

            x_d_info_list = self.audio2motion.cvt_fmt(res_kp_seq)                       # 모션 정보(포맷) 변환

            gen_frame_idx = 0
            for x_d_info in x_d_info_list:
                frame_idx = _mirror_index(gen_frame_idx, self.source_info_frames)
                ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)

                while not self.stop_event.is_set():
                    try:
                        self.motion_stitch_queue.put([frame_idx, x_d_info, ctrl_kwargs], timeout=1)
                        break
                    except queue.Full:
                        continue
                gen_frame_idx += 1

            break

        self.motion_stitch_queue.put(None)

        
    def _audio2motion_worker(self):
        is_end = False
        seq_frames = self.audio2motion.seq_frames
        valid_clip_len = self.audio2motion.valid_clip_len
        aud_feat_dim = self.wav2feat.feat_dim
        item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

        res_kp_seq = None
        res_kp_seq_valid_start = None if self.online_mode else 0
        
        global_idx = 0   # frame idx, for template
        local_idx = 0    # for cur audio_feat
        gen_frame_idx = 0
        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)    # audio feat
            except queue.Empty:
                continue
            if item is None:
                is_end = True
            else:
                item_buffer = np.concatenate([item_buffer, item], 0)

            if not is_end and item_buffer.shape[0] < valid_clip_len:
                # wait at least valid_clip_len new item
                continue
            else:
                self.audio_feat = np.concatenate([self.audio_feat, item_buffer], 0)
                item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

            while True:
                # print("self.audio_feat.shape:", self.audio_feat.shape, "local_idx:", local_idx, "global_idx:", global_idx)
                aud_feat = self.audio_feat[local_idx: local_idx+seq_frames]
                real_valid_len = valid_clip_len
                if len(aud_feat) == 0:
                    break
                elif len(aud_feat) < seq_frames:
                    if not is_end:
                        # wait next chunk
                        break
                    else:
                        # final clip: pad to seq_frames
                        real_valid_len = len(aud_feat)
                        pad = np.stack([aud_feat[-1]] * (seq_frames - len(aud_feat)), 0)
                        aud_feat = np.concatenate([aud_feat, pad], 0)

                aud_cond = self.condition_handler(aud_feat, global_idx + self.cond_idx_start)
                res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
                if res_kp_seq_valid_start is None:
                    # online mode, first chunk
                    res_kp_seq_valid_start = res_kp_seq.shape[1] - self.audio2motion.fuse_length
                    d0 = self.audio2motion.cvt_fmt(res_kp_seq[0:1])[0]
                    self.motion_stitch.d0 = d0

                    local_idx += real_valid_len
                    global_idx += real_valid_len
                    continue
                else:
                    valid_res_kp_seq = res_kp_seq[:, res_kp_seq_valid_start: res_kp_seq_valid_start + real_valid_len]
                    x_d_info_list = self.audio2motion.cvt_fmt(valid_res_kp_seq)

                    for x_d_info in x_d_info_list:
                        frame_idx = _mirror_index(gen_frame_idx, self.source_info_frames)
                        ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)

                        while not self.stop_event.is_set():
                            try:
                                self.motion_stitch_queue.put([frame_idx, x_d_info, ctrl_kwargs], timeout=1)
                                break
                            except queue.Full:
                                continue

                        gen_frame_idx += 1

                    res_kp_seq_valid_start += real_valid_len
                
                    local_idx += real_valid_len
                    global_idx += real_valid_len

                L = res_kp_seq.shape[1] 
                if L > seq_frames * 2:
                    cut_L = L - seq_frames * 2
                    res_kp_seq = res_kp_seq[:, cut_L:]
                    res_kp_seq_valid_start -= cut_L

                if local_idx >= len(self.audio_feat):
                    break

            L = len(self.audio_feat)
            if L > seq_frames * 2:
                cut_L = L - seq_frames * 2
                self.audio_feat = self.audio_feat[cut_L:]
                local_idx -= cut_L

            if is_end:
                break
        
        self.motion_stitch_queue.put(None)

    def close(self):
        # flush frames
        self.audio2motion_queue.put(None)
        # Wait for worker threads to finish
        for thread in self.thread_list:
            thread.join()

        try:
            self.writer.close()
            self.writer_pbar.close()
        except:
            traceback.print_exc()

        # Check if any worker encountered an exception
        if self.worker_exception is not None:
            raise self.worker_exception
        
    def run_chunk(self, audio_chunk, chunksize=(3, 5, 2)):
        # only for hubert
        t0 = time.perf_counter()
        aud_feat = self.wav2feat(audio_chunk, chunksize=chunksize)
        t_ms = (time.perf_counter() - t0) * 1000.0

        self.timing_stats['audio2feat_total_ms'] += t_ms
        self.timing_stats['audio2feat_per_step_ms'].append(t_ms)

        while not self.stop_event.is_set():
            try:
                self.audio2motion_queue.put(aud_feat, timeout=1)
                break
            except queue.Full:
                continue

    



