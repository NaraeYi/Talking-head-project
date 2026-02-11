import numpy as np
import torch
from ..utils.load_model import load_model


def make_beta(n_timestep, cosine_s=8e-3):
    timesteps = (
        torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
    )
    alphas = timesteps / (1 + cosine_s) * np.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = np.clip(betas, a_min=0, a_max=0.999)
    return betas.numpy()


class LMDM:
    def __init__(self, model_path, device="cuda", **kwargs):
        kwargs["module_name"] = "LMDM"
        
        self.use_meanflow = kwargs.get("use_meanflow", False)
        self.meanflow_mode = kwargs.get("meanflow_mode", "improved")

        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

        self.motion_feat_dim = kwargs.get("motion_feat_dim", 265)
        self.audio_feat_dim = kwargs.get("audio_feat_dim", 1024+35)
        self.seq_frames = kwargs.get("seq_frames", 80)

        # 디버그: 로드된 모델 타입 출력
        print(f"[LMDM] Loaded model: {model_path}")
        print(f"[LMDM] Model type: {self.model_type}, use_meanflow: {self.use_meanflow}, meanflow_mode: {self.meanflow_mode}")

        if self.model_type == "pytorch":
            pass
        else:
            if self.use_meanflow:
                # MeanFlow TensorRT 지원
                self._init_meanflow_trt()
            else:
                self._init_np()

    def setup(self, sampling_timesteps):
        # MeanFlow doesn't need setup (no diffusion timesteps)
        if self.use_meanflow:
            return
            
        if self.model_type == "pytorch":
            self.model.setup(sampling_timesteps)
        else:
            self._setup_np(sampling_timesteps)

    def _init_meanflow_trt(self):
        """MeanFlow TensorRT 초기화 (현재는 placeholder)"""
        # TensorRT 엔진은 이미 load_model에서 로드됨
        # 추가 초기화가 필요하면 여기에 구현
        pass

    def _init_np(self):
        self.sampling_timesteps = None
        self.n_timestep = 1000

        betas = torch.Tensor(make_beta(n_timestep=self.n_timestep))
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cpu().numpy()

    def _setup_np(self, sampling_timesteps):
        if self.sampling_timesteps == sampling_timesteps:
            return
        
        self.sampling_timesteps = sampling_timesteps

        total_timesteps = self.n_timestep
        eta = 1
        shape = (1, self.seq_frames, self.motion_feat_dim)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        self.time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        self.time_cond_list = []
        self.alpha_next_sqrt_list = []
        self.sigma_list = []
        self.c_list = []
        self.noise_list = []

        for time, time_next in self.time_pairs:
            time_cond = np.full((1,), time, dtype=np.int64)
            self.time_cond_list.append(time_cond)
            if time_next < 0:
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * np.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            c = np.sqrt(1 - alpha_next - sigma ** 2)
            noise = np.random.randn(*shape).astype(np.float32)
            
            self.alpha_next_sqrt_list.append(np.sqrt(alpha_next))
            self.sigma_list.append(sigma)
            self.c_list.append(c)
            self.noise_list.append(noise)

    def _one_step(self, x, cond_frame, cond, time_cond):
        if self.model_type == "onnx":
            pred = self.model.run(None, {"x": x, "cond_frame": cond_frame, "cond": cond, "time_cond": time_cond})
            pred_noise, x_start = pred[0], pred[1]
        elif self.model_type == "tensorrt":
            self.model.setup({"x": x, "cond_frame": cond_frame, "cond": cond, "time_cond": time_cond})
            self.model.infer()
            pred_noise, x_start = self.model.buffer["pred_noise"][0], self.model.buffer["x_start"][0]
        elif self.model_type == "pytorch":
            with torch.no_grad():
                pred_noise, x_start = self.model(x, cond_frame, cond, time_cond)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return pred_noise, x_start

    def _call_np(self, kp_cond, aud_cond, sampling_timesteps):
        self._setup_np(sampling_timesteps)

        cond_frame = kp_cond
        cond = aud_cond

        x = np.random.randn(1, self.seq_frames, self.motion_feat_dim).astype(np.float32)

        x_start = None
        i = 0
        for _, time_next in self.time_pairs:
            time_cond = self.time_cond_list[i]
            pred_noise, x_start = self._one_step(x, cond_frame, cond, time_cond)
            if time_next < 0:
                x = x_start
                continue

            alpha_next_sqrt = self.alpha_next_sqrt_list[i]
            c = self.c_list[i]
            sigma = self.sigma_list[i]
            noise = self.noise_list[i]
            x = x_start * alpha_next_sqrt + c * pred_noise + sigma * noise

            i += 1

        return x

    def _meanflow_sample_np(self, kp_cond, aud_cond, shared_noise=None):
        """
        MeanFlow TensorRT/ONNX 추론
        
        MeanFlow 1-step sampling: x = e - u(e, r=0, t=1)
        
        Args:
            kp_cond: [1, D] condition frame
            aud_cond: [1, T, audio_dim] audio condition
            shared_noise: [1, 1, D] shared noise (optional, for consistency)
        
        Returns:
            pred_kp_seq: [1, T, D] predicted keypoints
            used_noise: [1, 1, D] noise used (for consistency)
        """
        B, T, D = 1, self.seq_frames, self.motion_feat_dim
        
        # Noise 생성 (shared_noise가 있으면 사용, 없으면 새로 생성)
        if shared_noise is not None:
            if isinstance(shared_noise, torch.Tensor):
                e0 = shared_noise.cpu().numpy().astype(np.float32)
            else:
                e0 = shared_noise.astype(np.float32)
            # e0 shape: [1, 1, D] -> [1, T, D]로 확장
            if e0.shape[1] == 1:
                e = np.repeat(e0, T, axis=1)  # [1, T, D]
            else:
                e = e0  # 이미 [1, T, D] 형태
        else:
            # 새 노이즈 생성: [1, 1, D] -> [1, T, D]
            e0 = np.random.randn(B, 1, D).astype(np.float32)
            e = np.repeat(e0, T, axis=1)  # [1, T, D]
        
        # MeanFlow 입력 준비
        cond_frame = kp_cond.astype(np.float32)  # [1, D]
        cond = aud_cond.astype(np.float32)  # [1, T, audio_dim]
        r = np.zeros((B, 1, 1), dtype=np.float32)  # target time (0)
        t = np.ones((B, 1, 1), dtype=np.float32)  # current time (1)
        # cond_drop_prob = np.array(0.0, dtype=np.float32)  # 항상 0.0
        cond_drop_prob = np.zeros((1,), dtype=np.float32)  # shape: (1,)

        # ✅ 1) 후보 입력을 전부 모아두고
        feed_all = {
            "e": e,
            "cond_frame": cond_frame,
            "cond": cond,
            "r": r,
            "t": t,
            "cond_drop_prob": cond_drop_prob,
        }
        # ✅ 2) 모델이 실제로 요구하는 입력만 골라서 전달
        # TensorRT/ONNX 추론
        if self.model_type == "onnx":
            # ONNX Runtime
            needed = {i.name for i in self.model.get_inputs()}
            feed = {k: v for k, v in feed_all.items() if k in needed}
            output = self.model.run(None, feed)
            u = output[0]

            # output = self.model.run(
            #     None,
            #     {
            #         "e": e,
            #         "cond_frame": cond_frame,
            #         "cond": cond,
            #         "r": r,
            #         "t": t,
            #         "cond_drop_prob": cond_drop_prob,
            #     }
            # )
            # u = output[0]  # velocity prediction [1, T, D]
        elif self.model_type == "tensorrt":
            # TensorRT
            # TensorRT wrapper는 구현마다 다름 → 1) input_names가 있으면 사용, 2) 없으면 try/except로 안전 처리
            if hasattr(self.model, "input_names"):
                needed = set(self.model.input_names)
                feed = {k: v for k, v in feed_all.items() if k in needed}
                self.model.setup(feed)
            else:
                # fallback: 일단 다 넣어보고, 실패하면 cond_drop_prob 제거 후 재시도
                try:
                    self.model.setup(feed_all)
                except Exception:
                    feed_all.pop("cond_drop_prob", None)
                    self.model.setup(feed_all)

            self.model.infer()
            u = self.model.buffer["u"][0].copy()
            # self.model.setup({
            #     "e": e,
            #     "cond_frame": cond_frame,
            #     "cond": cond,
            #     "r": r,
            #     "t": t,
            #     "cond_drop_prob": cond_drop_prob,
            # })
            # self.model.infer()
            # u = self.model.buffer["u"][0].copy()  # velocity prediction [1, T, D]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # MeanFlow equation: z_0 = z_1 - u(z_1, r=0, t=1)
        # e: [1, T, D], u: [1, T, D]
        pred_kp_seq = e - u  # [1, T, D]
        
        return pred_kp_seq, e0  # used_noise 반환 (e0: [1, 1, D])

    def __call__(self, kp_cond, aud_cond, sampling_timesteps, shared_noise=None):
        if self.model_type == "pytorch":
            if self.use_meanflow:
                # MeanFlow: 1-step sampling with shared noise for consistency across clips
                pred_kp_seq, used_noise = self.model.meanflow_sample(
                    torch.from_numpy(kp_cond).to(self.device), 
                    torch.from_numpy(aud_cond).to(self.device),
                    shared_noise=shared_noise,
                )
                return pred_kp_seq.cpu().numpy(), used_noise
            else:
                # Diffusion: DDIM sampling
                pred_kp_seq = self.model.ddim_sample(
                    torch.from_numpy(kp_cond).to(self.device), 
                    torch.from_numpy(aud_cond).to(self.device), 
                    sampling_timesteps,
                ).cpu().numpy()
                return pred_kp_seq, None
        else:  # model_type is onnx or tensorrt
            if self.use_meanflow:
                # MeanFlow with TensorRT/ONNX
                pred_kp_seq, used_noise = self._meanflow_sample_np(kp_cond, aud_cond, shared_noise)
                return pred_kp_seq, used_noise
            else:
                # Diffusion (Ditto) with TensorRT/ONNX
                pred_kp_seq = self._call_np(kp_cond, aud_cond, sampling_timesteps)
                return pred_kp_seq, None


