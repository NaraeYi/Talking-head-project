# Ditto 학습 코드 구조 흐름

## 개요
`use_meanflow=False` (기본값)로 설정하면 **원본 Ditto Diffusion 구조**로 학습됩니다.
MeanFlow 코드는 완전히 분리되어 있어 원본 코드에 영향을 주지 않습니다.

---

## 학습 코드 실행 흐름

### 1. 진입점: `train.py`
```python
def main():
    opt = tyro.cli(TrainOptions)  # 명령줄 인자 파싱
    check_train_opt(opt)           # 옵션 검증
    trainer = Trainer(opt)          # Trainer 초기화
    trainer.train_loop()            # 학습 루프 시작
```

**실행 예시**:
```bash
python train.py \
    --experiment_name baseline_ditto \
    --data_list_json /path/to/data_list.json \
    --data_preload_pkl /path/to/data_preload.pkl \
    --data_preload \
    --use_emo --use_eye_open --use_eye_ball --use_sc
    # use_meanflow는 기본값 False이므로 명시하지 않아도 됨
```

---

### 2. Trainer 초기화: `trainer.py` → `__init__`

```python
class Trainer:
    def __init__(self, opt: TrainOptions):
        self.opt = opt
        self.use_meanflow = getattr(opt, "use_meanflow", False)  # 기본값: False
        
        # 1. Accelerate 초기화 (멀티 GPU 지원)
        self._init_accelerate()
        
        # 2. LMDM 모델 초기화 (원본 또는 MeanFlow)
        self.LMDM = self._init_LMDM()
        
        # 3. 데이터셋 초기화
        self.data_loader = self._init_dataset()
        
        # 4. 옵티마이저 초기화
        self.optim = self._init_optim()
        
        # 5. Accelerate 설정
        self._set_accelerate()
        
        # 6. 로깅 초기화
        self._init_log()
```

---

### 3. 모델 초기화: `_init_LMDM()`

#### `use_meanflow=False` (원본 Ditto)인 경우:
```python
def _init_LMDM(self):
    opt = self.opt
    
    # 원본 Ditto에 필요한 파라미터 로드
    part_w_dict = load_json(opt.part_w_dict_json) if opt.part_w_dict_json else None
    dim_ws = np.load(opt.dim_ws_npy) if opt.dim_ws_npy else None
    
    if self.use_meanflow:  # False이므로 이 분기는 실행되지 않음
        lmdm = MeanFlowLMDM(...)
    else:  # ✅ 이 분기 실행
        lmdm = LMDM(
            motion_feat_dim=opt.motion_feat_dim,
            audio_feat_dim=opt.audio_feat_dim,
            seq_frames=opt.seq_frames,
            part_w_dict=part_w_dict,      # 원본 Ditto 전용
            checkpoint=opt.checkpoint,
            device=self.device,
            use_last_frame_loss=opt.use_last_frame_loss,  # 원본 Ditto 전용
            use_reg_loss=opt.use_reg_loss,                # 원본 Ditto 전용
            dim_ws=dim_ws,                                # 원본 Ditto 전용
        )
    
    return lmdm
```

**결과**: 원본 `LMDM` (Diffusion 모델) 인스턴스 생성

---

### 4. 학습 루프: `train_loop()` → `_train_one_epoch()` → `_train_one_step()`

```python
def train_loop(self):
    for epoch in range(1, opt.epochs + 1):
        DAM = self._train_one_epoch()  # 한 에폭 학습
        
        if self.is_main_process:
            self._show_and_save(DAM)   # 로그 출력 및 체크포인트 저장

def _train_one_epoch(self):
    self.LMDM.train()  # 모델을 학습 모드로 설정
    
    for data_dict in data_loader:
        loss, loss_dict = self._train_one_step(data_dict)  # 한 스텝 학습
        self._loss_backward(loss)  # 역전파 및 옵티마이저 업데이트

def _train_one_step(self, data_dict):
    x = data_dict["kp_seq"]        # (B, L, kp_dim) - 타겟 키포인트 시퀀스
    cond_frame = data_dict["kp_cond"]  # (B, kp_dim) - 조건 프레임
    cond = data_dict["aud_cond"]    # (B, L, aud_dim) - 오디오 조건
    
    if self.use_meanflow:  # False이므로 이 분기는 실행되지 않음
        loss, loss_dict = self.LMDM.flow_matching_loss(x, cond_frame, cond)
    else:  # ✅ 이 분기 실행
        loss, loss_dict = self.LMDM.diffusion(
            x, cond_frame, cond, t_override=None
        )
    
    return loss, loss_dict
```

**결과**: 원본 Ditto의 `diffusion()` 메서드 호출 → Diffusion loss 계산

---

### 5. 원본 Ditto Loss 계산: `LMDM.diffusion()`

```python
# LMDM.py 내부
class LMDM:
    def diffusion(self, x, cond_frame, cond, t_override=None):
        """
        원본 Ditto Diffusion 학습
        
        Args:
            x: 타겟 키포인트 시퀀스 (B, L, kp_dim)
            cond_frame: 조건 프레임 (B, kp_dim)
            cond: 오디오 조건 (B, L, aud_dim)
            t_override: 타임스텝 오버라이드 (None이면 랜덤 샘플링)
        
        Returns:
            loss: 전체 loss
            loss_dict: 각 loss 구성 요소 (part별 loss 등)
        """
        # 1. 타임스텝 샘플링
        t = self.sample_timesteps(x.shape[0]) if t_override is None else t_override
        
        # 2. 노이즈 추가 (Forward Diffusion)
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t, noise)
        
        # 3. 모델 예측 (Denoising)
        pred = self.model(x_t, t, cond_frame, cond)
        
        # 4. Loss 계산
        loss = self.compute_loss(pred, noise, t, x)
        loss_dict = self.compute_loss_dict(pred, noise, t, x)
        
        return loss, loss_dict
```

---

## 코드 분기 요약

### `use_meanflow=False` (원본 Ditto) 경로:

```
train.py
  └─> Trainer.__init__()
       ├─> _init_LMDM()
       │    └─> LMDM() 생성 ✅ (원본 Diffusion 모델)
       │
       └─> train_loop()
            └─> _train_one_epoch()
                 └─> _train_one_step()
                      └─> LMDM.diffusion() ✅ (원본 Diffusion loss)
```

### `use_meanflow=True` (MeanFlow) 경로:

```
train.py
  └─> Trainer.__init__()
       ├─> _init_LMDM()
       │    └─> MeanFlowLMDM() 생성 ✅ (MeanFlow 모델)
       │
       └─> train_loop()
            └─> _train_one_epoch()
                 └─> _train_one_step()
                      └─> MeanFlowLMDM.flow_matching_loss() ✅ (Flow Matching loss)
```

---

## 확인 사항

### ✅ 원본 Ditto 학습이 정상 작동하는 이유:

1. **완전한 분리**: MeanFlow 코드는 `if self.use_meanflow:` 조건문으로 완전히 분리됨
2. **기본값**: `use_meanflow=False`가 기본값이므로 명시하지 않아도 원본 사용
3. **독립적인 모델**: `LMDM`과 `MeanFlowLMDM`은 완전히 별도의 클래스
4. **독립적인 Loss**: `diffusion()`과 `flow_matching_loss()`는 별도 메서드

### ✅ 원본 Ditto에만 필요한 파라미터:

- `part_w_dict`: 파트별 loss 가중치 (원본 전용)
- `use_last_frame_loss`: 마지막 프레임 loss 사용 여부 (원본 전용)
- `use_reg_loss`: 정규화 loss 사용 여부 (원본 전용)
- `dim_ws`: 차원별 가중치 (원본 전용)

이 파라미터들은 MeanFlow에서는 사용되지 않습니다.

---

## 학습 실행 예시

### 원본 Ditto 학습 (Baseline):
```bash
cd /root/ditto/ditto-talkinghead-train/MotionDiT

python train.py \
    --experiment_name baseline_ditto \
    --data_list_json /root/ditto/datasets/HDTF/HDTF_train/data_list.json \
    --data_preload_pkl /root/ditto/datasets/HDTF/HDTF_train/data_preload.pkl \
    --data_preload \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --use_sc \
    --motion_feat_dim 265 \
    --audio_feat_dim 1103 \
    --seq_frames 80 \
    --batch_size 512 \
    --epochs 1000 \
    --lr 1e-4
    # use_meanflow는 명시하지 않으면 False (기본값)
```

### MeanFlow 학습 (비교용):
```bash
python train.py \
    --experiment_name meanflow_ditto \
    --data_list_json /root/ditto/datasets/HDTF/HDTF_train/data_list.json \
    --data_preload_pkl /root/ditto/datasets/HDTF/HDTF_train/data_preload.pkl \
    --data_preload \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --use_sc \
    --motion_feat_dim 265 \
    --audio_feat_dim 1103 \
    --seq_frames 80 \
    --batch_size 512 \
    --epochs 1000 \
    --lr 1e-4 \
    --use_meanflow True \
    --time_sampler logit_normal
```

---

## 결론

✅ **`use_meanflow=False` (기본값)로 설정하면 원본 Ditto Diffusion 구조로 학습됩니다.**

- MeanFlow 코드는 완전히 분리되어 있어 원본 코드에 영향을 주지 않음
- 원본 Ditto의 모든 기능 (part loss, reg loss 등) 정상 작동
- Baseline 학습을 먼저 돌려도 문제없음

