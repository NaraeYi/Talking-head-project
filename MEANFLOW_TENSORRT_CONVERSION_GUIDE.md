# MeanFlow PyTorch → TensorRT 변환 가이드

## 개요
PyTorch 기반 MeanFlow 가중치(`train_100.pt`)를 TensorRT 엔진으로 변환하여 추론 속도를 향상시키는 과정입니다.

---

## 변환 과정 (3단계)

### **1단계: PyTorch → ONNX 변환**

#### 실행 명령어
```bash
cd /workspace/ditto/ditto-talkinghead-train

python scripts/export_meanflow_onnx.py \
    --checkpoint_path /ditto_meanflow_hdtf_20251229_225128/samples/train_100.pt \
    --output_onnx_path checkpoints/ditto_onnx/lmdm_meanflow.onnx \
    --motion_feat_dim 265 \
    --audio_feat_dim 1103 \
    --seq_frames 80 \
    --device cuda
```

#### 사용 파일
- **`scripts/export_meanflow_onnx.py`**: PyTorch 모델을 ONNX로 변환하는 스크립트
  - MeanFlow 모델 로드 및 state_dict 처리
  - `MeanFlowWrapper`로 ONNX export용 래퍼 생성
  - 입력: `(e, cond_frame, cond, r, t, cond_drop_prob)`
  - 출력: `u` (velocity prediction)

#### 주의사항
- `audio_feat_dim`은 체크포인트에 맞춰 설정 (기본값: 1103)
- 체크포인트 키에 `model.` prefix가 없으면 자동 추가
- `cond_drop_prob`는 ONNX export 시 0.0으로 고정

---

### **2단계: ONNX → TensorRT 엔진 변환**

#### 방법 A: `cvt_onnx_to_trt.py` 사용 (권장)
```bash
cd /workspace/ditto/ditto-talkinghead-train

python scripts/cvt_onnx_to_trt.py \
    --onnx_dir checkpoints/ditto_onnx \
    --trt_dir checkpoints/ditto_trt_Ampere_Plus
```

**주의**: `cvt_onnx_to_trt.py`는 디렉토리 내 모든 `.onnx` 파일을 변환합니다. MeanFlow만 변환하려면 방법 B 사용.

#### 방법 B: `polygraphy` 직접 사용
```bash
cd /workspace/ditto/ditto-talkinghead-train

polygraphy convert \
    checkpoints/ditto_onnx/lmdm_meanflow.onnx \
    -o checkpoints/ditto_trt_Ampere_Plus/lmdm_meanflow_fp32.engine \
    --hardware-compatibility-level=Ampere_Plus \
    --builder-optimization-level=5
```

#### 사용 파일
- **`scripts/cvt_onnx_to_trt.py`**: ONNX → TensorRT 변환 스크립트
  - `polygraphy` 또는 TensorRT Python API 사용
  - FP16/FP32 선택 가능
  - Hardware compatibility level 설정

#### 주의사항
- TensorRT 변환 시 `MultiheadAttention` 내부 GEMM 연산에서 오류가 발생할 수 있음
- 오류 발생 시 ONNX 모델 최적화 또는 TensorRT 버전 확인 필요

---

### **3단계: 코드 수정 (TensorRT 지원 추가)**

#### 3-1. `core/models/lmdm.py` 수정 ✅ (이미 완료됨)

**수정 내용**:
1. `__init__` 메서드 (라인 38-42): MeanFlow TensorRT 에러 제거
2. `_init_meanflow_trt()` 메서드 추가 (라인 54-58): MeanFlow TensorRT 초기화
3. `_meanflow_sample_np()` 메서드 추가 (라인 149-223): MeanFlow TensorRT/ONNX 추론 로직
4. `__call__` 메서드 (라인 243-251): MeanFlow TensorRT 분기 추가

**주요 로직**:
- MeanFlow 1-step sampling: `x = e - u(e, r=0, t=1)`
- 입력: `(e, cond_frame, cond, r, t, cond_drop_prob)`
- 출력: `u` (velocity prediction)
- 최종 출력: `pred_kp_seq = e - u`

---

#### 3-2. `core/atomic_components/cfg.py` 수정 (선택사항)

**목적**: MeanFlow TensorRT 엔진 경로를 자동으로 설정

**수정 위치**: `parse_cfg` 함수 (라인 83-91)

**현재 코드**:
```python
# checkpoint_path가 제공되면 model_path를 대체
if isinstance(replace_cfg, dict) and "checkpoint_path" in replace_cfg:
    checkpoint_path = replace_cfg["checkpoint_path"]
    if checkpoint_path:
        lmdm_cfg["model_path"] = checkpoint_path
```

**수정 후** (MeanFlow TensorRT 우선 사용):
```python
# checkpoint_path가 제공되면 model_path를 대체
if isinstance(replace_cfg, dict) and "checkpoint_path" in replace_cfg:
    checkpoint_path = replace_cfg["checkpoint_path"]
    if checkpoint_path:
        original_model_path = lmdm_cfg["model_path"]
        is_tensorrt_or_onnx = (
            original_model_path.endswith(".engine") or
            original_model_path.endswith(".trt") or
            original_model_path.endswith(".onnx")
        )
        
        # MeanFlow + TensorRT: TensorRT 엔진이 명시적으로 제공되면 사용
        if lmdm_cfg.get("use_meanflow", False) and is_tensorrt_or_onnx:
            # TensorRT 엔진 경로 유지 (model_path)
            pass
        elif lmdm_cfg.get("use_meanflow", False):
            # MeanFlow + PyTorch: checkpoint_path 사용
            lmdm_cfg["model_path"] = checkpoint_path
        elif is_tensorrt_or_onnx:
            # Ditto + TensorRT: TensorRT 엔진 우선 사용
            pass
        else:
            # TensorRT/ONNX가 아니면 checkpoint_path로 대체
            lmdm_cfg["model_path"] = checkpoint_path
```

**참고**: 이 수정은 선택사항입니다. 추론 시 `model_path`를 직접 TensorRT 엔진 경로로 설정하면 됩니다.

---

## 4단계: 추론 시 TensorRT 엔진 사용

### 방법 A: `cfg_pkl` 수정
`cfg_pkl` 파일에서 `audio2motion_cfg["model_path"]`를 TensorRT 엔진 경로로 설정:
```python
audio2motion_cfg = {
    "model_path": "checkpoints/ditto_trt_Ampere_Plus/lmdm_meanflow_fp32.engine",
    "use_meanflow": True,
    ...
}
```

### 방법 B: 추론 스크립트에서 `replace_cfg` 사용
```python
replace_cfg = {
    "audio2motion_cfg": {
        "model_path": "checkpoints/ditto_trt_Ampere_Plus/lmdm_meanflow_fp32.engine",
        "use_meanflow": True,
    }
}
```

### 방법 C: `checkpoint_path` 대신 `model_path` 직접 지정
```bash
python inference.py \
    --cfg_pkl configs/your_config.pkl \
    --model_path checkpoints/ditto_trt_Ampere_Plus/lmdm_meanflow_fp32.engine \
    --use_meanflow True \
    ...
```

---

## 요약: 수정해야 할 파일

1. **`scripts/export_meanflow_onnx.py`** ✅ (이미 존재, 사용만 하면 됨)
2. **`scripts/cvt_onnx_to_trt.py`** ✅ (이미 존재, 사용만 하면 됨)
3. **`core/models/lmdm.py`** ✅ (수정 완료)
   - `__init__` 메서드: MeanFlow TensorRT 에러 제거
   - `_init_meanflow_trt()` 메서드 추가
   - `_meanflow_sample_np()` 메서드 추가
   - `__call__` 메서드: MeanFlow TensorRT 분기 추가
4. **`core/atomic_components/cfg.py`** ⚠️ (선택사항, TensorRT 경로 자동 설정용)

---

## 검증

변환 후 다음 명령어로 테스트:
```bash
python inference_time_detail.py \
    --cfg_pkl configs/your_config.pkl \
    --checkpoint_path /ditto_meanflow_hdtf_20251229_225128/samples/train_100.pt \
    --use_meanflow True \
    --audio_path test.wav \
    --source_path test.jpg \
    --output_path output.mp4
```

**확인 사항**:
- `[LMDM] Model type: tensorrt, use_meanflow: True` 출력 확인
- 추론 시간이 PyTorch보다 빠른지 확인
- 출력 비디오 품질 확인

---

## 문제 해결

### 1. ONNX 변환 실패
- `audio_feat_dim` 값 확인 (체크포인트에 맞춰야 함)
- 체크포인트 키에 `model.` prefix 있는지 확인

### 2. TensorRT 변환 실패
- `MultiheadAttention` GEMM 오류: ONNX 모델 최적화 시도 또는 TensorRT 버전 확인
- `polygraphy` 버전 확인

### 3. 추론 시 오류
- TensorRT 엔진 입력/출력 이름 확인 (`e`, `cond_frame`, `cond`, `r`, `t`, `cond_drop_prob` → `u`)
- `_meanflow_sample_np()` 로직이 PyTorch 버전과 일치하는지 확인

---

## 참고

- MeanFlow는 1-step 생성 모델이므로 `sampling_timesteps`는 무시됨
- `shared_noise`는 여러 클립 간 일관성을 위해 사용 (선택사항)
- TensorRT 엔진은 특정 GPU 아키텍처에 최적화되므로, 다른 GPU에서 사용 시 재변환 필요

