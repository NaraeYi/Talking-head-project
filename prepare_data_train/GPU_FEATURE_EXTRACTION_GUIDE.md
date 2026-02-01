# VFHQ Feature Extraction GPU 사용 가이드

## GPU 환경 확인

현재 시스템에는 **4개의 NVIDIA A100-SXM4-80GB GPU**가 설치되어 있습니다.

## Feature 추출 모듈별 GPU 지원 현황

### ✅ GPU 지원 모듈

#### 1. Motion Feature (LivePortrait)
- **GPU 지원**: ✅ Yes
- **사용 방법**: `--device_id` 파라미터 사용
- **스크립트**: `scripts/extract_motion_feat_by_LP.py`
- **예시**:
  ```bash
  python scripts/extract_motion_feat_by_LP.py \
    -i data_info.json \
    --ditto_pytorch_path /path/to/ditto_pytorch \
    --device_id 0
  ```

#### 2. Audio Feature (HuBERT)
- **GPU 지원**: ✅ Yes (자동)
- **사용 방법**: ONNX Runtime의 CUDAExecutionProvider 자동 사용
- **스크립트**: `scripts/extract_audio_feat_by_Hubert.py`
- **상태**: CUDAExecutionProvider가 사용 가능하면 자동으로 GPU 사용
- **확인**: ONNX Runtime에서 `CUDAExecutionProvider` 사용 가능 확인됨

#### 3. Emotion Feature
- **GPU 지원**: ✅ Yes
- **사용 방법**: `--device_id` 파라미터 사용
- **스크립트**: `scripts/extract_emo_feat_from_video.py`
- **예시**:
  ```bash
  python scripts/extract_emo_feat_from_video.py \
    -i data_info.json \
    --device_id 0
  ```

### ❌ CPU만 사용하는 모듈

#### 4. Eye Feature (MediaPipe)
- **GPU 지원**: ❌ No
- **사용 방법**: CPU만 사용 (MediaPipe 제한)
- **스크립트**: `scripts/extract_eye_ratio_from_video.py`
- **참고**: MediaPipe는 CPU에서만 실행됩니다

## prepare_data.sh 수정 방법

현재 `prepare_data.sh`는 GPU device_id를 지정하지 않습니다. GPU를 사용하려면 다음과 같이 수정할 수 있습니다:

### 옵션 1: CUDA_VISIBLE_DEVICES 환경 변수 사용

```bash
# GPU 0 사용
export CUDA_VISIBLE_DEVICES=0
bash prepare_data.sh data_info.json data_list.json data_preload.pkl

# GPU 1 사용
export CUDA_VISIBLE_DEVICES=1
bash prepare_data.sh data_info.json data_list.json data_preload.pkl
```

### 옵션 2: 스크립트에 device_id 파라미터 추가

`prepare_data.sh`를 수정하여 각 스크립트에 `--device_id` 파라미터를 추가:

```bash
# Motion feature extraction (GPU 0 사용)
python scripts/extract_motion_feat_by_LP.py \
  -i "${data_info_json}" \
  --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
  --device_id 0

python scripts/extract_motion_feat_by_LP.py \
  -i "${data_info_json}" \
  --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
  --flip_flag \
  --device_id 0

# Emotion feature extraction (GPU 0 사용)
python scripts/extract_emo_feat_from_video.py \
  -i "${data_info_json}" \
  --device_id 0
```

### 옵션 3: 병렬 처리 (여러 GPU 사용)

여러 GPU를 사용하여 병렬로 처리할 수 있습니다:

```bash
# GPU 0에서 motion feature 추출
CUDA_VISIBLE_DEVICES=0 python scripts/extract_motion_feat_by_LP.py \
  -i "${data_info_json}" \
  --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
  --device_id 0 &

# GPU 1에서 emotion feature 추출
CUDA_VISIBLE_DEVICES=1 python scripts/extract_emo_feat_from_video.py \
  -i "${data_info_json}" \
  --device_id 0 &

wait  # 모든 작업 완료 대기
```

## 성능 최적화 팁

1. **여러 GPU 활용**: 4개의 A100 GPU를 활용하여 다른 feature를 병렬로 추출
2. **배치 처리**: 여러 비디오를 한 번에 처리하도록 스크립트 수정 고려
3. **MediaPipe 최적화**: Eye feature는 CPU만 사용하므로 별도 프로세스로 분리하여 다른 GPU 작업과 병렬 실행

## 확인 사항

- ✅ ONNX Runtime: CUDAExecutionProvider 사용 가능
- ✅ GPU 하드웨어: 4x A100 80GB 확인됨
- ⚠️ PyTorch: conda 환경에서 확인 필요 (ditto_train 환경)

## 다음 단계

1. conda 환경 `ditto_train` 활성화
2. `prepare_data.sh`에 GPU device_id 파라미터 추가
3. 작은 샘플로 GPU 사용 여부 테스트
4. 전체 데이터셋에 대해 병렬 처리 설정


