#!/bin/bash
# VFHQ 데이터셋 전처리 스크립트 (GPU 버전)
# GPU를 명시적으로 사용하고 백그라운드 실행 가능
# 사용법: nohup bash prepare_data_gpu.sh [data_info_json] [data_list_json] [data_preload_pkl] [gpu_device_id] > prepare_data.log 2>&1 &

set -euo pipefail

###############################################
# 경로 설정 (여기서 직접 수정 가능)
###############################################

# 데이터셋 경로 설정
DATA_DIR="/workspace/ditto/datasets/VFHQ/VFHQ_feat"

# 입력 파일 (이미 존재해야 함 - get_data_info_json_for_vfhq.py로 생성)
DATA_INFO_JSON="${DATA_DIR}/data_info.json"

# 출력 파일 (스크립트 실행 시 자동 생성됨)
DATA_LIST_JSON="${DATA_DIR}/data_list.json"      # Step 7에서 생성
DATA_PRELOAD_PKL="${DATA_DIR}/data_preload.pkl"  # Step 8에서 생성

# GPU device_id 설정 (단일 GPU 사용 시)
# 병렬 처리 사용 시: USE_PARALLEL_GPU=true로 설정
USE_PARALLEL_GPU=true
GPU_DEVICE_ID=0  # 단일 GPU 모드에서 사용

# 병렬 GPU 할당 (USE_PARALLEL_GPU=true일 때)
GPU_MOTION_ORIG=0   # Motion feature (원본)
GPU_MOTION_FLIP=1   # Motion feature (flip)
GPU_EMOTION=2       # Emotion feature
# Eye feature는 CPU만 사용 (MediaPipe 제한)

###############################################
# conda 환경 활성화 (자동으로 찾기)
###############################################
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

# conda activate 

SECONDS=0

DITTO_ROOT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"

DITTO_PYTORCH_PATH="${DITTO_ROOT_DIR}/checkpoints/ditto_pytorch"
HUBERT_ONNX="${DITTO_PYTORCH_PATH}/aux_models/hubert_streaming_fix_kv.onnx"
MP_FACE_LMK_TASK="${DITTO_PYTORCH_PATH}/aux_models/face_landmarker.task"

cd "${DITTO_ROOT_DIR}/prepare_data_train"

###############################################
# 인자 처리 (인자가 제공되면 기본값 오버라이드)
###############################################

# 인자가 제공되면 사용, 없으면 위에서 설정한 기본값 사용
if [ $# -ge 1 ] && [ -n "$1" ]; then
    DATA_INFO_JSON="$1"
fi

if [ $# -ge 2 ] && [ -n "$2" ]; then
    DATA_LIST_JSON="$2"
fi

if [ $# -ge 3 ] && [ -n "$3" ]; then
    DATA_PRELOAD_PKL="$3"
fi

if [ $# -ge 4 ] && [ -n "$4" ]; then
    GPU_DEVICE_ID="$4"
fi

# 환경변수로도 오버라이드 가능
DATA_INFO_JSON=${DATA_INFO_JSON_ENV:-${DATA_INFO_JSON}}
DATA_LIST_JSON=${DATA_LIST_JSON_ENV:-${DATA_LIST_JSON}}
DATA_PRELOAD_PKL=${DATA_PRELOAD_PKL_ENV:-${DATA_PRELOAD_PKL}}
GPU_DEVICE_ID=${GPU_DEVICE_ID_ENV:-${GPU_DEVICE_ID}}
USE_PARALLEL_GPU=${USE_PARALLEL_GPU_ENV:-${USE_PARALLEL_GPU}}
GPU_MOTION_ORIG=${GPU_MOTION_ORIG_ENV:-${GPU_MOTION_ORIG}}
GPU_MOTION_FLIP=${GPU_MOTION_FLIP_ENV:-${GPU_MOTION_FLIP}}
GPU_EMOTION=${GPU_EMOTION_ENV:-${GPU_EMOTION}}

# 변수명 통일
data_info_json="${DATA_INFO_JSON}"
data_list_json="${DATA_LIST_JSON}"
data_preload_pkl="${DATA_PRELOAD_PKL}"
DATA_DIR="$(dirname "${data_info_json}")"

# data_info_json 파일 존재 확인
if [ ! -f "${data_info_json}" ]; then
    echo "ERROR: data_info.json file not found: ${data_info_json}"
    echo ""
    echo "Please check:"
    echo "  1. File path is correct (edit the script to set VFHQ_DATA_DIR)"
    echo "  2. Run get_data_info_json_for_vfhq.py first to generate data_info.json"
    echo ""
    echo "Example:"
    echo "  python /workspace/ditto/datasets/VFHQ/get_data_info_json_for_vfhq.py \\"
    echo "      --source_dir /workspace/ditto/datasets/VFHQ/videos/cropped_videos_28038 \\"
    echo "      --output_dir ${VFHQ_DATA_DIR} \\"
    echo "      --data_info_json ${data_info_json}"
    exit 1
fi

echo "================================================"
echo "VFHQ Feature Extraction (GPU Version)"
echo "================================================"
echo "GPU Device ID: ${GPU_DEVICE_ID}"
echo "Data Info JSON: ${data_info_json}"
echo "Data List JSON: ${data_list_json}"
echo "Data Preload PKL: ${data_preload_pkl}"
echo "Start Time: $(date)"
echo "================================================"

# check ckpt
echo ""
echo "[Step 0] Checking checkpoints..."
python scripts/check_ckpt_path.py --ditto_pytorch_path ${DITTO_PYTORCH_PATH}

###############################################
# process data

### VIDEO ###
# crop video: 'fps25_video_list' -> 'video_list'
echo ""
echo "[Step 1] Cropping videos..."
# crop_video_by_LP.py는 device_id 옵션을 지원하지 않음
CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} python scripts/crop_video_by_LP.py \
    -i "${data_info_json}" \
    --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"
    # --device_id ${GPU_DEVICE_ID}

### AUDIO ###
# extract audio: 'video_list' -> 'wav_list'
echo ""
echo "[Step 2] Extracting audio from videos..."
python scripts/extract_audio_from_video.py -i "${data_info_json}"

### Feature ###
# audio feat: 'wav_list' -> 'hubert_aud_npy_list'
# HuBERT는 자동으로 GPU 사용 (CUDAExecutionProvider)
echo ""
echo "[Step 3] Extracting audio features (HuBERT - GPU auto)..."
python scripts/extract_audio_feat_by_Hubert.py \
    -i "${data_info_json}" \
    --Hubert_onnx "${HUBERT_ONNX}"

# motion feat: 'video_list' -> {'LP_pkl_list', 'LP_npy_list'} (_flip)
# eye feat: 'video_list' -> {'MP_lmk_npy_list', 'eye_open_npy_list', 'eye_ball_npy_list'} (_flip)
# emo feat: 'video_list' -> 'emo_npy_list'

if [ "${USE_PARALLEL_GPU}" = "true" ]; then
    # 병렬 처리 모드: 여러 GPU를 동시에 사용
    echo ""
    echo "================================================"
    echo "병렬 GPU 처리 모드 활성화"
    echo "================================================"
    echo "GPU 할당:"
    echo "  GPU ${GPU_MOTION_ORIG}: Motion (원본)"
    echo "  GPU ${GPU_MOTION_FLIP}: Motion (flip)"
    echo "  GPU ${GPU_EMOTION}: Emotion"
    echo "  CPU: Eye (MediaPipe)"
    echo "================================================"
    
    # 로그 디렉토리 생성
    LOG_DIR="${DATA_DIR}/parallel_logs"
    mkdir -p "${LOG_DIR}"
    MOTION_ORIG_LOG="${LOG_DIR}/motion_original.log"
    MOTION_FLIP_LOG="${LOG_DIR}/motion_flip.log"
    EYE_LOG="${LOG_DIR}/eye.log"
    EMOTION_LOG="${LOG_DIR}/emotion.log"
    
    # Step 4-1: Motion 특징 추출 (원본) - GPU 0 (백그라운드)
    echo ""
    echo "[Step 4-1] Extracting motion features (original) - GPU ${GPU_MOTION_ORIG}..."
    (
        echo "Motion 원본 처리 시작: $(date)"
        CUDA_VISIBLE_DEVICES=${GPU_MOTION_ORIG} python scripts/extract_motion_feat_by_LP.py \
            -i "${data_info_json}" \
            --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
            --device_id ${GPU_MOTION_ORIG} \
            > "${MOTION_ORIG_LOG}" 2>&1
        echo "Motion 원본 완료: $(date)"
    ) &
    MOTION_ORIG_PID=$!
    
    # Step 4-2: Motion 특징 추출 (flip) - GPU 1 (백그라운드)
    echo "[Step 4-2] Extracting motion features (flip) - GPU ${GPU_MOTION_FLIP}..."
    (
        echo "Motion flip 처리 시작: $(date)"
        CUDA_VISIBLE_DEVICES=${GPU_MOTION_FLIP} python scripts/extract_motion_feat_by_LP.py \
            -i "${data_info_json}" \
            --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
            --flip_flag \
            --device_id ${GPU_MOTION_FLIP} \
            > "${MOTION_FLIP_LOG}" 2>&1
        echo "Motion flip 완료: $(date)"
    ) &
    MOTION_FLIP_PID=$!
    
    # Step 5: Eye 특징 추출 (CPU) - MediaPipe는 GPU 미지원 (백그라운드)
    echo "[Step 5] Extracting eye features - CPU (MediaPipe)..."
    (
        echo "Eye 원본 처리 시작: $(date)"
        python scripts/extract_eye_ratio_from_video.py \
            -i "${data_info_json}" \
            --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" \
            > "${EYE_LOG}.original" 2>&1
        
        echo "Eye flip 처리 시작: $(date)"
        python scripts/extract_eye_ratio_from_video.py \
            -i "${data_info_json}" \
            --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" \
            --flip_lmk_flag \
            >> "${EYE_LOG}.flip" 2>&1
        
        echo "Eye 완료: $(date)"
    ) > "${EYE_LOG}" 2>&1 &
    EYE_PID=$!
    
    # Step 6: Emotion 특징 추출 (GPU 2) (백그라운드)
    echo "[Step 6] Extracting emotion features - GPU ${GPU_EMOTION}..."
    (
        echo "Emotion 처리 시작: $(date)"
        CUDA_VISIBLE_DEVICES=${GPU_EMOTION} python scripts/extract_emo_feat_from_video.py \
            -i "${data_info_json}" \
            --device_id ${GPU_EMOTION} \
            > "${EMOTION_LOG}" 2>&1
        echo "Emotion 완료: $(date)"
    ) &
    EMOTION_PID=$!
    
    echo ""
    echo "================================================"
    echo "병렬 프로세스 실행 중..."
    echo "================================================"
    echo "Motion (원본) PID: ${MOTION_ORIG_PID} (GPU ${GPU_MOTION_ORIG})"
    echo "Motion (flip) PID: ${MOTION_FLIP_PID} (GPU ${GPU_MOTION_FLIP})"
    echo "Eye PID: ${EYE_PID} (CPU)"
    echo "Emotion PID: ${EMOTION_PID} (GPU ${GPU_EMOTION})"
    echo ""
    echo "진행 상황 모니터링:"
    echo "  tail -f ${MOTION_ORIG_LOG}"
    echo "  tail -f ${MOTION_FLIP_LOG}"
    echo "  tail -f ${EYE_LOG}"
    echo "  tail -f ${EMOTION_LOG}"
    echo ""
    echo "모든 프로세스 완료 대기 중..."
    echo "================================================"
    
    # 모든 프로세스 완료 대기
    wait ${MOTION_ORIG_PID}
    MOTION_ORIG_EXIT=$?
    echo "Motion (원본) 완료 (exit code: ${MOTION_ORIG_EXIT})"
    
    wait ${MOTION_FLIP_PID}
    MOTION_FLIP_EXIT=$?
    echo "Motion (flip) 완료 (exit code: ${MOTION_FLIP_EXIT})"
    
    wait ${EYE_PID}
    EYE_EXIT=$?
    echo "Eye 완료 (exit code: ${EYE_EXIT})"
    
    wait ${EMOTION_PID}
    EMOTION_EXIT=$?
    echo "Emotion 완료 (exit code: ${EMOTION_EXIT})"
    
    # 에러 체크
    if [ ${MOTION_ORIG_EXIT} -ne 0 ] || [ ${MOTION_FLIP_EXIT} -ne 0 ] || [ ${EYE_EXIT} -ne 0 ] || [ ${EMOTION_EXIT} -ne 0 ]; then
        echo ""
        echo "❌ 일부 프로세스가 실패했습니다!"
        echo "Motion (원본) exit code: ${MOTION_ORIG_EXIT}"
        echo "Motion (flip) exit code: ${MOTION_FLIP_EXIT}"
        echo "Eye exit code: ${EYE_EXIT}"
        echo "Emotion exit code: ${EMOTION_EXIT}"
        echo ""
        echo "로그 확인:"
        echo "  ${MOTION_ORIG_LOG}"
        echo "  ${MOTION_FLIP_LOG}"
        echo "  ${EYE_LOG}"
        echo "  ${EMOTION_LOG}"
        exit 1
    fi
    
    echo ""
    echo "================================================"
    echo "병렬 GPU 처리 완료!"
    echo "================================================"
else
    # 단일 GPU 모드: 순차 처리
    echo ""
    echo "[Step 4-1] Extracting motion features (original) - GPU ${GPU_DEVICE_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} python scripts/extract_motion_feat_by_LP.py \
        -i "${data_info_json}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
        --device_id ${GPU_DEVICE_ID}
    
    echo ""
    echo "[Step 4-2] Extracting motion features (flip) - GPU ${GPU_DEVICE_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} python scripts/extract_motion_feat_by_LP.py \
        -i "${data_info_json}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
        --flip_flag \
        --device_id ${GPU_DEVICE_ID}
    
    # eye feat: 'video_list' -> {'MP_lmk_npy_list', 'eye_open_npy_list', 'eye_ball_npy_list'} (_flip)
    # MediaPipe는 CPU만 사용
    echo ""
    echo "[Step 5-1] Extracting eye features (original) - CPU (MediaPipe)..."
    python scripts/extract_eye_ratio_from_video.py \
        -i "${data_info_json}" \
        --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}"
    
    echo ""
    echo "[Step 5-2] Extracting eye features (flip) - CPU (MediaPipe)..."
    python scripts/extract_eye_ratio_from_video.py \
        -i "${data_info_json}" \
        --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" \
        --flip_lmk_flag
    
    # emo feat: 'video_list' -> 'emo_npy_list'
    echo ""
    echo "[Step 6] Extracting emotion features - GPU ${GPU_DEVICE_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID} python scripts/extract_emo_feat_from_video.py \
        -i "${data_info_json}" \
        --device_id ${GPU_DEVICE_ID}
fi

###############################################
# get data_list_json for train
echo ""
echo "[Step 7] Gathering data list JSON for training..."
python scripts/gather_data_list_json_for_train.py \
    -i "${data_info_json}" \
    -o "${data_list_json}" \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --with_flip

# get preload data_pkl ([option] for faster training speed)
echo ""
echo "[Step 8] Preloading training data to PKL..."
python scripts/preload_train_data_to_pkl.py \
    --data_list_json "${data_list_json}" \
    --data_preload_pkl "${data_preload_pkl}" \
    --use_sc \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --motion_feat_dim 265

cd "${DITTO_ROOT_DIR}"

###############################################

echo ""
echo "================================================"
echo "[prepare_data_gpu] DONE"
echo "================================================"
echo "data_list_json: ${data_list_json}"
echo "data_preload_pkl: ${data_preload_pkl}"
echo "End Time: $(date)"
echo "Elapsed time: $SECONDS seconds ($(($SECONDS / 60)) minutes)"
echo "================================================"
