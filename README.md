<h2 align='center'>Talking Head Project</h2>


## 🛠️ Environment

Tested Environment
- System: Centos 7.2  
- GPU: A100  
- Python: 3.10  


Clone the codes from [GitHub](https://github.com/antgroup/ditto-talkinghead) and switch to `train` branch :  
```bash
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead

git checkout train
```

Create `conda` environment:
```bash
conda env create -f environment.yaml -n ditto_train
conda activate ditto_train
```

***If you have trouble setting up the environment with Conda, feel free to use your preferred method based on the dependencies listed in [environment.yaml](environment.yaml).***  


## 📥 Checkpoints Preparation

To begin with, acquire the model used for data processing. This requirement is similar to the inference setup in the main branch. Please download the necessary checkpoints from [HuggingFace](https://huggingface.co/digital-avatar/ditto-talkinghead). 

```bash
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

For the preprocessing of training data, only the model located in the `ditto_pytorch` directory is required, as illustrated below:

```text
./checkpoints/
├── ...
└── ditto_pytorch
    ├── aux_models
    │   ├── 2d106det.onnx
    │   ├── det_10g.onnx
    │   ├── face_landmarker.task
    │   ├── hubert_streaming_fix_kv.onnx
    │   └── landmark203.onnx
    └── models
        ├── appearance_extractor.pth
        ├── decoder.pth
        ├── motion_extractor.pth
        ├── stitch_network.pth
        ├── warp_network.pth
        └── ...

```


## ⭕ Quick Start

To quickly get started, we have provided a few example videos under the `example/trainset_example` directory as training data. Before diving into the detailed steps, we will first use these example datasets to walk through the entire data processing and training pipeline.  

**Note: This quick start guide only uses a small amount of data and limited training steps to demonstrate the full workflow. To achieve reasonable generation performance, more high-quality training data and longer training are required.**  


```shell
DITTO_PATH="<your-ditto-talkinghead-absolute-path>"


## Prepare data_info.json
python example/get_data_info_json_for_trainset_example.py
# you will get `example/trainset_example/data_info.json`


## Process Videos into Training Features
DATA_INFO_JSON="${DITTO_PATH}/example/trainset_example/data_info.json"
DATA_LIST_JSON="${DITTO_PATH}/example/trainset_example/data_list.json"
DATA_PRELOAD_PKL="${DITTO_PATH}/example/trainset_example/data_preload.pkl"

bash prepare_data/prepare_data.sh ${DATA_INFO_JSON} ${DATA_LIST_JSON} ${DATA_PRELOAD_PKL}
# results in `example/trainset_example/`


## MotionDiT Training
cd MotionDiT

EXP_DIR="${DITTO_PATH}/example/exp_dir"
EXP_NAME="exp_trainset_example"

accelerate launch train.py \
    --experiment_dir ${EXP_DIR} \
    --experiment_name ${EXP_NAME} \
    --use_sc \
    --use_last_frame \
    --use_last_frame_loss \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --audio_feat_dim 1103 \
    --motion_feat_dim 265 \
    --batch_size 100 \
    --num_workers 8 \
    --epochs 3 \
    --save_ckpt_freq 1 \
    --data_list_json ${DATA_LIST_JSON} \
    --data_preload \
    --data_preload_pkl ${DATA_PRELOAD_PKL} \

# training outputs in `example/exp_dir/exp_trainset_example`
```


## 📁 Data Preparation

### Process Training Data

> **Before proceeding, ensure that your video data has been preprocessed. This includes cleaning, filtering, shot detection, and frame rate normalization to 25fps. The final videos should be in MP4 format with synchronized audio and video, and each frame should clearly show the target face.**  


**step 1: Prepare `data_info.json`**  
Based on your video data, manually create a corresponding `data_info.json` file (you can refer to the example: [example/get_data_info_json_for_trainset_example.py](example/get_data_info_json_for_trainset_example.py)).
The structure of the JSON file is as follows:

```python
# data_info.json

# Used to specify the mapping between all video files and their corresponding feature storage paths. All paths in this file must be absolute paths.

data_info = {
    'fps25_video_list': fps25_video_list,           # [*.mp4, ...], your video data
    'video_list': video_list,                       # [*.mp4, ...], cropped video
    'wav_list': wav_list,                           # [*.wav, ...], audio
    'hubert_aud_npy_list': hubert_aud_npy_list,     # [*.npy, ...], audio feat
    'LP_pkl_list': LP_pkl_list,                     # [*.pkl, ...], LP motion
    'LP_npy_list': LP_npy_list,                     # [*.npy, ...], LP motion
    'MP_lmk_npy_list': MP_lmk_npy_list,             # [*.npy, ...], MP lmk
    'eye_open_npy_list': eye_open_npy_list,         # [*.npy, ...], eye open state
    'eye_ball_npy_list': eye_ball_npy_list,         # [*.npy, ...], eye ball state
    'emo_npy_list': emo_npy_list,                   # [*.npy, ...], emo label
}

```

**Step 2: Process Videos into Training Features**  
Based on the `data_info.json`, process the raw videos into the feature format required for training.


```bash
DATA_INFO_JSON="<path-to-data-info-json>"       # input:  data_info.json
DATA_LIST_JSON="<path-to-data-list-json>"       # output: data_list.json    (for train)
DATA_PRELOAD_PKL="<path-to-data-preload-pkl>"   # output: data_preload.pkl  (for train)

bash prepare_data/prepare_data.sh ${DATA_INFO_JSON} ${DATA_LIST_JSON} ${DATA_PRELOAD_PKL}

```


## 🏋️ Model Training

Don't forget to run accelerate config to set up the default configuration for Accelerate, or include the appropriate accelerate arguments in the training command shown below.

```shell

cd MotionDiT

EXP_DIR="<path-to-experiment-dir>"
EXP_NAME="<experiment-name>"

DATA_LIST_JSON="<path-to-data-list-json>"
DATA_PRELOAD_PKL="<path-to-data-preload-pkl>"


accelerate launch train.py \
--experiment_dir ${EXP_DIR} \
--experiment_name ${EXP_NAME} \
--use_sc \
--use_last_frame \
--use_last_frame_loss \
--use_emo \
--use_eye_open \
--use_eye_ball \
--audio_feat_dim 1103 \
--motion_feat_dim 265 \
--batch_size 1024 \
--num_workers 8 \
--epochs 500 \
--save_ckpt_freq 1 \
--data_list_json ${DATA_LIST_JSON} \
--data_preload \
--data_preload_pkl ${DATA_PRELOAD_PKL} \

# You can find the training outputs in `${EXP_DIR}/${EXP_NAME}`.

```

