## Demo

1. [Prepare dataset](#1-prepare-dataset)
2. [Analyse video](#2-analyse-video)
3. [Real time analysis](#3-real-time-analysis)


### 1. Prepare dataset

**prepare_dataset.py** - script used to prepare dataset - estimate 2D coordinates of keypoints using MMPose and generate 3D coordinates basis on them, using program VideoPose3D

- *mmpose-path* - Absolute path to MMPose **required**
- *video-pose-3d-path* - Absolute path to VideoPose3D **required**
- *input-directory* - Path to input data **required**
- *frame-width* - Width of frame **required**
- *frame-height* - Height of frame **required**
- *output-directory* - Path to generated results output directory (default='results')
- *joints-count* - Count of joints given as input (by default 17)
- *joints-left* - Joints from left part of silhouette (by default [1, 3, 5, 7, 9, 11, 13, 15])
- *joints-right* - Joints from right part of silhouette (by default [2, 4, 6, 8, 10, 12, 14, 16])

Example structure of input:
```
dataset/
├─ cluster1/
│  ├─ cam1/
│  │  ├─ action1/
│  │  │  ├─ video_001.avi
│  │  │  ├─ video_002.avi
│  │  │  ├─ video_003.avi
│  │  ├─ action2/
│  │  │  ├─ video_001.avi
│  │  │  ├─ video_002.avi
│  │  │  ├─ video_003.avi
│  ├─ cam2/
│  │  ├─ action1/
│  │  │  ├─ video_001.avi
│  │  │  ├─ video_002.avi
│  │  │  ├─ video_003.avi
│  │  ├─ action2/
│  │  │  ├─ video_001.avi
│  │  │  ├─ video_002.avi
│  │  │  ├─ video_003.avi
```

#### Example

```
python prepare_dataset.py \
    --mmpose-path /mmpose \
    --video-pose-3d-path /VideoPose3D \
    --input-directory input_data \
    --frame-height 720 \
    --frame-width 1280

python prepare_dataset.py \
    --mmpose-path /mmpose \
    --video-pose-3d-path /VideoPose3D \
    --input-directory input_data \
    --frame-height 720 \
    --frame-width 1280 \
    --output-directory datasets_processed \
    --joints-count 17 \
    --joints-left 1 3 5 7 9 11 13 15 \
    --joints-right 2 4 6 8 10 12 14 16
```


### 2. Analyse video

**run_video_analysis.py** - script used to run analysis of video

- *har-model-path* - path to har trained model **required** 
- *mmpose-path* - Absolute path to MMPose **required**
- *video-pose-3d-path* - Absolute path to VideoPose3D **required**
- *joints-count* - Count of joints given as input (by default 17)
- *joints-left* - Joints from left part of silhouette (by default [1, 3, 5, 7, 9, 11, 13, 15])
- *joints-right* - Joints from right part of silhouette (by default [2, 4, 6, 8, 10, 12, 14, 16])


#### Example

```
python run_video_analysis.py \
    --har-model-path /har-model-path.pth \
    --mmpose-path /mmpose \
    --video-pose-3d-path /VideoPose3D \
    --video-path /multiple_exercises_2.mp4
```

### 3. Real time analysis

**run_real_time_analysis.py** - script used to run real time analysis from camera

- *har-model-path* - path to har trained model **required** 
- *mmpose-path* - Absolute path to MMPose **required**
- *video-pose-3d-path* - Absolute path to VideoPose3D **required**
- *joints-count* - Count of joints given as input (by default 17)
- *joints-left* - Joints from left part of silhouette (by default [1, 3, 5, 7, 9, 11, 13, 15])
- *joints-right* - Joints from right part of silhouette (by default [2, 4, 6, 8, 10, 12, 14, 16])


#### Example

```
python run_real_time_analysis.py \
    --har-model-path /har-model-path.pth \
    --mmpose-path /mmpose \
    --video-pose-3d-path /VideoPose3D \
    --video-path /multiple_exercises_2.mp4
```

### 4. Run evaluation

**run_evaluation.py** - demo evaluation script


### 5. Run fit

**run_fit.py** - demo fit script


### 6. Run training

**run_training.py** - demo training script
