### Prepare dataset

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

#### Example

```
python prepare_dataset.py \
    --mmpose-path /home/kuba/workspace/pose_estimation/mmpose \
    --video-pose-3d-path /home/kuba/workspace/human_action_recognition/VideoPose3D \
    --input-directory input_data \
    --frame-height 720 \
    --frame-width 1280

python prepare_dataset.py \
    --mmpose-path /home/kuba/workspace/pose_estimation/mmpose \
    --video-pose-3d-path /home/kuba/workspace/human_action_recognition/VideoPose3D \
    --input-directory input_data \
    --frame-height 720 \
    --frame-width 1280 \
    --output-directory datasets_processed \
    --joints-count 17 \
    --joints-left 1 3 5 7 9 11 13 15 \
    --joints-right 2 4 6 8 10 12 14 16
```