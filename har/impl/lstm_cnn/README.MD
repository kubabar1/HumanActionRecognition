## LSTM CNN

This implementation basis on article **"Skeleton-based action recognition using LSTM and CNN"**.

1. [Evaluation all](#1-evaluation)

2. [Additional info](#3-additional-info)

3. [Article](#4-article)


### 1. Evaluation

| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **classes*** | - | list of classes (class ID must comply with labels ID's) - order must agree with this used during training NN |
| **test_data*** | - | list of test data sequences e.g. list containing 100 sequences of size (125, 17, 3) |
| **test_labels*** | - | list containing action labels for each test sequence |
| **jtm_model_front*** | - | trained jtm model loaded using function load_model from jtm module (for front orientation) |
| **jtm_model_top*** | - | trained jtm model loaded using function load_model from jtm module (for top orientation) |
| **jtm_model_side*** | - | trained jtm model loaded using function load_model from jtm module (for side orientation) |
| **model_rp*** | - | trained lstm simple model loaded using function load_model from lstm_simple module (for GeometricFeature RELATIVE_POSITION) |
| **model_jjd*** | - | trained lstm simple model loaded using function load_model from lstm_simple module (for GeometricFeature JOINT_JOINT_DISTANCE) |
| **model_jld*** | - | trained lstm simple model loaded using function load_model from lstm_simple module (for GeometricFeature JOINT_LINE_DISTANCE) |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **image_width*** | - | width of input image (video frame) |
| **image_height*** | - | height of input image (video frame) |
| **input_type** | SPLIT | object defining type of input data (STEP or SPLIT) (must be of type **DatasetInputType**, defined in *har/utils/dataset_util.py* ) - must agree with type of input used during training NN |
| **steps** | 32 | size of step (only for input_type DatasetInputType.STEP) - must agree with type of input used during training NN |
| **split** | 20 | size of split (only for input_type DatasetInputType.SPLIT) - must agree with type of input used during training NN |
| **show_diagram** | True | boolean defining if show diagram after evaluation |
| **results_path** | 'results' | path, where generated results should be placed |
| **use_normalization** | True | boolean defining if normalization should be performed on input data |

#### Example - evaluation
```
import har.impl.jtm.evaluate
import har.impl.lstm_simple.evaluate
from har.impl.lstm_cnn.evaluate import evaluate_tests
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset, berkeley_frame_width, \
    berkeley_frame_height, GeometricFeature


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)

    model_path_front = 'results/model_jtm_en_100_bs_64_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_network_ALEXNET_1634605601206_front.pth'
    model_path_top = 'results/model_jtm_en_100_bs_64_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_network_ALEXNET_1634605601206_top.pth'
    model_path_side = 'results/model_jtm_en_100_bs_64_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_network_ALEXNET_1634605601206_side.pth'

    model_path_rp = 'results/model_lstm_simple_en_10000_bs_128_lr_0.0001_op_RMSPROP_geo_RELATIVE_POSITION_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D.pth'
    model_path_jjd = 'results/model_lstm_simple_en_10000_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_JOINT_DISTANCE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D.pth'
    model_path_jld = 'results/model_lstm_simple_en_10000_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_LINE_DISTANCE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D.pth'

    model_front = har.impl.jtm.evaluate.load_model(model_path_front, len(berkeley_mhad_classes))
    model_top = har.impl.jtm.evaluate.load_model(model_path_top, len(berkeley_mhad_classes))
    model_side = har.impl.jtm.evaluate.load_model(model_path_side, len(berkeley_mhad_classes))

    model_rp = har.impl.lstm_simple.evaluate.load_model(model_path_rp, len(berkeley_mhad_classes),
                                                        geometric_feature=GeometricFeature.RELATIVE_POSITION)
    model_jjd = har.impl.lstm_simple.evaluate.load_model(model_path_jjd, len(berkeley_mhad_classes),
                                                         geometric_feature=GeometricFeature.JOINT_JOINT_DISTANCE)
    model_jld = har.impl.lstm_simple.evaluate.load_model(model_path_jld, len(berkeley_mhad_classes),
                                                         geometric_feature=GeometricFeature.JOINT_LINE_DISTANCE)

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, model_front, model_top, model_side, model_rp, model_jjd,
                              model_jld, video_pose_3d_kpts, berkeley_frame_width, berkeley_frame_height)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
```

### 2. Additional info

#### Example analysed_kpts_description object
```
{
    'right_wrist': 16, 
    'left_wrist': 13, 
    'right_elbow': 15, 
    'left_elbow': 12, 
    'right_shoulder': 14, 
    'left_shoulder': 11, 
    'right_hip': 1, 
    'left_hip': 4, 
    'right_knee': 2, 
    'left_knee': 5, 
    'right_ankle': 3, 
    'left_ankle': 6
}
```

### 3. Article

```
@inproceedings{li2017skeleton,
  title={Skeleton-based action recognition using LSTM and CNN},
  author={Li, Chuankun and Wang, Pichao and Wang, Shuang and Hou, Yonghong and Li, Wanqing},
  booktitle={2017 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={585--590},
  year={2017},
  organization={IEEE}
}
```