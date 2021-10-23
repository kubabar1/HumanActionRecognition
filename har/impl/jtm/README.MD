## Joint Trajectory Maps (JTM)
This solution uses method described in article "Action recognition based on joint trajectory maps using convolutional neural networks".

1. [Training](#1-training)

2. [Load model](#2-load-model)

3. [Evaluation min](#3-evaluation---min)

4. [Evaluation all](#4-evaluation---all)

5. [Fit min](#5-fit---min)

6. [Fit all](#6-fit-all)

7. [Additional info](#7-additional-info)

8. [Article](#8-article)

### 1. Training

This method returns 3 models - for front, top and side view images.

- [Parameters](#parameters---training)

- [Example](#example---training)

#### Parameters - training
| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **classes*** | - | list of classes (class ID must comply with labels ID's) |
| **training_data*** | - | list of training data sequences e.g. list containing 100 sequences of size (125, 17, 3) - input data coordinates must be in range <-1, 1> |
| **training_labels*** | - | list containing action labels for each training sequence |
| **validation_data*** | - | list of validation data sequences e.g. list containing 100 sequences of size (125, 17, 3) |
| **validation_labels*** | - | list containing action labels for each validation sequence |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **image_width*** | - | width of input image (video frame) |
| **image_height*** | - | height of input image (video frame) |
| **epoch_nb** | 100 | count of training iterations |
| **batch_size** | 64 | size of input data batch |
| **learning_rate** | 0.0001 | learning rate |
| **gamma_step_lr** | 0.1 | scheduler step LR gamma parameter |
| **step_size_lr** | 30 | scheduler size of step LR |
| **momentum** | 0.9 | optimizer momentum parameter |
| **weight_decay** | 0 | optimizer weight decay parameter |
| **print_every** | 50 | print training results rate |
| **val_every** | 5 | validate network rate |
| **optimizer_type** | SGD | type of optimizer (must be of type **Optimizer**, defined in *har/utils/training_utils.py* ) |
| **results_path** | 'results' | path, where generated results should be placed |
| **model_name_suffix** | '' | suffix added to generated model name |
| **action_repetitions** | 100 | Repetition of each action (because of large number possible input data configuration - different angle of rotation in X and Y axis - during process of generation input data angles for x - [0, 15, 30, 45] - and y axis - [-45, -30, -15, 0, 15, 30, 45] - are randomly selected; it is repeated as many times as specified by the "action_repetitions" parameter) |
| **save_loss** | True | boolean defining if generate file containing array of losses |
| **save_diagram** | True | boolean defining if save generated diagram of losses and accuracies |
| **save_model** | True | boolean defining if model should be saved (for evaluation) |
| **save_model_for_inference** | False | boolean defining if model should be saved (for inference - it will enable run training again in future) |
| **print_results** | True | boolean defining if print training and validation results during training |
| **use_cache** | False | boolean defining if cache should be generated; after set to True, directory *dataset_cache* will be created during training and it will store all processed inputs which will be sampled during batch generation; thanks to bath generation before each iteration will took less time (however processing all data before first iteration will took much time); this option is enabled for all geometric_feature options except JOINT_COORDINATE; **!IMPORTANT** - remember to remove cache after some changes in input data (if remove cache option isn't set to true) |
| **remove_cache** | False | boolean defining if cache directory for selected input options should be removed before use (only when use_cache is set to True) |
| **show_diagram** | True | boolean defining if show diagram after training |
| **add_timestamp** | True | boolean defining if add timestamp to generated output files names |
| **use_normalization** | True | boolean defining if normalization should be performed on input data |
| **neural_network_model** | ALEXNET | type of neural network (must be of type **NeuralNetworkModel**, defined in *har/impl/jtm/train.py* ) |


#### Example - training
```
from har.impl.jtm.train import train
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, berkeley_frame_height, \
    berkeley_frame_width


def main():
    training_data, training_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.VALIDATION)

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, video_pose_3d_kpts,
          berkeley_frame_width, berkeley_frame_height)


if __name__ == '__main__':
    main()
```


### 2. Load model

- [Parameters](#parameters---load-model)

- [Example](#example---load-model)

#### Parameters - load model
| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **model_path*** | - | path to model generated during training |
| **classes_count*** | - | count of classes |


#### Example - load model
```
from har.impl.jtm.evaluate import evaluate_tests, ModelType, load_model
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset, berkeley_frame_height, \
    berkeley_frame_width


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = 'results/model_jtm_en_5_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_front.pth'
    jtm_model = load_model(model_path, len(berkeley_mhad_classes))


if __name__ == '__main__':
    main()
```


### 3. Evaluation - min
Evaluate using only one JTM neural network model

- [Parameters](#parameters---evaluation)

- [Example](#example---evaluation)

#### Parameters - evaluation
| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **classes*** | - | list of classes (class ID must comply with labels ID's) - order must agree with this used during training NN |
| **test_data*** | - | list of test data sequences e.g. list containing 100 sequences of size (125, 17, 3) |
| **test_labels*** | - | list containing action labels for each test sequence |
| **jtm_model*** | - | trained model loaded using function load_model |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **image_width*** | - | width of input image (video frame) |
| **image_height*** | - | height of input image (video frame) |
| **model_type*** | - | type of model (trained on front, top or side images) - parameter of type "ModelType" - Enum defined in file "har/impl/jtm/evaluate.py" |
| **show_diagram** | True | boolean defining if show diagram after evaluation |
| **results_path** | 'results' | path, where generated results should be placed |
| **use_normalization** | True | boolean defining if normalization should be performed on input data |

#### Example - evaluation
```
from har.impl.jtm.evaluate import evaluate_tests_min, ModelType, load_model
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset, berkeley_frame_height, \
    berkeley_frame_width


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = 'results/model_jtm_en_5_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_front.pth'
    jtm_model = load_model(model_path, len(berkeley_mhad_classes))

    accuracy = evaluate_tests_min(berkeley_mhad_classes, test_data, test_labels, jtm_model, video_pose_3d_kpts, berkeley_frame_width,
                              berkeley_frame_height, ModelType.FRONT)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
```


### 4. Evaluation - all
Evaluate using all 3 JTM models
| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **classes*** | - | list of classes (class ID must comply with labels ID's) - order must agree with this used during training NN |
| **test_data*** | - | list of test data sequences e.g. list containing 100 sequences of size (125, 17, 3) |
| **test_labels*** | - | list containing action labels for each test sequence |
| **jtm_model_front*** | - | trained model loaded using function load_model (for front orientation) |
| **jtm_model_top*** | - | trained model loaded using function load_model (for top orientation) |
| **jtm_model_side*** | - | trained model loaded using function load_model (for side orientation) |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **image_width*** | - | width of input image (video frame) |
| **image_height*** | - | height of input image (video frame) |
| **show_diagram** | True | boolean defining if show diagram after evaluation |
| **results_path** | 'results' | path, where generated results should be placed |
| **use_normalization** | True | boolean defining if normalization should be performed on input data |

#### Example - evaluation
```
from har.impl.jtm.evaluate import evaluate_tests, load_model
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset, berkeley_frame_height, \
    berkeley_frame_width


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path_front = 'results/model_jtm_en_5_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_front.pth'
    model_path_top = 'results/model_jtm_en_5_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_top.pth'
    model_path_side = 'results/model_jtm_en_5_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_side.pth'
    model_path_front = load_model(model_path_front, len(berkeley_mhad_classes))
    model_path_top = load_model(model_path_top, len(berkeley_mhad_classes))
    model_path_side = load_model(model_path_side, len(berkeley_mhad_classes))

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, model_path_front, model_path_top, model_path_side,
                              video_pose_3d_kpts, berkeley_frame_width, berkeley_frame_height)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
```

### 5. Fit - min

Run fit using only one JTM model

- [Parameters](#parameters---fit)

- [Example](#example---fit)

#### Parameters - fit
| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **classes*** | - | list of classes (class ID must comply with labels ID's) - order must agree with this used during training NN |
| **data*** | - | input data as vector of shape (frames_count, kpts_count, kpt_coordinates_count) e.g. (59, 17, 3) |
| **jtm_model*** | - | trained model loaded using function load_model |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **image_width*** | - | width of input image (video frame) |
| **image_height*** | - | height of input image (video frame) |
| **model_type*** | - | type of model (trained on front, top or side images) - parameter of type "ModelType" - Enum defined in file "har/impl/jtm/evaluate.py" |
| **use_normalization** | True | boolean defining if normalization should be performed on input data |

#### Example - fit
```
from random import randrange

from har.impl.jtm.evaluate import fit_min, ModelType, load_model
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, berkeley_frame_width, \
    berkeley_frame_height


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = 'results/model_jtm_en_5_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_front.pth'
    jtm_model = load_model(model_path, len(berkeley_mhad_classes))

    predicted = fit_min(berkeley_mhad_classes, test_sequence, jtm_model, video_pose_3d_kpts, berkeley_frame_width, berkeley_frame_height,
                    ModelType.FRONT)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()
```


### 6. Fit all

Run fit using only all 3 JTM models

- [Parameters](#parameters---fit)

- [Example](#example---fit)

#### Parameters - fit
| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **classes*** | - | list of classes (class ID must comply with labels ID's) - order must agree with this used during training NN |
| **data*** | - | input data as vector of shape (frames_count, kpts_count, kpt_coordinates_count) e.g. (59, 17, 3) |
| **jtm_model_front*** | - | trained model loaded using function load_model (for front orientation) |
| **jtm_model_top*** | - | trained model loaded using function load_model (for top orientation) |
| **jtm_model_side*** | - | trained model loaded using function load_model (for side orientation) || **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **image_width*** | - | width of input image (video frame) |
| **image_height*** | - | height of input image (video frame) |
| **model_type*** | - | type of model (trained on front, top or side images) - parameter of type "ModelType" - Enum defined in file "har/impl/jtm/evaluate.py" |
| **use_normalization** | True | boolean defining if normalization should be performed on input data |

#### Example - fit
```
from random import randrange

from har.impl.jtm.evaluate import fit, load_model
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, berkeley_frame_width, \
    berkeley_frame_height


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]

    model_path_front = 'results/model_jtm_en_5_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_front.pth'
    model_path_top = 'results/model_jtm_en_5_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_top.pth'
    model_path_side = 'results/model_jtm_en_5_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_100_stlr_30_gammastlr_0.1_side.pth'

    model_path_front = load_model(model_path_front, len(berkeley_mhad_classes))
    model_path_top = load_model(model_path_top, len(berkeley_mhad_classes))
    model_path_side = load_model(model_path_side, len(berkeley_mhad_classes))

    predicted = fit(berkeley_mhad_classes, test_sequence, model_path_front, model_path_top, model_path_side, video_pose_3d_kpts,
                    berkeley_frame_width, berkeley_frame_height)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()
```

### 7. Additional info

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

### 8. Article

```
@inproceedings{wang2016action,
  title={Action recognition based on joint trajectory maps using convolutional neural networks},
  author={Wang, Pichao and Li, Zhaoyang and Hou, Yonghong and Li, Wanqing},
  booktitle={Proceedings of the 24th ACM international conference on Multimedia},
  pages={102--106},
  year={2016}
}
```