## LSTM Simple
This solution uses Pytorch implementation of plain LSTM neural network.

1. [Training](#1-training)

2. [Evaluation](#2-load-model)

3. [Evaluation](#3-evaluation)

4. [Fit](#4-fit)

5. [Additional info](#5-additional-info)

6. [Article](#6-article)

### 1. Training

- [Parameters](#parameters---training)

- [Example](#example---training)

#### Parameters - training
| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **classes*** | - | list of classes (class ID must comply with labels ID's) |
| **training_data*** | - | list of training data sequences e.g. list containing 100 sequences of size (125, 17, 3) |
| **training_labels*** | - | list containing action labels for each training sequence |
| **validation_data*** | - | list of validation data sequences e.g. list containing 100 sequences of size (125, 17, 3) |
| **validation_labels*** | - | list containing action labels for each validation sequence |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **epoch_nb** | 10000 | count of training iterations |
| **batch_size** | 128 | size of input data batch |
| **learning_rate** | 0.0001 | learning rate |
| **hidden_size** | 128 | size of LSTM hidden layer |
| **hidden_layers** | 3 | count of hidden layers in LSTM network |
| **dropout** | 0.5 | applied dropout between LSTM layers |
| **momentum** | 0.9 | optimizer momentum parameter |
| **weight_decay** | 0 | optimizer weight decay parameter |
| **print_every** | 50 | print training results rate |
| **val_every** | 5 | validate network rate |
| **input_type** | SPLIT | object defining type of input data (STEP or SPLIT) (must be of type **DatasetInputType**, defined in *har/utils/dataset_util.py* ) |
| **geometric_feature** | JOINT_COORDINATE | type of geometric feature used during training (basis on article *"On Geometric Features for Skeleton-Based Action Recognition using Multilayer LSTM Networks"*) (must be of type **GeometricFeature**, defined in *har/utils/dataset_util.py* )|
| **optimizer_type** | RMSPROP | type of optimizer (must be of type **Optimizer**, defined in *har/utils/training_utils.py* ) |
| **steps** | 32 | size of step (only for input_type DatasetInputType.STEP) |
| **split** | 20 | size of split (only for input_type DatasetInputType.SPLIT) |
| **results_path** | 'results' | path, where generated results should be placed |
| **model_name_suffix** | '' | suffix added to generated model name |
| **save_loss** | True | boolean defining if generate file containing array of losses |
| **save_diagram** | True | boolean defining if save generated diagram of losses and accuracies |
| **save_model** | True | boolean defining if model should be saved (for evaluation) |
| **save_model_for_inference** | False | boolean defining if model should be saved (for inference - it will enable run training again in future) |
| **add_random_rotation_y** | False | is random rotation should be performed on training data during training |
| **use_cache** | False | boolean defining if cache should be generated; after set to True, directory *dataset_cache* will be created during training and it will store all processed inputs which will be sampled during batch generation; thanks to bath generation before each iteration will took less time (however processing all data before first iteration will took much time); this option is enabled for all geometric_feature options except JOINT_COORDINATE; **!IMPORTANT** - remember to remove cache after some changes in input data (if remove cache option isn't set to true) |
| **remove_cache** | False | boolean defining if cache directory for selected input options should be removed before use (only when use_cache is set to True) |
| **is_3d** | True | boolean defining if input coordinates are 3D (for False input coordinates should be 2D) |
| **show_diagram** | True | boolean defining if show diagram after training |
| **print_results** | True | boolean defining if print training and validation results during training |

#### Example - training
```
from har.impl.lstm_simple.train import train
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts


def main():
    training_data, training_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.VALIDATION)

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, video_pose_3d_kpts)


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
| **analysed_kpts_count** | 12 | count of analysed keypoints |
| **hidden_size** | 128 | size of LSTM hidden layer used during training NN |
| **hidden_layers** | 3 | count of hidden layers in LSTM network used during training NN |
| **is_3d** | True | boolean defining if input coordinates are 3D (for False input coordinates should be 2D) - must agree with value set during training|


#### Example - load model
```
from har.impl.lstm_simple.evaluate import load_model
from har.utils.dataset_util import berkeley_mhad_classes, video_pose_3d_kpts


def main():
    model_path = 'results/lstm_simple_ep_10001_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_3_dr_0.5_split_20_steps_32.pth'
    lstm_simple_model = load_model(model_path, len(berkeley_mhad_classes))


if __name__ == '__main__':
    main()
```


### 3. Evaluation

- [Parameters](#parameters---evaluation)

- [Example](#example---evaluation)

#### Parameters - evaluation
| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **classes*** | - | list of classes (class ID must comply with labels ID's) - order must agree with this used during training NN |
| **test_data*** | - | list of test data sequences e.g. list containing 100 sequences of size (125, 17, 3) |
| **test_labels*** | - | list containing action labels for each test sequence |
| **lstm_simple_model*** | - | trained model loaded using function load_model |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **input_type** | SPLIT | object defining type of input data (STEP or SPLIT) (must be of type **DatasetInputType**, defined in *har/utils/dataset_util.py* ) - must agree with type of input used during training NN |
| **steps** | 32 | size of step (only for input_type DatasetInputType.STEP) - must agree with type of input used during training NN |
| **split** | 20 | size of split (only for input_type DatasetInputType.SPLIT) - must agree with type of input used during training NN |
| **show_diagram** | True | boolean defining if show diagram after evaluation |
| **results_path** | 'results' | path, where generated results should be placed |

#### Example - evaluation
```
from har.impl.lstm_simple.evaluate import evaluate_tests, load_model
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = 'results/lstm_simple_ep_10000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_STEP_mm_0.9_wd_0_hl_3_dr_0.5.pth'
    lstm_simple_model = load_model(model_path, len(berkeley_mhad_classes))

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, lstm_simple_model, video_pose_3d_kpts)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
```


### 4. Fit

- [Parameters](#parameters---fit)

- [Example](#example---fit)

#### Parameters - fit
| Parameter name  | Default value  | Description |
| :--------- |:--------------: | :---- |
| **classes*** | - | list of classes (class ID must comply with labels ID's) - order must agree with this used during training NN |
| **test_data*** | - | input data as vector of shape (frames_count, kpts_count, kpt_coordinates_count) e.g. (59, 17, 3) |
| **lstm_simple_model*** | - | trained model loaded using function load_model |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **input_type** | SPLIT | object defining type of input data (STEP or SPLIT) (must be of type **DatasetInputType**, defined in *har/utils/dataset_util.py* ) - must agree with type of input used during training NN |
| **split** | 20 | size of split (only for input_type DatasetInputType.SPLIT) - must agree with type of input used during training NN |

#### Example - fit
```
from random import randrange

from har.impl.lstm_simple.evaluate import fit, load_model
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts


def main():
    test_data, test_labels = get_berkeley_dataset('datasets/BerkeleyMHAD/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = 'results/lstm_simple_ep_10001_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_3_dr_0.5_split_20_steps_32.pth'

    lstm_simple_model = load_model(model_path, len(berkeley_mhad_classes))

    predicted = fit(berkeley_mhad_classes, test_sequence, lstm_simple_model, video_pose_3d_kpts)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()
```



### 5. Additional info

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

### 6. Article

Geometric features config basis on **"On Geometric Features for Skeleton-Based Action Recognition using Multilayer LSTM Networks"**.

```
@INPROCEEDINGS{7926607,
  author={Zhang, Songyang and Liu, Xiaoming and Xiao, Jun},
  booktitle={2017 IEEE Winter Conference on Applications of Computer Vision (WACV)}, 
  title={On Geometric Features for Skeleton-Based Action Recognition Using Multilayer LSTM Networks}, 
  year={2017},
  volume={},
  number={},
  pages={148-157},
  doi={10.1109/WACV.2017.24}
}
```