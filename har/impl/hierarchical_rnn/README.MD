## Hierarchical RNN

This implementation basis on article **"Hierarchical Recurrent Neural Network for Skeleton Based Action Recognition"**.

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
| **momentum** | 0.9 | optimizer momentum parameter |
| **weight_decay** | 0 | optimizer weight decay parameter |
| **print_every** | 50 | print training results rate |
| **val_every** | 5 | validate network rate |
| **input_type** | SPLIT | object defining type of input data (STEP or SPLIT) (must be of type **DatasetInputType**, defined in *har/utils/dataset_util.py* ) |
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
| **is_3d** | True | boolean defining if input coordinates are 3D (for False input coordinates should be 2D) |
| **show_diagram** | True | boolean defining if show diagram after training |
| **print_results** | True | boolean defining if print training and validation results during training |
| **use_normalization** | True | boolean defining if normalization should be performed on input data |
| **add_timestamp** | True | boolean defining if add timestamp to generated output files names |

#### Example - training
```
from har.impl.hierarchical_rnn.train import train
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
| **hidden_size** | 128 | size of LSTM hidden layer used during training NN |
| **is_3d** | True | boolean defining if input coordinates are 3D (for False input coordinates should be 2D) - must agree with value set during training|

#### Example - load model
```
from har.impl.hierarchical_rnn.evaluate import load_model, evaluate_tests
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = 'results/model_hierarchical_rnn_en_500_bs_128_lr_0.0001_op_RMSPROP_hs_128_it_SPLIT_momentum_0.9_wd_0_split_20_steps_32_3D.pth'
    p_simple_model = load_model(model_path, len(berkeley_mhad_classes))


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
| **hrnn_model*** | - | trained model loaded using function load_model |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **input_type** | SPLIT | object defining type of input data (STEP or SPLIT) (must be of type **DatasetInputType**, defined in *har/utils/dataset_util.py* ) - must agree with type of input used during training NN |
| **steps** | 32 | size of step (only for input_type DatasetInputType.STEP) - must agree with type of input used during training NN |
| **split** | 20 | size of split (only for input_type DatasetInputType.SPLIT) - must agree with type of input used during training NN |
| **show_diagram** | True | boolean defining if show diagram after evaluation |
| **results_path** | 'results' | path, where generated results should be placed |
| **use_normalization** | True | boolean defining if normalization should be performed on input data |
| **use_part_of_sequence** | False | use during training only random part of sequence - size of random sequence is equal to value of 'steps' param (should be used only for SPLIT input type) |
| **add_random_rotation_y** | False | is random rotation should be performed on training data during training |

#### Example - evaluation
```
from har.impl.hierarchical_rnn.evaluate import load_model, evaluate_tests
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = 'results/model_hierarchical_rnn_en_500_bs_128_lr_0.0001_op_RMSPROP_hs_128_it_SPLIT_momentum_0.9_wd_0_split_20_steps_32_3D.pth'
    p_simple_model = load_model(model_path, len(berkeley_mhad_classes))
    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, p_simple_model, video_pose_3d_kpts)
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
| **hrnn_model*** | - | trained model loaded using function load_model |
| **analysed_kpts_description*** | - | object describing keypoints meaning in training and validation data list (example object described [here](#example-analysed_kpts_description-object)) |
| **input_type** | SPLIT | object defining type of input data (STEP or SPLIT) (must be of type **DatasetInputType**, defined in *har/utils/dataset_util.py* ) - must agree with type of input used during training NN |
| **split** | 20 | size of split (only for input_type DatasetInputType.SPLIT) - must agree with type of input used during training NN |
| **steps** | 32 | size of step (only for input_type DatasetInputType.STEP) - must agree with type of input used during training NN |
| **use_normalization** | True | boolean defining if normalization should be performed on input data |

#### Example - fit
```
from random import randrange

from har.impl.hierarchical_rnn.evaluate import fit, load_model
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = 'results/model_hierarchical_rnn_en_500_bs_128_lr_0.0001_op_RMSPROP_hs_128_it_SPLIT_momentum_0.9_wd_0_split_20_steps_32_3D.pth'

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

```
@inproceedings{du2015hierarchical,
  title={Hierarchical recurrent neural network for skeleton based action recognition},
  author={Du, Yong and Wang, Wei and Wang, Liang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1110--1118},
  year={2015}
}
```