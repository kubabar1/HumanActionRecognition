## LSTM Simple
This solution uses Pytorch implementation of plain LSTM neural network.

##### Example:
```
from har.impl.lstm_simple.train import train
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, video_pose_3d_kpts

def main():
    training_data, training_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.VALIDATION)

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, video_pose_3d_kpts)

if __name__ == '__main__':
    main()
```

##### Params:
* **classes** - list of classes (class ID must comply with labels ID's)
* **training_data** - list of training data sequences e.g. list containing 100 sequences of size (125, 17, 3)
* **training_labels** - list containing action labels for each training sequence
* **validation_data** - list of validation data sequences e.g. list containing 100 sequences of size (125, 17, 3)
* **validation_labels** - list containing action labels for each validation sequence 
* **analysed_kpts_description** - object describing keypoints meaning in training and validation data list, e.g:
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
* **input_size** - (default 36) size of input vector data (e.g. 12 coordinates (x, y, z) -> 36)
* **hidden_layers** - (default 3) count of hidden layers in LSTM network
* **dropout** - (default 0.5) applied dropout between LSTM layers
* **epoch_nb** - (default 10000) count of training iterations
* **batch_size** - (default 128) size of input data batch
* **hidden_size** - (default 128) size of LSTM hidden layer
* **learning_rate** - (default 0.0001) learning rate
* **print_every** - (default 50) print results every iterations rate
* **weight_decay** - (default 0) optimizer weight decay parameter
* **momentum** - (default 0.9) optimizer momentum parameter
* **val_every** - (default 5) validate network every iteration rate
* **input_type** - (default DatasetInputType.SPLIT) object defining type of input data (STEP or SPLIT); input using enum defined in DatasetInputType from utils.dataset_util
* **save_loss** - (default True) boolean defining if generate file containing array of losses
* **save_diagram** - (default True) boolean defining if save generated diagrams of loses and accuracy
* **results_path** - (default 'results') path, where generated results should be placed
* **optimizer_type** - (default Optimizer.RMSPROP) type of optimizer (must be of type **Optimizer**, defined in *har/utils/training_utils.py* )
* **save_model** - (default True) is model should be saved (to evaluation)
* **save_model_for_inference** - (default False) is model should be saved for inference (it will enable run training again in future)
* **add_random_rotation_y** - (default False) is random rotation should be performed on training data during training
* **steps** - (32) size of step (only for input_type DatasetInputType.STEP)
* **split** - (20) size of split (only for input_type DatasetInputType.SPLIT)


### Evaluate

Evaluate generated model on test dataset.

Example use of evaluate method:
```
from har.impl.lstm_simple.evaluate import evaluate_tests
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = 'results/lstm_simple_ep_20000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_3_dr_0.5_split_20_steps_32.pth'

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, model_path, video_pose_3d_kpts)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
```

##### Params:
* **classes** - list of classes (class ID must comply with labels ID's)
* **test_data** - list of test data sequences e.g. list containing 100 sequences of size (125, 17, 3)
* **test_labels** - list containing action labels for each test sequence 
* **model_path** - path to generated model
* **analysed_kpts_description** - object describing keypoints meaning in training and validation data list, e.g:
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
* **hidden_layers** - (default 3) count of hidden layers in LSTM network used during training
* **input_size** - (default 36) size of input vector data for every part used during training (part param is described below) (e.g. 4 parts, each containing 3 keypoints, each described by coordinates (x, y, z) -> 4 parts x 9)
* **input_type** - (default DatasetInputType.SPLIT) object defining type of input data used during training (STEP or SPLIT); input using enum defined in DatasetInputType from utils.dataset_util
* **hidden_size** - (default 128) size of LSTM hidden layer used during training
* **split** - (default 20) size of split used during training (only for input_type DatasetInputType.SPLIT)

### Fit

Evaluate generated model on given single sequence.

Example use of evaluate method:
```
from random import randrange

from har.impl.lstm_simple.evaluate import fit
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, DatasetInputType


def main():
    test_data, test_labels = get_berkeley_dataset('datasets/BerkeleyMHAD/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = 'results/lstm_simple_ep_20000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_3_dr_0.5_split_20_steps_32.pth'

    predicted = fit(berkeley_mhad_classes, test_sequence, model_path, video_pose_3d_kpts)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()

```

##### Params:
* **classes** - list of classes (class ID must comply with labels ID's)
* **data** - input sequence of shape (sequence_length, keypoints_count, kpt_coordinates_count), e.g. (190, 17, 3)
* **model_path** - path to generated model
* **analysed_kpts_description** - object describing keypoints meaning in training and validation data list, e.g:
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
* **hidden_layers** - (default 3) count of hidden layers in LSTM network used during training
* **input_size** - (default 9) size of input vector data for every part used during training (part param is described below) (e.g. 4 parts, each containing 3 keypoints, each described by coordinates (x, y, z) -> 4 parts x 9)
* **input_type** - (default DatasetInputType.SPLIT) object defining type of input data used during training (STEP or SPLIT); input using enum defined in DatasetInputType from utils.dataset_util
* **hidden_size** - (default 128) size of LSTM hidden layer used during training
* **split** - (default 20) size of split used during training (only for input_type DatasetInputType.SPLIT)


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