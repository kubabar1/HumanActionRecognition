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
* **epoch_nb** - (default 10000) count of training iterations
* **batch_size** - (default 128) size of input data batch
* **hidden_size** - (default 128) size of LSTM hidden layer
* **learning_rate** - (default 0.00001) learning rate
* **print_every** - (default 50) print results every iterations rate
* **weight_decay** - (default 0) optimizer weight decay parameter
* **momentum** - (default 0.9) optimizer momentum parameter
* **val_every** - (default 5) validate network every iteration rate
* **save_loss** - (default True) boolean defining if generate file containing array of losses
* **save_diagram** - (default True) boolean defining if save generated diagrams of loses and accuracy
* **results_path** - (default 'results') path, where generated results should be placed
* **optimizer_type** - (default Optimizer.RMSPROP) type of optimizer (must be of type **Optimizer**, defined in *har/utils/training_utils.py* )
* **save_model** - (default True) is model should be saved (to evaluation)
* **save_model_for_inference** - (default False) is model should be saved for inference (it will enable run training again in future)
* **input_type** - (default DatasetInputType.STEP) object defining type of input data (STEP or SPLIT); input using enum defined in DatasetInputType from utils.dataset_util