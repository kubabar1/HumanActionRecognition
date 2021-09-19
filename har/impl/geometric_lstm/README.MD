## Multilayer LSTM with Geometric Features

This implementation basis on article **"On Geometric Features for Skeleton-Based Action Recognition using Multilayer LSTM Networks"**.

Example use of train method:
```
from har.impl.geometric_multilayer_lstm.train import train
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes

def main():
    training_data, training_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.VALIDATION)

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels)

if __name__ == '__main__':
    main()
```

Article:
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