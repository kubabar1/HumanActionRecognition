## Hierarchical RNN

This implementation basis on article **"Hierarchical Recurrent Neural Network for Skeleton Based Action Recognition"**.

Example use of train method:
```
from har.impl.hierarchical_rnn.train import train
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
@inproceedings{du2015hierarchical,
  title={Hierarchical recurrent neural network for skeleton based action recognition},
  author={Du, Yong and Wang, Wei and Wang, Liang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1110--1118},
  year={2015}
}
```