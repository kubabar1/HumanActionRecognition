## Trust Gates ST-LSTM

This implementation basis on LSTM cell modification proposed in article **"Spatio-temporal lstm with trust gates for 3d human action recognition"**.

Example use of train method:
```
from har.impl.trust_gates_st_lstm.train import train
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes

def main():
    training_data, training_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', train_test_val_ratio=(0.7, 0.2, 0.1),
                                                             set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D',
                                                                 train_test_val_ratio=(0.7, 0.2, 0.1),
                                                                 set_type=SetType.VALIDATION)

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels)

if __name__ == '__main__':
    main()
```

Article:
```
@inproceedings{liu2016spatio,
  title={Spatio-temporal lstm with trust gates for 3d human action recognition},
  author={Liu, Jun and Shahroudy, Amir and Xu, Dong and Wang, Gang},
  booktitle={European conference on computer vision},
  pages={816--833},
  year={2016},
  organization={Springer}
}
```