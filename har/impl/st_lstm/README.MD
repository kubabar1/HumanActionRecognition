## ST-LSTM

This implementation basis on LSTM cell modification proposed in article **"Spatio-temporal lstm with trust gates for 3d human action recognition"**.

Example use of train method:
```
from har.impl.st_lstm.train import train
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, DatasetInputType


def main():
    training_data, training_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.VALIDATION)

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, video_pose_3d_kpts)


if __name__ == '__main__':
    main()
```
Example use of evaluation method:
```
from har.impl.st_lstm.evaluate import evaluate_tests
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = './results' \
                 '/st_lstm_ep_1000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32.pth'

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, model_path, video_pose_3d_kpts)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
```
Example use of fit method:
```
from random import randrange

from har.impl.st_lstm.evaluate import fit
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, DatasetInputType


def main():
    test_data, test_labels = get_berkeley_dataset('datasets/BerkeleyMHAD/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = 'results' \
                 '/st_lstm_ep_1000_b_128_h_128_lr_0.0001_opt_RMSPROP_inp_SPLIT_mm_0.9_wd_0_hl_None_dr_0.5_split_20_steps_32.pth'

    predicted = fit(berkeley_mhad_classes, test_sequence, model_path, video_pose_3d_kpts)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


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