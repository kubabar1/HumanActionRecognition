## Joint Trajectory Maps (JTM)
This solution uses method described in article "Action recognition based on joint trajectory maps using convolutional neural networks".

Example use of train method:
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

Example use of evaluate tests method:
```
from har.impl.jtm.evaluate import evaluate_tests, ModelType
from har.utils.dataset_util import SetType, berkeley_mhad_classes, video_pose_3d_kpts, get_berkeley_dataset, berkeley_frame_height, \
    berkeley_frame_width


def main():
    test_data, test_labels = get_berkeley_dataset('datasets_processed/berkeley/3D', set_type=SetType.TEST)
    model_path = './results' \
                 '/jtm_ep_20_b_32_h_None_lr_0.0001_opt_SGD_inp_None_mm_None_wd_None_hl_None_dr_None_split_None_steps_None_front.pth'

    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, model_path, video_pose_3d_kpts, berkeley_frame_width,
                              berkeley_frame_height, ModelType.FRONT)

    print('Test accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
```

Example use of test method:
```
from random import randrange

from har.impl.jtm.evaluate import fit, ModelType
from har.utils.dataset_util import get_berkeley_dataset, SetType, berkeley_mhad_classes, video_pose_3d_kpts, berkeley_frame_width, \
    berkeley_frame_height


def main():
    test_data, test_labels = get_berkeley_dataset('datasets/BerkeleyMHAD/3D', set_type=SetType.TEST)
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    model_path = './results' \
                 '/jtm_ep_20_b_32_h_None_lr_0.0001_opt_SGD_inp_None_mm_None_wd_None_hl_None_dr_None_split_None_steps_None_front.pth'

    predicted = fit(berkeley_mhad_classes, test_sequence, model_path, video_pose_3d_kpts, berkeley_frame_width, berkeley_frame_height,
                    ModelType.FRONT)

    print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
    print('PREDICTED: {}'.format(predicted))


if __name__ == '__main__':
    main()
```

Article:
```
@inproceedings{wang2016action,
  title={Action recognition based on joint trajectory maps using convolutional neural networks},
  author={Wang, Pichao and Li, Zhaoyang and Hou, Yonghong and Li, Wanqing},
  booktitle={Proceedings of the 24th ACM international conference on Multimedia},
  pages={102--106},
  year={2016}
}
```