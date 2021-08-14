## Joint Trajectory Maps (JTM)
This solution uses method described in article "Action recognition based on joint trajectory maps using convolutional neural networks".

Example use of train method:
```
from har.impl.jtm.train import train
from har.utils.dataset_util import get_berkeley_dataset_3d, SetType, berkeley_mhad_classes, berkeley_frame_width, \
    berkeley_frame_height, get_analysed_keypoints

def main():
    training_data, training_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.TRAINING)
    validation_data, validation_labels = get_berkeley_dataset_3d('datasets/BerkeleyMHAD/3D', set_type=SetType.VALIDATION)

    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, berkeley_frame_width,
          berkeley_frame_height)


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