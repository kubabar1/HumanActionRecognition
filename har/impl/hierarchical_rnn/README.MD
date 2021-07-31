## Hierarchical RNN

This implementation basis on article **"Hierarchical Recurrent Neural Network for Skeleton Based Action Recognition"**.

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Du_Hierarchical_Recurrent_Neural_2015_CVPR_paper.pdf
```
@inproceedings{du2015hierarchical,
  title={Hierarchical recurrent neural network for skeleton based action recognition},
  author={Du, Yong and Wang, Wei and Wang, Liang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1110--1118},
  year={2015}
}
```


Example use of train method:
```
train(ntu_rgbd_classes, get_batch_ntu_rgbd, dataset_path='./datasets/nturgbd/3D')
```