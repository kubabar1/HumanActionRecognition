# Human Activity Recognition
This repository contains implementation of some popular neural networks architectures allowing to recognise human action basis on coordinates of human skeleton keypoints.
All methods were implemented using PyTorch framework.
To estimate coordinates MMPose library is used.
Most of implemented methods basis on 3D coordinates so to generate 3D coordinates from 2D **VideoPose3D** program was used.

## Installation

To use **pose_2d** module which allow estimate human pose keypoints coordinates in 2D **MMPose** and **MMDetection** needs to be installed. Installation process is described here: https://github.com/open-mmlab/mmpose and here: https://github.com/open-mmlab/mmdetection.
To run modules using **MMPose** **mmcv** and correct version of **PyTorch** also needs to be installed. 
During implementation below versions were used:

- **mmcv-full_1.2.7**

- **mmdet_2.10.0**

- **mmpose_0.12.0**


Below there is example process of installation PyTorch, MMCV, MMDetecition and MMPose which was used during implementation and researches on GoogleColab: 
```
# install dependencies: (use cu101 because colab has CUDA 10.1)
!pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
 
# install mmcv-full thus we could use CUDA operators
!pip install mmcv-full==1.2.7

# install mmdetecion
%cd /content
!rm -rf mmdetection
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!git checkout 31afd0d3b9fad2553dcc8d168f59a42c24e3879d
!pip install -r requirements.txt
!pip install .

# install mmpose
%cd /content
!rm -rf mmpose
!rm -rf mmps
!pip uninstall mmpose
!git clone https://github.com/open-mmlab/mmpose.git
!mv mmpose mmps
!rm -rf mmpose
%cd mmps
!git checkout 75099571f217612503e8caeadba9a696db5aeed5
!pip install -r requirements.txt
!pip install numpy==1.20.0
!pip install .
```

To use **pose_3d** module **VideoPose3D** needs to be configured. In this case only this project needs to be cloned from this repo: https://github.com/facebookresearch/VideoPose3D
```
!git clone https://github.com/facebookresearch/VideoPose3D.git
```

To use remaining modules libraries from requirements needs to be installed:
```
pip install -r requirements.txt
```

At the end **har_skeleton** project needs to be installed. Below command needs to be executed in project root directory containing **setup.py** file:
```
pip install .

# or developement installation
pip install -e .
``` 

## Training

### Dataset generation
Process of dataset generation consists of 2 steps: estimations of coordinates using MMPose and estimation 3D coordinates using VideoPose3D.
This step can be done in 2 ways.
The most recommend solution is to use **prepare_dataset.py** script from **demo** directory - it will estimate 2D keypoints and generate 3D coordinates automatically.
As input it receiving directory containing videos with actions. It is described more precisely [here](demo/README.md)  file.

Other way consists of 2 separate steps - generate 2D coordinates using script from **pose_3d** module - it is described more precisely [here](pose_2d/README.md)  in this module.
Then generated output needs to be manually processed using below commands:
```
for i in $(find -type d -name bbox); do rm -rf $i; done
for i in $(find -type d -name pose_results); do mv $i/* $i/..; done
for i in $(find -type d -name pose_results); do rmdir $i; done
for i in $(find -type d -name "res152_coco_384x288"); do mv $i/csv/* "$i/.."; done
for i in $(find -type d -name csv); do rmdir $i; done
for i in $(find -type d -name res152_coco_384x288); do rmdir $i; done
```
At the end 3D coordinates needs to be generated using **pose_3d** module - it is described more precisely [here](pose_3d/README.md) in this module.

### Run training
Example script fot running training was added in **demo** module - file **run_training.py**.
It is described more precisely [here](demo/README.md).
There were implemented several methods which were described in detail:
- [**Hierarchical RNN**](har/impl/hierarchical_rnn/README.MD)
- [**LSTM**](har/impl/lstm_simple/README.MD)
- [**ST-LSTM**](har/impl/st_lstm/README.MD)
- [**P-LSTM**](har/impl/p_lstm_ntu/README.MD)
- [**JTM**](har/impl/jtm/README.MD)
- [**LSTM CNN**](har/impl/lstm_cnn/README.MD)

## Evaluation

### Load model
Loading generated model during training was used during fit and evaluation in files **run_evaluation.py** and **run_fit.py**.
It is described more precisely [here](demo/README.md).
There were implemented several methods which were described in detail:
- [**Hierarchical RNN**](har/impl/hierarchical_rnn/README.MD)
- [**LSTM**](har/impl/lstm_simple/README.MD)
- [**ST-LSTM**](har/impl/st_lstm/README.MD)
- [**P-LSTM**](har/impl/p_lstm_ntu/README.MD)
- [**JTM**](har/impl/jtm/README.MD)
- [**LSTM CNN**](har/impl/lstm_cnn/README.MD)

### Run evaluation
Example for script running evaluation was added in **demo** module - file **run_evaluation.py**.
It is described more precisely [here](demo/README.md).
There were implemented several methods which were described in detail:
- [**Hierarchical RNN**](har/impl/hierarchical_rnn/README.MD)
- [**LSTM**](har/impl/lstm_simple/README.MD)
- [**ST-LSTM**](har/impl/st_lstm/README.MD)
- [**P-LSTM**](har/impl/p_lstm_ntu/README.MD)
- [**JTM**](har/impl/jtm/README.MD)
- [**LSTM CNN**](har/impl/lstm_cnn/README.MD)


### Run fit
Example script for running fit (prediction for single sequence) was added in **demo** module - file **run_fit.py**.
It is described more precisely [here](demo/README.md).
There were implemented several methods which were described in detail:
- [**Hierarchical RNN**](har/impl/hierarchical_rnn/README.MD)
- [**LSTM**](har/impl/lstm_simple/README.MD)
- [**ST-LSTM**](har/impl/st_lstm/README.MD)
- [**P-LSTM**](har/impl/p_lstm_ntu/README.MD)
- [**JTM**](har/impl/jtm/README.MD)
- [**LSTM CNN**](har/impl/lstm_cnn/README.MD)


### Visualisation
Module containing example scripts for evaluation purposes.
It allows to generate accuracy and loss functions charts using data generated during training and draw 2D/3D coordinates estimated by MMPose or generated by VideoPose3D.
It is described more precisely [here](visualisation/README.md).
