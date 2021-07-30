# HumanActionRecognition
This repository contains implementation of some popular neural networks architectures allowing to recognise human action basis on coordinates of human skeleton keypoints.
All methods were implemented using PyTorch framework.

To generate 2D coordinates MMPose is used. Scripts allowing to generate 2D keypoints for human poses from image sequences or videos are described in **hpe** module.

Some methods basis on 3D coordinates. 3D coordinates can be generated from 2D ones using VideoPose3D. Script allowing to generate 3D coordinates from estimated poses by script from **hpe** module is described in **pose_3d** module. 

