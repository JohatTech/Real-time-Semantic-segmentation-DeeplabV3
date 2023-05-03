## Introduction

I built a real time semantic segmentation app, that take live frame from video camera and segmented the frames using deep learning models from PyTorch. 

The goal of this project is to implement a fast way to give robot vision to any device that it needed.

## Application Overview

### Deep learning model

For this project, I am using a pre-trained deep learning model from PyTorch hub called DeepLabV3

**DeepLab** is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image. 

I implement specifically “***deeplabv3_mobilenet_v3_large***” which is version of deeplabv3 with a backbone of ***“mobilenet”*** architecture, which is the fastest and suitable model for any low-end devices, such as microcomputer.

### Computer vision tools

The next library I used was OpenCV To handle computer vision task like the followings:

- Read camera frames
- pre-processing frames data
- write segmented frames on mp4 video files

### Functionality and results

![image](https://user-images.githubusercontent.com/86735728/235811641-97f3ee67-43f5-45c5-b977-f1e92cc999e1.png)

As you can see in the image above, the application successfully segment real time data from the video camera at different scales, which is really useful for any project that require robot vision.

In this case semantic segmentation, but we can aggregate different kinds of robot vision like, instance segmentation and object detection.

The best third-party model this type of robot vision task in low-end devices is Yolov5, which is the easiest way to execute instance segmentation and object detection quickly

as you can see in the image below, we have object detection, the best part of Yolov5 is that the classes detected on the execution can be stored as a panda data frame for further analysis. 

![image](https://user-images.githubusercontent.com/86735728/235811671-e707a746-4f16-44e4-9b79-0eb591651c7f.png)

## Deployment configurations

In the future, this project will be deployed as a Dockerfile on Docker hub, the goal is to make it easy to deploy the app in any low-end devices, giving access to deep learning capabilities to any robotic projects.
