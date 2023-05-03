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

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cd7528be-9323-4638-9c22-6ec031678274/Untitled.png)

As you can see in the image above, the application successfully segment real time data from the video camera at different scales, which is really useful for any project that require robot vision.

In this case semantic segmentation, but we can aggregate different kinds of robot vision like, instance segmentation and object detection.

The best third-party model this type of robot vision task in low-end devices is Yolov5, which is the easiest way to execute instance segmentation and object detection quickly

as you can see in the image below, we have object detection, the best part of Yolov5 is that the classes detected on the execution can be stored as a panda data frame for further analysis. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c0b0efc8-ab67-4ad0-b38e-6cf573eada27/Untitled.png)

## Deployment configurations

In the future, this project will be deployed as a Dockerfile on Docker hub, the goal is to make it easy to deploy the app in any low-end devices, giving access to deep learning capabilities to any robotic projects.
