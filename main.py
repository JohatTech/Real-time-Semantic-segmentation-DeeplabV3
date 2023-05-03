import torch
import numpy as np
import cv2 as cv 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import draw_segmentation_masks
from torchvision import transforms
import torchvision

import subprocess

exit_code = subprocess.call('.\yolov5\detect.py --weights yolov5s.pt --source 0')