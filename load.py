import torch
import numpy as np
import cv2 as cv 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import draw_segmentation_masks
from torchvision import transforms
import torchvision
from ultralytics import YOLO


vid = cv.VideoCapture(0)
width= int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height= int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

colors = [(0, 0, 0),        # background
          (128, 0, 0),      # aeroplane
          (0, 128, 0),      # bicycle
          (128, 128, 0),    # bird
          (0, 0, 128),      # boat
          (128, 0, 128),    # bottle
          (0, 128, 128),    # bus
          (128, 128, 128),  # car
          (64, 0, 0),       # cat
          (192, 0, 0),      # chair
          (64, 128, 0),     # cow
          (192, 128, 0),    # dining table
          (64, 0, 128),     # dog
          (192, 0, 128),    # horse
          (64, 128, 128),   # motorbike
          (192, 128, 128),  # person
          (0, 64, 0),       # potted plant
          (128, 64, 0),     # sheep
          (0, 192, 0),      # sofa
          (128, 192, 0),    # train
          (0, 64, 128)]     # tv/monitor



#declaring different model to be used 
deeplabv3 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large')
deeplabv3.eval()


#preprocessing step for the input frames
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_frame(frame):
    input_to_PIL = Image.fromarray(frame)
    input_tensor = preprocess(input_to_PIL)
    return input_tensor

# Define the post-processing steps
def process_semantic(outputs, input_batch):
    normal_mask = torch.nn.functional.softmax(outputs, dim=0)
    num_classes = normal_mask.shape[0]
    masks = normal_mask[0]

    class_dim = 0 
    all_class_mask = masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]
    all_mask = draw_segmentation_masks(input_batch.to(torch.uint8), masks=all_class_mask, alpha=0.6, colors = colors)

    segmentated_frame = np.transpose(all_mask, (1,2,0))
    return segmentated_frame.numpy().astype(np.uint8)

#recording function to store the frames.
def record(frames):
    filename = "result.mp4"
    fcc4 = cv.VideoWriter_fourcc(*'mp4v')
    fps = 20
    frameSize = (width,height)
    writer= cv.VideoWriter(filename, fcc4, fps, frameSize)
    writer.write(frames)

while True:

    #read frame and preprocessing 
    ret, input_frame = vid.read()
    input_batch = process_frame(input_frame)
    tensor_input = input_batch.unsqueeze(0) 

    #take output prediction and processing to display on screen
    with torch.no_grad():
        outputs = deeplabv3(tensor_input)["out"]
        segmentated_frame = process_semantic(outputs, input_batch)
        record(segmentated_frame)
    

    cv.imshow("frame", segmentated_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()


