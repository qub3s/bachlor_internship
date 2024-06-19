import cv2
from ultralytics import YOLO
import numpy as np
import torch
import matplotlib.pyplot as plt

img= cv2.imread('test.jpg')
print(type(img))
model = YOLO('yolov8m-seg.pt')
results = model.predict(source=img.copy(), save=True, save_txt=False, stream=True)
for result in results:
    # get array results
    masks = result.masks.data.cpu().numpy()

    final = np.zeros((masks.shape[1],masks.shape[2]))
    
    for mask in masks:
        final = np.logical_or(final, mask)
        print(mask.max())
    
    plt.imshow(final)