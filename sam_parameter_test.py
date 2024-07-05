import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import pipeline
from PIL import Image
import scipy

model_type = "vit_b"
sam_checkpoint = "model/"+model_type+".pth"

torch.cuda.set_device(0)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cuda")

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# Depth Models
# Load Depth Model
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)

def bw_frame(frame):
    return (frame[...,0] + frame[...,1] + frame[...,2])/3

def three_dim(x):
    return np.stack( ( x.copy()*234, x.copy()*345, x.copy()*567 ), 2).astype(np.uint8)

def three_dim_bw(x):
    return np.stack( ( x.copy()*255, x.copy()*255, x.copy()*255 ), 2).astype(np.uint8)   

def extract_masks(masks, features):
    extracted_masks = []
    for j,x in enumerate(masks):
        extracted_masks.append(x["segmentation"])
        if j == features:
            break
    return extracted_masks

def mask_depth(masks, depth):
    mean_mask_depth = []

    for mask in masks:
        area = (mask == 1).sum()
        mean_mask_depth.append( ( depth * mask ).sum() / area )
    return np.array(mean_mask_depth)

def background_removal(masks, depths, threshhold_depth):

    mean = depths.mean()
    foreground_masks = []
    probabilty = []
    
    for i,x in enumerate(depths):
        prob = scipy.stats.norm(mean, 0.2).pdf(x)
        
        if  prob > threshhold_depth:
            foreground_masks.append(masks[i])
        
        probabilty.append((prob,i))
        
            
    return foreground_masks, probabilty

def unify_masks(masks):
    uni = np.full(masks[0].shape, False)
    
    for x in masks:
        uni = np.logical_or(uni, x)
    
    return uni

def calc_iou(uni, an):
    uni = uni.squeeze()
    an = an.squeeze()

    intersection = np.logical_and(uni, an)
    union = np.logical_or(uni,an)
    return np.sum(intersection)/np.sum(union)

number_of_extracted_masks = 35

import random
iterations = 200
im_per_iter = 2

rand_images = [ x for x in range(iterations) ]
rand_frames = [ random.randint(0,23) for x in range(im_per_iter*iterations) ]

for xss in range(0,10):
    actual_threshold = xss/200
    iou_av = 0
    counter = 1

    for iii, i in enumerate(rand_images):
        print(iii, file=sys.stderr)
        sys.stdout.flush()
        image = np.load('data/MOVIE/videos/'+str(i)+'.npy')
        segmentation = np.load('data/MOVIE/segmentations/'+str(i)+'.npy')

        images_to_eval = (rand_frames[iii*2], rand_frames[iii*2+1])
        

        # predict the depth values
        for f in images_to_eval:
            im = image[f]
            bw_image = bw_frame(im.copy())

            depth_vals = np.array(pipe(Image.fromarray(bw_image.copy()))["depth"])
            depth_vals = depth_vals / depth_vals.max()

            masks = mask_generator.generate(im.copy())

            masks = extract_masks(masks, number_of_extracted_masks)

            mask_depths = mask_depth(masks, depth_vals) 

            foreground, probabilty = background_removal(masks, mask_depths, actual_threshold)

            uni = unify_masks(foreground)
            iou_av += calc_iou(uni, segmentation[f].squeeze())
            print(iou_av/counter, file=sys.stderr)
            counter = counter+1
            sys.stdout.flush()
    
    print(str(actual_threshold) +" : "+str(iou_av/(iterations*im_per_iter)))
    sys.stdout.flush()