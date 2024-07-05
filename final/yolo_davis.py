import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import os
import random
import cv2
import time
from transformers import SamConfig, SamModel, SamProcessor, SamVisionConfig, SamPromptEncoderConfig, SamMaskDecoderConfig
import ultralytics
import sys

# load data

data_dir = "data/DAVIS/"

image_path = data_dir + "JPEGImages/480p/"
annotation_path = data_dir + "Annotations/480p/"

image_dirs = os.listdir(image_path)
annotation_dirs = os.listdir(annotation_path)

def get_num_images():
    return len(image_dirs)


def get_random_image():
    i = random.choice(range(0,len(image_dirs)))

    list_of_images = []
    list_of_annotations = []

    for im, an in zip(sorted([f for f in os.listdir(image_path+image_dirs[i]) if os.path.isfile(os.path.join(image_path+image_dirs[i], f))]), sorted([f for f in os.listdir(annotation_path+annotation_dirs[i]) if os.path.isfile(os.path.join(annotation_path+annotation_dirs[i], f))])):
        temp = cv2.cvtColor(cv2.imread(image_path+image_dirs[i] +  "/" + im), cv2.COLOR_RGB2BGR)
        list_of_images.append(temp)
        
        temp = cv2.cvtColor(cv2.imread(annotation_path+annotation_dirs[i] +  "/" + an), cv2.COLOR_RGB2BGR)
        list_of_annotations.append(temp)

    
    return np.array(list_of_images), np.array(list_of_annotations)

def get_image( i ):
    list_of_images = []
    list_of_annotations = []

    for im, an in zip(sorted(os.listdir(image_path+image_dirs[i])), sorted(os.listdir(annotation_path+annotation_dirs[i]))):
        temp = cv2.cvtColor(cv2.imread(image_path+image_dirs[i] +  "/" + im), cv2.COLOR_RGB2BGR)
        list_of_images.append(temp)
        
        temp = cv2.cvtColor(cv2.imread(annotation_path+annotation_dirs[i] +  "/" + an), cv2.COLOR_RGB2BGR)
        list_of_annotations.append(temp)

    
    return image_dirs[i], np.array(list_of_images), np.array(list_of_annotations)

x,y = get_random_image()

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_yolo_mask(raw_image, masks):
    fig, axes = plt.subplots(1, 1, figsize=(15, 15))
    axes.imshow(np.array(raw_image))
    axes.axis("off")

    for mask in masks:
      show_box( [mask[0],mask[1],mask[2],mask[3]], axes)
    plt.show()

from segment_anything import sam_model_registry, SamPredictor

# Define SAM
torch.cuda.set_device(0)

sam_checkpoint = "model/vit_h.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cuda")
predictor = SamPredictor(sam)

# Define Yolo
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True, verbose=False)

def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask_yolo(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()

def show_mask_yolo(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def yolo_eval_batch( model, batch, threshhold ):
    res = []
    
    for x in batch:
        t = model(x).xyxy[0].to("cpu").numpy()
        res.append( t )
    # structure xmin, ymin, xmax, ymax, confidence, type
    return res

def sam_with_yolo_mask( model, predictor, bbox, im):
    if bbox.size != 0:
        input_boxes = torch.from_numpy(np.delete(bbox, [4,5], 1).astype(int)).to(predictor.device)

        predictor.set_image(im)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, im.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    else:
       masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=None,
            multimask_output=False,
        )
       
    return masks.cpu().numpy()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def display_mask(sam_mask):
    for m in sam_mask:
        show_mask(m, plt.gca(), random_color=True)

def unify_masks(masks):
    uni = np.full(masks[0].shape, False)
    
    for x in masks:
        uni = np.logical_or(uni, x)
    
    return uni

def calc_iou_full(masks, an):
    uni = unify_masks(masks)
    an = unify_masks(an)                 # has 3 channels
    
    intersection = np.logical_and(uni, an)
    union = np.logical_or(uni,an)
    
    return np.sum(intersection)/np.sum(union)

def three_dim_int(x):
    return x[0] +  x[1]*256 + x[2]*65536

def calc_iou_per(masks, an):
    an_2 = torch.from_numpy(an).permute(1,2,0).numpy()
    unique_colors = np.unique(an_2.copy().reshape(-1,an_2.shape[2]), axis=0)

    annotations = three_dim_int(an.copy().astype(np.longlong))
    annotations = np.digitize(annotations, bins=np.unique(annotations)) - 1

    anno_for_masks = []

    size = []

    for i in range(1,  annotations.max() + 1 ):
        anno_for_masks.append([])
        c_mask = annotations == i
        size.append(c_mask.sum())
        for mask in masks:
            mask = mask.squeeze()
            intersection = np.logical_and(mask, c_mask)
            union = np.logical_or(mask,c_mask)   
            anno_for_masks[i-1].append(np.sum(intersection)/np.sum(union))          
    
    final_values = []

    for x in anno_for_masks:
        final_values.append(max(x))
    
    size = np.array(size)
    size = size/size.sum()

    summe = 0
    for s, v in zip(size, final_values):
        summe += v * s

    return summe, np.array(final_values).mean()

def masks_indexes(masks):
    final = np.zeros(masks[0].shape)

    for i,x in enumerate(masks):
        temp = np.logical_and(x, final == 0).astype(int)
        temp *= (i+1)
        final += temp
    
    return final

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    ax.axis("off")

iou_list_full = []
iou_list_per = []
iou_list_per_amor = []
print(get_num_images())

for x in range(get_num_images()): 
    print(x)     
    sys.stdout.flush() 
    name , im, an = get_image(x)

    iou_list_full.append([])
    iou_list_per.append([])
    iou_list_per_amor.append([])
    
    masks = yolo_eval_batch( model, im, 0 )

    iou_av = 0
    sam_masks = []
    for i in range(len(masks)):
        an_permute = torch.from_numpy(an[i]).permute(2,0,1).numpy()
        sam_mask = sam_with_yolo_mask(model, predictor, masks[i], im[i])
        sam_masks.append(masks_indexes(sam_mask.copy()))

        iou_list_full[x].append(calc_iou_full( sam_mask.copy(), an_permute.copy() ))
        
        amo_per, per = calc_iou_per( sam_mask.copy(), an_permute.copy() )
        
        iou_list_per[x].append(per)
        iou_list_per_amor[x].append(amo_per)
    
    path = "davis_masks/"+name
    np.save(path, np.concatenate(sam_masks).astype(np.ubyte))




print(iou_list_full)
print()
print(iou_list_per)
print()
print(iou_list_per_amor)
