import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


number_of_videos = 5
extracted_features = 10
model_type = "vit_b"
sam_checkpoint = "model/"+model_type+".pth"

torch.cuda.set_device(5)
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

def unify_masks(masks):
    uni = np.full(masks[0].shape, False)
    
    for x in masks:
        uni = np.logical_or(uni, x)
    
    return uni

def calc_iou(masks, an):
    uni = unify_masks(masks)
    #an = unify_masks(an)                 # has 3 channels

    uni = uni.squeeze()
    an = an.squeeze()

    intersection = np.logical_and(uni, an)
    union = np.logical_or(uni,an)
    
    return np.sum(intersection)/np.sum(union)

print("starting...",model_type)
for extracted_features in range(5,25,3):
    print("Features",extracted_features)
    sys.stdout.flush()
    for i in range(number_of_videos):
        image = np.load('data/MOVIE/videos/'+str(i)+'.npy')
        segmentation = np.load('data/MOVIE/segmentations/'+str(i)+'.npy')
        frameSize = (128,128)

        name = 'videos/'+str(i)+'_sam_at_'+str(extracted_features)+'_features_'+model_type+'.mp4'
        vid = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'mp4v'), 12, frameSize)

        frameSize = (image.shape[1],image.shape[2])
        num_av = 0
        iou_av = 0

        for im, seg in zip(image,segmentation):
            masks = mask_generator.generate(im.copy())
            
            extracted_masks = []
            for ii,x in enumerate(masks):
                extracted_masks.append(x)
                if ii == extracted_features:
                    break

            num_av += len(masks)
            seg_mask = []
            
            for x in extracted_masks:
                mask = x["segmentation"]            
                mask = np.stack((np.zeros(mask.shape).astype(bool), np.zeros(mask.shape).astype(bool), mask.copy()), 2)
                im[mask] = 255

                seg_mask.append(x["segmentation"])
            
            x = calc_iou(seg_mask,seg)
            iou_av += x
        
            vid.write(im)
        vid.release()

        print("Average masks: ",num_av/image.shape[0])
        sys.stdout.flush()
        
    print("\n\n\n")
