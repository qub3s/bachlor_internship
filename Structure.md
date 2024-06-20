### General Questions
- Research Questions: Automatic Prompting of Segment Anything with limited Data on the MOVIE and DAVIS Dataset ?
- should the data used for the plot be put in the appendix ? - dont show the data
- should i use yolo segmentation on davis dataset ? - maybe later
- how does per object iou work -> mean best overlap (mean over all best match)

Moviee
NO TRAINING

### TODO
- Rerun Davis and check results
- rerun 250 images with best, first and new threshhold

- Word vs Latex - Latex
- Not graded
- max. 20 pages - 15+
- What is the exact research Question ?
- How much Literature Work is required (more talk about what i did or more talk about sam/yolo and the other methods -> just quick introduction or more)
- Write something about dificulties for the general setup at the start ?
- should i provide pictures of all of my claims ("dense optical flow doesn't work due to...., as you can see ....")

### Davis
- predict bb with Yolo
- feed bb to sam

### Movie

1. trieed Yolo -> Yolo doesnt predict many bbs and the quality is extremly poor (likely due to differnt object classs and images)

2. tried best x masks of the segment anything model
    - Offers good predictions of most or all of the objects, but also predicts big background masks
    - Performance is ok 
    - tried to sort the sam masks by area, predicted_iou and stability but all of these produce drastically worse results -> sam seems to have some internal metric of sorting the masks which is much better ( performance drops from 50 to ~25-30)

3. tried sobel filter
    - offers excellent object detection but the output is hard to use with sam

4. tried optical flow
    - optical flow would work only for a subset of the objects in the image, because many of the objects dont move
    - dense optical flow creates quite vague masks at best (doesnt work well, because background is often moving aswell which makes the movement harder)
    - sparse optical flow create semi useful points, but this is more due to the "interesting points" selection by opencv (doesnt recognise many objects)

5. tried Shi Tomasi Feature Tracking
    - offers decent points prediction
    - but not all objects are beeing detected 
    - big pro point is, that the predictions are most of the time on objects

6. tried to create own approach
    1. use connected components on sobel filters -> receive many regions
    2. fuse these regions, based on proximity, depth and color
    3. predict masks of the pointers of the center of these regions
    4. use intersection over union threshhold to remove masks with large similary
    5. select the largest submask and draw bb around it -> with pointers you need to assemble the masks at the end which gets quickly to computationally expensive (was also implementd...)
    6. predict masks with sam
    7. sort the background masks based on the depth difference to the mean mask

7. tried to work with unprompted sam masks
    - (we no longer try to prompt the model, but to sort through the output)
    -  sort out the background masks using depth 
        - assume normal distribution of masks (with modified sigma to make slopes steeper)
        - calculate the percentage and threshold it (learnable parameter)


Runtime:
- My Approach: 70 sec

### Data:
#### Sam Data
```
- 0.0  : 0.49837786135213413
- 0.05 : 0.49852650549924515
- 0.1  : 0.49903211127687813
- 0.15 : 0.5013830668502673
- 0.2  : 0.5026377288796688


Sam for 0-50
- 0.05 : 0.49852650549924515
Sam for 50-100
- 0.05 : 0.49852650549924515
```

#### Own Approach Data (fittet on images 0-50)

```
starting parameters:
    first 50:
        - Mean:                 0.44878014863775784
        - Std:                  0.225229389011353

    check 50-100:
        - Mean:                 0.42566393592986573
        - Std:                  0.2735224148298474
    
    parameters:
        - iou_error             0.90
        - lam_dis               1
        - lam_col               1
        - num_masks_removal     3
        - number_of_masks       30
        - bb_max_size           128*128/3
        - threshhold_depth      0.1

training: 1 frame per video
- all videos all frames

fittet parameters
    first 50:
        - Mean:                 0.538554379457963
        - Std:                  0.22957278341418427
    
    check 50-100:
        - Mean:                 0.5078257597463253
        - Std:                  0.27765901055310466

    parameters:
        - lam_dis:              0.9232850925460072
        - lam_col:              1.2250144924809447
        - number_of_masks:      72.07403218046714
        - bb_max_size:          3411.1650107295036
        - iou_error:            0.835886654594725
        - threshhold_depth:     0.0477593896838067

final parameters:
    parameters:
        - iou_error:            0.9010009189131205
        - lam_dis:              0.9226001013561762
        - lam_col:              1.236892974627475
        - number_of_masks:      77.00859223880954
        - bb_max_size:          3560.2626059055583
        - threshhold_depth:     0.047680908839153956
    
    check 50-100:
        - Mean:                 0.513383546966974
        - Std:                  0.2747382025116963

```

Ablation study ?, Cross validation ? ... tweaking parameters
never say training -> use fitting