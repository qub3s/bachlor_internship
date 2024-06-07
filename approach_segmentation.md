1. mean over the 3 images channels sobel filter 
2. threshold with the mean of the image
3. connected components with the digitized masks -> large number of masks (idea: reduce the number of masks)
4. calculate mean color and the center of the mask
5. unify areas under color and distance threshholds
6. recalculate center
7. predict masks with sam based on singular points
8. remove masks with high intersection over union
9. unify the masks(how ???) (maybe, find the best subset ( num_masks <= x) of masks that cover the plane (smallest iou) ? how to implement (best guess minimize the overlap between masks and the ))

9: 
- what do i do with uncovered space
- what do i do with space covered by 2 masks