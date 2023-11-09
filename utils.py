import numpy as np
import torch
from functools import partial

# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decoding(regressed_boxes_t, flatten_proposals, device='cpu'):
    x_p = (flatten_proposals[:, 0] +  flatten_proposals[:, 2])/2
    y_p = (flatten_proposals[:, 1] +  flatten_proposals[:, 3])/2
    w_p = flatten_proposals[:, 2] - flatten_proposals[:, 0]
    h_p = flatten_proposals[:, 3] - flatten_proposals[:, 1]
    t_x, t_y, t_w, t_h = regressed_boxes_t[:,0], regressed_boxes_t[:,1], regressed_boxes_t[:,2], regressed_boxes_t[:,3] 
    x = w_p * t_x + x_p
    y = h_p * t_y + y_p 
    w = w_p * torch.exp(t_w)
    h = h_p * torch.exp(t_h)
    bboxes = torch.stack([x-w/2, y-h/2, x+w/2, y+h/2], dim=1)
    return bboxes 

    
def compute_iou(box1, box2):
        (box1_x1, box1_y1, box1_x2, box1_y2) = box1[...,0].view(-1, 1),  box1[...,1].view(-1, 1),  box1[...,2].view(-1, 1), box1[...,3].view(-1, 1)
        (box2_x1, box2_y1, box2_x2, box2_y2) = box2[...,0],  box2[...,1],  box2[...,2], box2[...,3]
        
         
        # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
        xi1 = torch.max(box1_x1, box2_x1)
        yi1 = torch.max(box1_y1, box2_y1)
        xi2 = torch.min(box1_x2, box2_x2)
        yi2 = torch.min(box1_y2, box2_y2)
        inter_width = torch.clamp(xi2-xi1, 0)
        inter_height = torch.clamp(yi2-yi1, 0)
        inter_area = inter_width * inter_height
            
        # Union(A,B) = A + B - Inter(A,B)
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
            
        # compute the IoU
        iou = inter_area / union_area 
        return iou

def compute_iou_np(box1, box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = np.maximum(box1_x1, box2_x1)
    yi1 = np.maximum(box1_y1, box2_y1)
    xi2 = np.minimum(box1_x2, box2_x2)
    yi2 = np.minimum(box1_y2, box2_y2)
    
    inter_width = np.clip(xi2-xi1, 0, None)
    inter_height = np.clip(yi2-yi1, 0, None)
    inter_area = inter_width * inter_height
    
    # Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area 
    
    return iou
    
    
def precision_recall_curve(predictions, targets, target_class):
    iou_threshold = 0.5
    preds_in_class = predictions[predictions[:, 1] == target_class]
    targets_in_class = targets[targets[:, 1]-1 == target_class]
    total_gt_bboxes = len(targets_in_class)
    sorted_preds = preds_in_class[preds_in_class[:, 2].argsort()][::-1]

    not_matched = np.ones(targets_in_class.shape[0], dtype=bool)
  
    TP = np.zeros(sorted_preds.shape[0])
    for idx, pred in enumerate(sorted_preds):
        img_idx = pred[0]
        gt = targets_in_class[(targets_in_class[:, 0] == img_idx) & not_matched] 
        if len(gt) == 0:
            continue 
            
        gt_bboxes = tuple(np.split(gt[:, 2:], 4, axis=1))
        pred_box = (pred[3], pred[4], pred[5], pred[6])
        
        iou_values = compute_iou_np(pred_box, gt_bboxes)
        max_iou_idx = np.argmax(iou_values)
       
        if iou_values[max_iou_idx] > iou_threshold: 
            TP[idx] = 1
            # Identify and remove the ground truth box that matched with the prediction
            filtered_indices = np.where((targets_in_class[:, 0] == img_idx) & not_matched)[0]
            matched_gt_idx = filtered_indices[max_iou_idx] 
            not_matched[matched_gt_idx] = 0
            
    TP_cumsum = np.cumsum(TP)
    FP_cumsum = np.cumsum(1 - TP)
    recalls = TP_cumsum / total_gt_bboxes
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum)  
    precisions = np.insert(precisions, 0, 1)
    recalls = np.insert(recalls, 0, 0) 
    return recalls, precisions



def average_precision(predictions, targets, target_class):
    recalls, precisions = precision_recall_curve(predictions, targets, target_class)
    for i in range(len(precisions)):
        precisions[i] = max(precisions[i:])
    average_precision_val = np.trapz( precisions, recalls)
    return average_precision_val
   

def non_max_suppression(bboxes):
    all_retained_boxes = []
    for class_idx in range(3):
        class_bboxes = bboxes[bboxes[:, 0] == class_idx]
        if len(class_bboxes):
            retained_boxes = suppress_for_class(class_bboxes, 0.4)
            all_retained_boxes.append(retained_boxes)
    if len(all_retained_boxes):
        return torch.vstack(all_retained_boxes)
    else:
        return torch.tensor([])

def suppress_for_class(sorted_bboxes, iou_threshold):
    retained_boxes = []
    while sorted_bboxes.shape[0] > 1:
        current_box = sorted_bboxes[0]
        retained_boxes.append(current_box)
                                                                                                                                                                             
        remaining_boxes = sorted_bboxes[1:]
        ious = compute_iou(current_box[2:6], remaining_boxes[:, 2:6]).squeeze()
        mask = ious < iou_threshold
        sorted_bboxes = remaining_boxes[mask]
    
    if len(sorted_bboxes) == 1:
        retained_boxes.append(sorted_bboxes[0])
    retained_boxes = torch.vstack(retained_boxes)
    return retained_boxes