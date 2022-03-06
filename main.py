import torch
import torch.nn as nn
import torchvision
from math import ceil
from easydict import EasyDict as edict
import PIL
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset,DataLoader
from collections import Counter
import time
import os.path



# Anchor generation layer

def _whctrs(anchor):
    """
    Description:
        Return width, height, x center and y center for an anchor (window).
    Input:
        anchor - form - [x_tl,y_tl,x_br,y_br]
    Output:
        w,h,x_ctr,y_ctr - Individual values of tensor class
    """
    w=anchor[2]-anchor[0]+1
    h=anchor[3]-anchor[1]+1
    #Subtract 1 to get exact center
    #For [0,0,15,15],exact center - (7.5,7.5)
    x_ctr=anchor[0]+0.5*(w-1)
    y_ctr=anchor[1]+0.5*(h-1)
    return w,h,x_ctr,y_ctr

def _mkanchors(ws,hs,x_ctr,y_ctr):
    """
    Description:
        Given a tensor of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
    Inputs:
        ws & hs - 1D tensor of shape - anchor_ratios.shape[0] or anchor_scales.shape[0]
        x_ctr & y_ctr - Individual values belonging to tensor class
    Output:
        Shape - anchor_ratios.shape[0] x 4, form - [x_tl,y_tl,x_br,y_br]
    """
    ws=ws.unsqueeze(-1) 
    hs=hs.unsqueeze(-1)
    #ws and hs, shape: -1 x 1
    #Convert to [x_tl,y_tl,x_br,y_br]
    return torch.hstack([x_ctr-0.5*(ws-1),y_ctr-0.5*(hs-1),x_ctr+0.5*(ws-1),y_ctr+0.5*(hs-1)])
    
def _ratio_enum(anchor,ratios):
    """
    Description:
        Enumerate a set of anchors for each height/width ratio wrt an anchor.
    Inputs: 
        anchor - Window, default = torch.tensor([0,0,15,15])
        ratios - 1D tensor of height/width ratios
    Output: 
        anchors - Shape - anchor_ratios.shape[0] x 4, form - [x_tl,y_tl,x_br,y_br]
    """
    #Convert to w,h,x_ctr,y_ctr
    w,h,x_ctr,y_ctr=_whctrs(anchor)
    #size - Area of the receptive field
    size=w*h
    #ratio - Tensor of ratios
    size_ratios=size/ratios
    ws=torch.round(torch.sqrt(size_ratios))
    hs=torch.round(ws*ratios)
    #ws and hs shape - ratios.shape[0]
    anchors=_mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors

def _scale_enum(anchor,scales):
    """
    Description:
        Enumerate a set of anchors for each scale wrt an anchor.
    Inputs:
        anchor - Transformed tensor using ratios, form - [x_tl,y_tl,x_br,y_br]
        scales - 1D tensor , default - torch.tensor([8,16,32])
    Output:
        anchors - Shape - scales.shape[0] x 4
    """
    w,h,x_ctr,y_ctr=_whctrs(anchor)
    ws=w*scales 
    hs=h*scales 
    #ws and hs shape - scales.shape[0]
    anchors=_mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors
    
def generate_anchors(ratios,scales,base_size=16):
    """
    Description:
        Generate anchor (reference) windows by enumerating aspect ratios X 
        scales wrt a reference (0, 0, 15, 15) window.
    Inputs: 
        ratios - A 1D tensor containing different height/width ratios
        scales - A 1D tensor containing different scalar values to scale the anchors
        base_size - A value indicating how big the reference window is, default - 16
    Output:
        anchors - ratios.shape[0]*scales.shape[0] x 4, form - [x_tl,y_tl,x_br,y_br],
                    dtype - torch.float32 (default)
    """
    base_anchor=torch.tensor([1,1,base_size,base_size])-1
    #default : base anchor = tensor([0,0,15,15])
    ratio_anchors=_ratio_enum(base_anchor,ratios)
    #ratio_anchors shape - ratios.shape[0] x 4
    anchors=torch.vstack([_scale_enum(ratio_anchors[i,:],scales) for i in range(ratio_anchors.shape[0])])
    return anchors

def generate_anchors_pre(height,width,feat_stride,anchor_scales,anchor_ratios):
    """
    Description:
        A wrapper function to generate anchors given different scales and ratios.
        Also return the number of anchors
    Inputs:
        height & width - h & w of the feature maps
        feat_stride - Subsampling ratio
        anchor_scales - Shape - -1
        anchor_ratios - Shape - -1
    Output:
        anchors - Shape - height*width*num_anchors ('A' in the code below) x 4, 
                  form - [x_tl,y_tl,x_br,y_br], dtype - torch.float32 (set)
                  Eg: For a feature map of h = 50, w = 50, num_anchors = 9, 50*50*9 x 4 
        length - A value representing number of anchors generated
    """
    anchors=generate_anchors(anchor_ratios,anchor_scales)
    #This will generate anchor_ratios.shape[0]*anchor_scales.shape[0] anchors for a fixed [0,0,15,15] 
    #reference window
    A=anchors.shape[0] #Number of anchors at each center
    shift_x=torch.arange(0,width)*feat_stride #Shape - (width,)
    shift_y=torch.arange(0,height)*feat_stride #Shape - (height,)
    shift_x,shift_y=torch.meshgrid(shift_x,shift_y)
    #Creates shift_x,shift_y of shape - height x width
    shift_x=shift_x.transpose(0,1).contiguous() #Shape - (height x width)
    shift_y=shift_y.transpose(0,1).contiguous() #Shape - (height x width)
    shifts=torch.vstack([shift_x.view(-1),shift_y.view(-1),shift_x.view(-1),shift_y.view(-1)]).transpose(0,1)
    #Shape - (height*width x 4)
    K=shifts.shape[0] #Number of centers
    anchors=anchors.unsqueeze(0)+shifts.unsqueeze(0).transpose(0,1)
    anchors=anchors.reshape(K*A,4).to(torch.float32)
    return anchors,anchors.shape[0]

# Anchor target layer

def bbox_transform(anchors,gt_boxes):
    '''
    Inputs:
        anchors - Shape - n(variable length) x 4, form - [x_tl,y_tl,x_br,y_br]
        gt_boxes - Shape - n x 4, form - [x_tl,y_tl,x_br,y_br]
    Output:
        targets - Shape - n x 4, form - [delta_x_ctr,delta_y_ctr,delta_w,delta_h]
    '''
    a_widths=anchors[:,2]-anchors[:,0]+1.0
    a_heights=anchors[:,3]-anchors[:,1]+1.0
    a_x_ctrs=anchors[:,0]+0.5*a_widths
    a_y_ctrs=anchors[:,1]+0.5*a_heights
    
    gt_widths=gt_boxes[:,2]-gt_boxes[:,0]+1.0
    gt_heights=gt_boxes[:,3]-gt_boxes[:,1]+1.0
    gt_x_ctrs=gt_boxes[:,0]+0.5*gt_widths
    gt_y_ctrs=gt_boxes[:,1]+0.5*gt_heights
    #All of the above are 1D tensors
    
    #above calculations changes the co-ordinates of the anchors placed
    #Eg: [16,16,7.5,7.5] to [16,16,8,8]
    
    targets_dx=(gt_x_ctrs-a_x_ctrs)/a_widths
    targets_dy=(gt_y_ctrs-a_y_ctrs)/a_heights
    targets_dw=torch.log(gt_widths/a_widths)
    targets_dh=torch.log(gt_heights/a_heights)
    #a_widths, a_heights are always positive floating point numbers
    
    targets=torch.hstack([targets_dx.unsqueeze(-1),
                          targets_dy.unsqueeze(-1),
                          targets_dw.unsqueeze(-1),
                          targets_dh.unsqueeze(-1)])
    
    return targets

def _compute_targets(anchors,gt_boxes):
    """Compute bounding-box regression targets for an image."""
    #assert anchors.shape[0]==gt_boxes.shape[0]
    #assert anchors.shape[1]==4
    #assert gt_boxes.shape[1]==5
    return bbox_transform(anchors,gt_boxes[:,:4]).to(torch.float32)

def _unmap(data,count,inds,fill=0):
    '''
    Unmap a subset of item (data) back to the original set of items (of size count)
    data should be torch.float32
    '''
    device=data.device
    if (len(data.shape)==1):
        ret=torch.empty((count,),dtype=torch.float32,device=device)
        ret.fill_(fill)
        ret[inds]=data
    else:
        ret=torch.empty((count,)+data.shape[1:],dtype=torch.float32,device=device)
        ret.fill_(fill)
        ret[inds,:]=data
    return ret

def get_overlaps(anchors,gt_boxes):
    '''
    Assuming anchors and gt_boxes are of the form - [x_tl,y_tl,x_br,y_br]
    Inputs: 
        anchors - Shape - -1 x 4
        gt_boxes - Shape - -1 x 5 where the last col has class number
    Output:
        ious - Shape - anchors.shape[0] x gt_boxes.shape[0]
    '''
    x1=torch.maximum(anchors[:,0].unsqueeze(-1),gt_boxes[:,0].unsqueeze(0))
    y1=torch.maximum(anchors[:,1].unsqueeze(-1),gt_boxes[:,1].unsqueeze(0))
    x2=torch.minimum(anchors[:,2].unsqueeze(-1),gt_boxes[:,2].unsqueeze(0))
    y2=torch.minimum(anchors[:,3].unsqueeze(-1),gt_boxes[:,3].unsqueeze(0))
    
    zero_areas=torch.logical_and(x1!=x2,y1!=y2) #For [0,3,2,5],[2,1,4,4]
    
    intersects=torch.clamp(x2-x1+1,0)*torch.clamp(y2-y1+1,0)
    intersects=intersects*zero_areas
    
    anchors_area=((anchors[:,2]-anchors[:,0]+1)*(anchors[:,3]-anchors[:,1]+1)).unsqueeze(-1)
    gts_area=((gt_boxes[:,2]-gt_boxes[:,0]+1)*(gt_boxes[:,3]-gt_boxes[:,1]+1)).unsqueeze(0)
    areas=anchors_area+gts_area-intersects+1e-8
    
    res=intersects/areas
    ious=torch.where(intersects>0,res,torch.zeros_like(res))
    return ious

def anchor_target_layer(rpn_cls_score,gt_boxes,im_info,all_anchors,num_anchors):
    """
    Description:
        Creates TRAIN.RPN_BATCH_SIZE number of training samples for the RPN
    Inputs:
        rpn_cls_score - Output from RPN, Shape -  N x h x w x num_anchors*2
                        dtype - torch.float32
        gt_boxes - Shape - variable_length x 5, dtype=torch.float32,
                   form - [x_tl,y_tl,x_br,y_br,class_number]
        im_info - A tuple a tuple of form - ((img_height,img_width,...)) of the input image size
        all_anchors - Shape - h*w*num_anchors x 4, dtype=torch.float32, form - [x_tl,y_tl,x_br,y_br]
                        x_tl - top left x co-ordinate, x_br - bottom right x co-ordinate
        num_anchors - A value equal to anchor_ratios.shape[0]*anchor_scales.shape[0]
    Outputs:
        (dtype for all output tensors: torch.float32)
        rpn_labels - Labels for the training samples, fg samples (1), bg_samples (0), rest (-1)
                     Shape - num_anchors*h*w
        rpn_bbox_targets - Contains regression coefficients, Shape - 1 x h x w x num_anchors*4
                           form - [delta_x,delta_y,delta_w,delta_h], don't care samples all deltas are 0
        rpn_bbox_inside_weights - 1 for fg samples, 0 for the rest, Shape - 1 x h x w x num_anchors*4
        rpn_bbox_outside_weights - Shape - 1 x h x w x num_anchors*4
    """
    A=num_anchors
    total_anchors=all_anchors.shape[0]
    K=total_anchors//num_anchors 
    #K=height * width (of feature maps)
    im_info=im_info[0]
    #Now, im_info is a tuple of form - (img_height,img_width)
    height,width=rpn_cls_score.shape[1:3]

    #only keep anchors inside the image
    inds_inside=torch.where(torch.logical_and(torch.logical_and(all_anchors[:,0]>=0,all_anchors[:,1]>=0)
                        ,torch.logical_and(all_anchors[:,2]<im_info[1],all_anchors[:,3]<im_info[0])))[0]
    #For a 300 x 300 image, valid values are [0,299]
    
    #keep only inside anchors
    anchors=all_anchors[inds_inside,:]
    
    #For the anchors which are inside the image, you need to classify them, you have 3 classes
    #create labels, label: 1 - positive, 0 - negative, -1 - don't care
    labels=torch.empty((len(inds_inside),),dtype=torch.float32,device=inds_inside.device)
    #labels - A 1D tensor
    labels.fill_(-1)
    
    overlaps=get_overlaps(anchors.to(torch.float64),gt_boxes.to(torch.float64))
    #overlaps between the anchors and the gt boxes
    #2D tensor of form - (anchors.shape[0],gt_bboxes.shape[0])
    
    argmax_overlaps=overlaps.argmax(axis=1)
    #argmax_overlaps - With what gt_box does each anchor has max overlap with?
    #A 1D tensor of shape anchors.shape[0]
    
    max_overlaps=overlaps[torch.arange(len(inds_inside)),argmax_overlaps]
    #A 1D tensor containing max overlap values for each anchor, shape - anchors.shape[0]
    
    #There will be only few anchors that have >= TRAIN.RPN_POSITIVE_OVERLAP overlap values
    #Inorder to tackle this problem, set anchors that have max overlap value = max overlap for a gt box
    #as fg samples
    gt_argmax_overlaps=overlaps.argmax(axis=0)
    gt_max_overlaps=overlaps[gt_argmax_overlaps,torch.arange(overlaps.shape[1])]
    #gt_argmax_overlaps - For each gt_box, what's the max overlap value with any anchor?
    gt_argmax_overlaps=torch.where(overlaps==gt_max_overlaps)[0]
    #since, a gt_box can have multiple anchors with same max overlap value
    #gt_argmax_overlaps - All anchor indices that are additionally selected as fg samples
    
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        #assign bg labels first so that positive labels can clobber them
        #first set the negatives
        labels[max_overlaps<cfg.TRAIN.RPN_NEGATIVE_OVERLAP]=0

    #fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps]=1
    #These fg samples might not have >= TRAIN.RPN_POSITIVE_OVERLAP overlap value

    #fg label: above or equal threshold IOU
    labels[max_overlaps>=cfg.TRAIN.RPN_POSITIVE_OVERLAP]=1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps<cfg.TRAIN.RPN_NEGATIVE_OVERLAP]=0
    
    #subsample positive labels if we have too many
    num_fg=int(cfg.TRAIN.RPN_FG_FRACTION*cfg.TRAIN.RPN_BATCHSIZE)
    #I need num_fg fg samples, so if len(fg_inds)>num_fg, set some fg samples to don't care
    fg_inds=torch.where(labels==1)[0]
    if (len(fg_inds)>num_fg):
        disable_inds=torch.randperm(len(fg_inds))[:len(fg_inds)-num_fg]
        labels[fg_inds[disable_inds]]=-1
    
    #subsample negative labels if we have too many
    num_bg=cfg.TRAIN.RPN_BATCHSIZE-torch.sum(labels==1)
    bg_inds=torch.where(labels==0)[0]
    if (len(bg_inds)>num_bg):
        disable_inds=torch.randperm(len(bg_inds))[:len(bg_inds)-num_bg]
        labels[bg_inds[disable_inds]]=-1
        
    #After setting some fg samples or some bg samples to don't care, we now have 
    #exacty TRAIN.RPN_BATCH_SIZE RPN training samples
    #Note: The training set can be skewed i.e. num_bgs >> num_fgs sometimes
    
    #compute actual regression co-efficients
    #bbox_targets=torch.zeros_like(anchors,dtype=torch.float32) 
    bbox_targets=_compute_targets(anchors,gt_boxes[argmax_overlaps,:]) #Output shape - len(inds_inside) x 4
    
    bbox_inside_weights=torch.zeros_like(anchors,dtype=torch.float32)
    
    #only the positive ones have regression targets
    bbox_inside_weights[labels==1,:]=torch.tensor(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS,
                                                  device=anchors.device,dtype=torch.float32)
    
    #Everything related to bbox_outside_weights
    bbox_outside_weights=torch.zeros_like(anchors,dtype=torch.float32)
    #uniform weighting (given non-uniform sampling)
    num_examples=torch.sum(labels>=0)
    #positive_weights=torch.ones((1,4),device=anchors.device,dtype=torch.float32)/num_examples
    #negative_weights=torch.ones((1,4),device=anchors.device,dtype=torch.float32)/num_examples
    #bbox_outside_weights[labels==1,:]=positive_weights
    #bbox_outside_weights[labels==0,:]=negative_weights
    bbox_outside_weights[labels==1,:]=torch.ones_like(bbox_outside_weights[labels==1,:])/num_examples
    bbox_outside_weights[labels==0,:]=torch.ones_like(bbox_outside_weights[labels==0,:])/num_examples
    
    #map up to original set of anchors
    #Up till now, we consider inside anchors, now we want to unmap so that it covers all anchors
    #and not only the inside anchors
    labels=_unmap(labels,total_anchors,inds_inside,fill=-1)
    bbox_targets=_unmap(bbox_targets,total_anchors,inds_inside,fill=0)
    bbox_inside_weights=_unmap(bbox_inside_weights,total_anchors,inds_inside,fill=0)
    bbox_outside_weights=_unmap(bbox_outside_weights,total_anchors,inds_inside,fill=0)
    
    #labels
    #Shape - h*w*num_anchors
    #labels=labels.reshape((1,height,width,A)).permute(0,3,1,2) #Shape - 1 x A x h x w
    #labels=labels.reshape((1,1,A*height,width)) #Shape - 1 x 1 x A*h x w
    rpn_labels=labels
    
    #bbox_targets
    #Shape - h*w*num_anchors x 4
    bbox_targets=bbox_targets.reshape((1,height,width,num_anchors*4))
    rpn_bbox_targets=bbox_targets
    
    #bbox_inside_weights
    bbox_inside_weights=bbox_inside_weights.reshape((1,height,width,num_anchors*4))
    rpn_bbox_inside_weights=bbox_inside_weights
    
    #bbox_outside_weights
    bbox_outside_weights=bbox_outside_weights.reshape((1,height,width,num_anchors*4))
    rpn_bbox_outside_weights=bbox_outside_weights
    
    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights

# Proposal layer

def bbox_transform_inv(boxes,deltas):
    '''
    Inputs:
        boxes - all generated anchors - Shape h*w*num_anchors x 4, form - [x_tl,y_tl,x_br,y_br]
        deltas - Shape h*w*num_anchors x 4, form - [delta_ctr_x,delta_ctr_y,delta_w,delta_h]
    Output:
        pred_boxes - Shape h*w*num_anchors x 4, form - [x_tl,y_tl,x_br,y_br]
    '''
    if boxes.shape[0]==0:
        return torch.zeros((0,deltas.shape[1]),dtype=deltas.dtype,device=deltas.device)

    boxes=boxes.to(dtype=deltas.dtype)
    
    #Convert [x_tl,y_tl,x_br,y_br] to ctr_x,ctr_y,widths,heights
    widths=boxes[:,2]-boxes[:,0]+1.0
    heights=boxes[:,3]-boxes[:,1]+1.0
    ctr_x=boxes[:,0]+0.5*widths
    ctr_y=boxes[:,1]+0.5*heights
    
    #Extract deltas
    dx=deltas[:,0::4] 
    #:: To maintain shape i.e. if you extract 1 column from a 2D tensor, output's shape - num_rows x 1 
    dy=deltas[:,1::4]
    dw=deltas[:,2::4]
    dh=deltas[:,3::4]
    #convert to 128 bit floating point numbers to avoid overflows later?
    #dh.astype(Decimal)
    #dw.astype(Decimal)
    
    #Apply the deltas to get transformed boxes
    widths=widths.unsqueeze(-1)
    heights=heights.unsqueeze(-1)
    pred_ctr_x=dx*widths + ctr_x.unsqueeze(-1)
    pred_ctr_y=dy*heights + ctr_y.unsqueeze(-1)
    pred_w=torch.exp(dw)*widths
    pred_h=torch.exp(dh)*heights
    
    #Convert transformed boxes to [x_tl,y_tl,x_br,y_br]
    pred_boxes=torch.zeros_like(deltas)
    #x_tl
    pred_boxes[:,0:1]=pred_ctr_x-0.5*pred_w
    #y_tl
    pred_boxes[:,1:2]=pred_ctr_y-0.5*pred_h
    #x_br
    pred_boxes[:,2:3]=pred_ctr_x+0.5*pred_w
    #y_br
    pred_boxes[:,3:4]=pred_ctr_y+0.5*pred_h
    
    return pred_boxes 

def clip_boxes(boxes,im_shape):
    """
    Description:
        Clip boxes to image boundaries
    Inputs: 
        boxes - transformed boxes, form - [x_tl,y_tl,x_br,y_br]
        im_shape - A tuple containing width and height of the image, form - (h,w)
    Output:
        boxes - Clipped transformed boxes, form - [x_tl,y_tl,x_br,y_br]
    """
    #cboxes=boxes.detach().clone()
    img_height_sub,img_width_sub=im_shape[0]-1,im_shape[1]-1
    
    boxes[:,0:1]=torch.clamp_(boxes[:,0:1],0,img_width_sub)
    boxes[:,1:2]=torch.clamp_(boxes[:,1:2],0,img_height_sub)
    boxes[:,2:3]=torch.clamp_(boxes[:,2:3],0,img_width_sub)
    boxes[:,3:4]=torch.clamp_(boxes[:,3:4],0,img_height_sub)
    return boxes

def proposal_layer(rpn_cls_prob,rpn_bbox_pred,im_info,anchors,num_anchors):
    #proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors)
    '''
    Description:
        A simplified version compared to fast/er RCNN
    Inputs:
        rpn_cls_prob - Shape - 1 x h x w x num_anchors*2 
        rpn_bbox_pred - Shape - 1 x h x w x num_anchors*4
        im_info - A tuple a tuple of form - ((img_height,img_width,...)) of the input image size
        anchors - All generated anchors, h*w*num_anchors x 4, form - [x_tl,y_tl,x_br,y_br]
        num_anchors - Individual value specifying number of anchors used
    Outputs:
        blob - Shape - post_nms_topN x 5, form - [0,x_tl,y_tl,x_br,y_br], dtype=torch.float32
        scores - Shape - post_nms_topN x 1, dtype=dtype of output of RPN
    '''
    pre_nms_topN=cfg.TRAIN.RPN_PRE_NMS_TOP_N
    post_nms_topN=cfg.TRAIN.RPN_POST_NMS_TOP_N
    nms_thresh=cfg.TRAIN.RPN_NMS_THRESH
    
    im_info=im_info[0]
    
    #Get the scores and bounding boxes
    #scores=rpn_cls_prob[:,:,:,num_anchors:]

    rpn_cls_prob=rpn_cls_prob.reshape(-1,2)
    scores=rpn_cls_prob[:,1]
    rpn_bbox_pred=rpn_bbox_pred.reshape((-1, 4)) #h*w*num_anchors x 4
    scores=scores.reshape((-1,1))
    
    proposals=bbox_transform_inv(anchors,rpn_bbox_pred)
    proposals=clip_boxes(proposals,im_info[:2])
    
    #Pick the top region proposals
    order=scores.squeeze(-1).argsort(descending=True) 
    #returns tensor of indices that would sort the input 1D tensor
    #Pick pre_nms_topN boxes
    order=order[:pre_nms_topN]
    #Select pre_nms_topN clipped transformed boxes and corresponding foreground scores
    proposals=proposals[order,:]
    scores=scores[order]
    
    #Non-maximal suppression
    keep=torchvision.ops.nms(proposals,scores.squeeze(-1),nms_thresh)
    #proposals expected to be in (x_tl,y_tl,x_br,y_br) format with 0 <= x_tl < x_br and 0 <= y_tl < y_br
    #A proposal needs to have iou > nms_thresh with a higher scoring proposal to be discarded
    
    #What happens when NMS eliminates lots of boxes resulting in len(keep) < post_nms_topN?

    #Pick the top region proposals after NMS
    keep=keep[:post_nms_topN]
    proposals=proposals[keep,:]
    scores=scores[keep]
    
    #Only support single image as input
    batch_inds=torch.zeros((proposals.shape[0],1),dtype=torch.float32,device=proposals.device)
    blob=torch.hstack([batch_inds,proposals.to(dtype=torch.float32)])

    return blob,scores

def proposal_layer_test(rpn_cls_prob,rpn_bbox_pred,im_info,anchors,num_anchors):
    #proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors)
    '''
    Description:
        A simplified version compared to fast/er RCNN
    Inputs:
        rpn_cls_prob - Shape - 1 x h x w x num_anchors*2 
        rpn_bbox_pred - Shape - 1 x h x w x num_anchors*4
        im_info - A tuple a tuple of form - ((img_height,img_width,...)) of the input image size
        anchors - All generated anchors, h*w*num_anchors x 4, form - [x_tl,y_tl,x_br,y_br]
        num_anchors - Individual value specifying number of anchors used
    Outputs:
        blob - Shape - post_nms_topN x 5, form - [0,x_tl,y_tl,x_br,y_br], dtype=torch.float32
        scores - Shape - post_nms_topN x 1, dtype=dtype of output of RPN
    '''
    pre_nms_topN=cfg.TEST.RPN_PRE_NMS_TOP_N
    post_nms_topN=cfg.TEST.RPN_POST_NMS_TOP_N
    nms_thresh=cfg.TEST.RPN_NMS_THRESH
    
    im_info=im_info[0]
    
    #Get the scores and bounding boxes
    #scores=rpn_cls_prob[:,:,:,num_anchors:]

    rpn_cls_prob=rpn_cls_prob.reshape(-1,2)
    scores=rpn_cls_prob[:,1]
    rpn_bbox_pred=rpn_bbox_pred.reshape((-1, 4)) #h*w*num_anchors x 4
    scores=scores.reshape((-1,1))
    
    proposals=bbox_transform_inv(anchors,rpn_bbox_pred)
    proposals=clip_boxes(proposals,im_info[:2])
    
    #Pick the top region proposals
    order=scores.squeeze(-1).argsort(descending=True) 
    #returns tensor of indices that would sort the input 1D tensor
    #Pick pre_nms_topN boxes
    order=order[:pre_nms_topN]
    #Select pre_nms_topN clipped transformed boxes and corresponding foreground scores
    proposals=proposals[order,:]
    scores=scores[order]
    
    #Non-maximal suppression
    keep=torchvision.ops.nms(proposals,scores.squeeze(-1),nms_thresh)
    #proposals expected to be in (x_tl,y_tl,x_br,y_br) format with 0 <= x_tl < x_br and 0 <= y_tl < y_br
    #A proposal needs to have iou > nms_thresh with a higher scoring proposal to be discarded
    
    #What happens when NMS eliminates lots of boxes resulting in len(keep) < post_nms_topN?

    #Pick the top region proposals after NMS
    keep=keep[:post_nms_topN]
    proposals=proposals[keep,:]
    scores=scores[keep]
    
    #Only support single image as input
    batch_inds=torch.zeros((proposals.shape[0],1),dtype=torch.float32,device=proposals.device)
    blob=torch.hstack([batch_inds,proposals.to(dtype=torch.float32)])

    return blob,scores

def proposal_top_layer(rpn_cls_prob,rpn_bbox_pred,im_info,anchors,num_anchors):
    #(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors)
    """
    Description:
        A layer that just selects the top region proposals without using non-maximal suppression
    Inputs:
        rpn_cls_prob - Shape - 1 x h x w x num_anchors*2 
        rpn_bbox_pred - Shape - 1 x h x w x num_anchors*4
        im_info - A tuple a tuple of form - ((height,width,...)) of the input image size
        anchors - All generated anchors, h*w*num_anchors x 4, form - [x_tl,y_tl,x_br,y_br]
        num_anchors - Individual value specifying number of anchors used
    Outputs:
        blob - Shape - TEST.RPN_TOPN_N x 5, form - [0,x_tl,y_tl,x_br,y_br], dtype=torch.float32
        scores - Shape - TEST.RPN_TOP_N x 1, dtype=dtype of output of RPN
    """
    rpn_top_n=cfg.TEST.RPN_TOP_N
    im_info=im_info[0]
    #scores=rpn_cls_prob[:,:,:,num_anchors:]
    rpn_cls_prob=rpn_cls_prob.reshape(-1,2)
    scores=rpn_cls_prob[:,1]
    rpn_bbox_pred=rpn_bbox_pred.reshape((-1, 4)) #h*w*num_anchors x 4
    scores=scores.reshape((-1, 1)) #h*w*num_anchors x 1
    length=scores.shape[0]
    if (length<rpn_top_n):
        #Random selection, maybe unnecessary and loses good proposals
        #But such case rarely happens
        top_inds=torch.randint(high=length,size=(rpn_top_n,))
    else:
        #Pick the top region proposals
        top_inds=scores.squeeze(-1).argsort(descending=True)[:rpn_top_n]
        
    #Do the selection here
    anchors=anchors[top_inds]
    rpn_bbox_pred=rpn_bbox_pred[top_inds]
    scores=scores[top_inds]
    
    #Convert anchors into proposals via bbox transformations
    proposals=bbox_transform_inv(anchors,rpn_bbox_pred)
    
    #Clip predicted boxes to image
    proposals=clip_boxes(proposals,im_info[:2])
    
    #Output rois blob
    #RPN implementation only supports a single input image, so all
    #batch inds are 0
    batch_inds=torch.zeros((proposals.shape[0],1),dtype=torch.float32,device=proposals.device)
    blob=torch.hstack([batch_inds,proposals.to(torch.float32)])
    return blob,scores

# Proposal target layer

def _compute_targets_pt(anchors,gt_boxes,labels):
    '''
    Description:
        Compute bounding-box regression targets for an image.
    '''
    #assert anchors.shape[0]==gt_boxes.shape[0]
    #assert anchors.shape[1]==4
    #assert gt_boxes.shape[1]==4
    targets=bbox_transform(anchors,gt_boxes)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        #Optionally normalize targets by a precomputed mean and stdev
        targets=((targets-torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS,dtype=targets.dtype,device=targets.device))
                 /torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS,dtype=targets.dtype,device=targets.device))
    return torch.hstack([labels.unsqueeze(-1),targets]).to(dtype=torch.float32)

def _get_bbox_regression_labels(bbox_target_data,num_classes):
    '''
    Description:
        Bounding-box regression targets (bbox_target_data) are stored in a compact form
        N x (class, tx, ty, tw, th)
        This function expands those targets into the 4-of-4*K representation used by the network
        (i.e. only one class has non-zero targets).
    Inputs:
        bbox_target_data - Tensor of form - [class_number,delta_x,delta_y,delta_w,delta_h]
                           Contains selected fg and bg samples and deltas may be normalized
        num_classes - A value representing actual_num_classes + 1 (for background class)
    Outputs:
        bbox_targets - Tensor of shape - N x 4*num_classes, where only one class entries have deltas
        bbox_inside_weights - Tensor of shape - N x 4*num_classes, where only fg samples just 
                              enteries corresponding to correct class have 1s
    '''
    #extract class info
    clss=bbox_target_data[:,0]
    
    #create bbox_targets and bbox_inside_weights of all zeros 
    bbox_targets=torch.zeros((clss.shape[0],4*num_classes),dtype=torch.float32,
                              device=bbox_target_data.device)
    bbox_inside_weights=torch.zeros_like(bbox_targets)
    
    #inds should contain only the inds for which class_number is not 0
    #0 belongs to background
    inds=torch.where(clss>0)[0]
    
    for ind in inds: 
        scls=clss[ind]
        start=int(scls*4)
        end=start+4
        bbox_targets[ind,start:end]=bbox_target_data[ind,1:]
        bbox_inside_weights[ind,start:end]=torch.tensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS,dtype=bbox_inside_weights.dtype,device=bbox_inside_weights.device)
    
    return bbox_targets,bbox_inside_weights

def _sample_rois(all_rois,all_scores,gt_boxes,fg_rois_per_image,rois_per_image,num_classes):
    '''
    Description:
        Generate a random sample of RoIs comprising foreground and background examples.
    Inputs:
        all_rois - Shape - (selected_transformed_boxes.shape[0]+gt_boxes.shape[0] x 5),
                   form - [0,x_tl,y_tl,x_br,y_br]
        all_scores - Shape - selected_transformed_boxes.shape[0]+gt_boxes.shape[0] x 1
                     Contains foreground scores but for gt boxes the scores are set to 0
        gt_boxes - Shape - variable_length x 5, dtype=torch.float32,
                   form - [x_tl,y_tl,x_br,y_br,class_number]
        fg_rois_per_image - Total number of fg_samples to include in the training set
        rois_per_image - Total number of anchors to be selected to create a batch for training
        num_classes - An integer value containing actual_num_classes + 1 (for background)
    Outputs:
        labels - Shape - -1, contains class_number for each roi, for bg samples, class_number=0
        rois - Same shape and form as all_rois, only the selected fg and bg samples are kept
        rois_scores - Same shape and form as all_rois, only the selected fg and bg samples are kept
                      Contains foreground scores assigned by RPN
        bbox_targets - Shape - -1 x num_classes*4, only the class assigned have deltas
        bbox_inside_weights - Same shape as bbox_targets
    '''
    #compute overlaps 2D tensor (all_rois.shape[0] x gt_boxes.shape[0])
    overlaps=get_overlaps(all_rois[:,1:5].to(torch.float64),gt_boxes.to(torch.float64))
    #for each roi, get a gt box that has maximum overlap with a roi
    gt_assignment=overlaps.argmax(axis=1)
    
    #Max overlap value of each roi with a gt box
    #max_overlaps=overlaps.max(axis=1)
    #TypeError: '>=' not supported between instances of 'torch.return_types.max' and 'float'
    max_overlaps=overlaps[torch.arange(all_rois.shape[0]),gt_assignment]
    
    #each roi gets a label
    labels=gt_boxes[gt_assignment,4]
    
    #Select foreground RoIs as those with >= FG_THRES overlap
    fg_inds=torch.where(max_overlaps>=cfg.TRAIN.FG_THRESH)[0]
    
    #Guard against the case when an image has fewer than fg_rois_per_image
    #Not completed
    
    #Select background RoIs as those within [BG_THRES_LO, BG_THRES_HI)
    bg_inds=torch.where(torch.logical_and(max_overlaps<cfg.TRAIN.BG_THRESH_HI, 
                                          max_overlaps>=cfg.TRAIN.BG_THRESH_LO))[0]
    
    #Small modification to the original version where we ensure a fixed number of regions are sampled
    fg_rois_per_image=int(fg_rois_per_image)
    rois_per_image=int(rois_per_image)
    #If at least 1 fg and bg sample present
    if (fg_inds.shape[0]>0 and bg_inds.shape[0]>0):
        #To ensure we have rois_per_image samples. If very few fg samples, bg samples will be repeated 
        #to make the size of the training samples to be rois_per_image
        fg_rois_per_image=min(fg_inds.shape[0],fg_rois_per_image)
        #2 cases, fg_inds are less or fg_rois_per_image is less
        inds=torch.randperm(len(fg_inds))[:fg_rois_per_image]
        fg_inds=fg_inds[inds]
        bg_rois_per_image=rois_per_image-fg_rois_per_image
        to_replace=bg_inds.shape[0]<bg_rois_per_image
        if (to_replace):
            inds=torch.randint(high=len(bg_inds),size=(bg_rois_per_image,))
        else:
            inds=torch.randperm(len(bg_inds))[:bg_rois_per_image]
        bg_inds=bg_inds[inds]
    #If no bg samples and at least 1 fg sample present
    elif (fg_inds.shape[0]>0):
        #Fill the batch with all fg samples by repeating some fg samples if they are < rois_per_image
        to_replace=fg_inds.shape[0]<rois_per_image
        if (to_replace):
            inds=torch.randint(high=len(fg_inds),size=(rois_per_image,))
        else:
            inds=torch.randperm(len(fg_inds))[:rois_per_image]
        fg_inds=fg_inds[inds]
        fg_rois_per_image=rois_per_image
    #If no fg samples and at least 1 bg sample present
    elif (bg_inds.shape[0]>0):
        to_replace=bg_inds.shape[0]<rois_per_image
        if (to_replace):
            inds=torch.randint(high=len(bg_inds),size=(rois_per_image,))
        else:
            inds=torch.randperm(len(bg_inds))[:rois_per_image]
        bg_inds=bg_inds[inds]
        fg_rois_per_image=0
    else:
        print("No fg and bg samples, shouldn't happen.")
        
    #The indices that we're selecting (both fg and bg)
    keep_inds=torch.cat([fg_inds,bg_inds])
    
    #Select sampled values from various arrays:
    labels=labels[keep_inds]
    #Clamp labels for the background RoIs to 0
    labels[fg_rois_per_image:]=0
    rois=all_rois[keep_inds]
    roi_scores=all_scores[keep_inds]
    
    bbox_target_data=_compute_targets_pt(rois[:,1:5],gt_boxes[gt_assignment[keep_inds],:4],labels)
    #For every anchor, use the gt box that has max overlap with that anchor
    
    bbox_targets,bbox_inside_weights=_get_bbox_regression_labels(bbox_target_data,num_classes)
    
    return labels,rois,roi_scores,bbox_targets,bbox_inside_weights

def proposal_target_layer(rpn_rois,rpn_scores,gt_boxes,_num_classes):
    '''
    Description:
        Assign object detection proposals to ground-truth targets. Produces proposal
        classification labels and bounding-box regression targets.
    Inputs: 
        rpn_rois - selected_transformed_boxes x 5, form - [0,x_tl,y_tl,x_br,y_br]
        rpn_scores - selected_transformed_boxes x 1
                     Contains foreground scores assigned by RPN to selected anchors
        gt_boxes - actual target bboxes of shape - variable_length x 5, form - [x_tl,y_tl,x_br,y_br,class_number]
        _num_classes - An integer value containing actual_num_classes + 1 (for background)
    Outputs:
        rois - Shape - rois_per_image x 5, dtype - torch.float32 (default)
               Form - [0,x_tl,y_tl,x_br,y_br]
        rois_scores - Shape - rois_per_image
        labels - Assigned class labels to rois. For bg samples, assigned labels will be 0
                 Shape - rois_per_image x 1
        bbox_targets - Only the class, roi belongs to will have regression co-efficients, 
                       rest will have 0s
                       Shape - rois_per_image x 4*_num_classes, dtype - torch.float32
        bbox_inside_weights - Only the class rois belong to will have 1s, rest will have 0s
                              Shape and dtype - Similar to bbox_targets
        bbox_outside_weights - Exactly similar to bbox_inside_weights
    '''
    #Proposal ROIs (0, x_tl, y_tl, x_br, y_br) coming from proposal layer
    all_rois=rpn_rois
    all_scores=rpn_scores

    #Include ground-truth boxes in the set of candidate rois
    if cfg.TRAIN.USE_GT:
        zeros=torch.zeros((gt_boxes.shape[0],1),dtype=gt_boxes.dtype,device=gt_boxes.device)
        all_rois=torch.vstack([all_rois,torch.hstack([zeros,gt_boxes[:,:-1]])])
        #not sure if it is wise appending, but anyway i am not using it
        all_scores=torch.vstack([all_scores,zeros])
    
    num_images=1
    rois_per_image=cfg.TRAIN.BATCH_SIZE/num_images
    fg_rois_per_image=ceil(cfg.TRAIN.FG_FRACTION*rois_per_image)
    
    #Sample rois with classification labels and bounding box regression targets
    labels,rois,roi_scores,bbox_targets,bbox_inside_weights=_sample_rois(all_rois,all_scores,
                                                            gt_boxes,fg_rois_per_image,
                                                            rois_per_image,_num_classes)
    
    rois=rois.reshape(-1,5)
    roi_scores=roi_scores.reshape(-1)
    labels=labels.reshape(-1,1)
    bbox_targets=bbox_targets.reshape(-1,_num_classes*4)
    bbox_inside_weights=bbox_inside_weights.reshape(-1,_num_classes*4)
    bbox_outside_weights=(bbox_inside_weights>0).to(dtype=bbox_inside_weights.dtype,device=bbox_inside_weights.device)

    return rois,roi_scores,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

# Trainable net

class trainable_net(nn.Module):
    def __init__(self,num_anchors,num_classes):
        super(trainable_net,self).__init__()
        
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.feature_extractor = vgg16.features[:30]
        self.tail = vgg16.classifier[:5]
        
        #Freeze layers before conv3_1
        for param in self.feature_extractor[:10].parameters():
            param.requires_grad_(False)
        
        #New layers
        self.rpn_conv = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.rpn_cls = nn.Conv2d(in_channels=512,out_channels=2*num_anchors,kernel_size=1)
        self.rpn_regress = nn.Conv2d(in_channels=512,out_channels=4*num_anchors,kernel_size=1)
        self.final_cls = nn.Linear(4096,num_classes)
        self.final_regress = nn.Linear(4096,num_classes*4)

        #Weight initialization
        nn.init.normal_(self.rpn_conv.weight,mean=0.0,std=0.01)
        nn.init.normal_(self.rpn_cls.weight,mean=0.0,std=0.01)
        nn.init.normal_(self.rpn_regress.weight,mean=0.0,std=0.01)
        nn.init.normal_(self.final_cls.weight,mean=0.0,std=0.01)
        nn.init.normal_(self.final_regress.weight,mean=0.0,std=0.001)
        
    def extract_features(self,x):
        return self.feature_extractor(x)
    
    def rpn_forward(self,x):
        x=self.rpn_conv(x)
        return self.rpn_cls(x),self.rpn_regress(x)
    
    def get_output(self,x):
        x=self.tail(x)
        return self.final_cls(x),self.final_regress(x)
    
# Mean average precision

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """ 
    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    c=1
    detections = []
    ground_truths = []

    # Go through all predictions and targets,
    # and only add the ones that belong to the
    # current class c
    for detection in pred_boxes:
        if detection[1] == c:
            detections.append(detection)

    for true_box in true_boxes:
        if true_box[1] == c:
            ground_truths.append(true_box)

    # find the amount of bboxes for each training example
    # Counter here finds how many ground truth bboxes we get
    # for each training example, so let's say img 0 has 3,
    # img 1 has 5 then we will obtain a dictionary with:
    # amount_bboxes = {0:3, 1:5}
    amount_bboxes = Counter([gt[0] for gt in ground_truths])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    # sort by box probabilities which is index 2
    detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)
    
    # If none exists for this class then we can safely skip
    #if total_true_bboxes == 0:
    #    continue
    for detection_idx, detection in enumerate(detections):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

        num_gts = len(ground_truth_img)
        best_iou = 0

        for idx, gt in enumerate(ground_truth_img):
            iou = get_overlaps(torch.tensor(detection[3:]).unsqueeze(0), 
                                torch.tensor(gt[3:]).unsqueeze(0)).item()

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # if IOU is lower then the detection is a false positive
        else:
            FP[detection_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    # torch.trapz for numerical integration
    average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

# Full network

class Network:
    def __init__(self, im_info, num_classes):
        self._feat_stride = 16
        self._batch_size = 1
        self._predictions = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._im_info = im_info
        self.CELoss = torch.nn.CrossEntropyLoss(reduction='mean')
        #Added
        self._anchors = generate_anchors_pre(im_info[0][0]//self._feat_stride,
                                            im_info[0][1]//self._feat_stride,
                                            self._feat_stride,cfg.ANCHOR_SCALES,cfg.ANCHOR_RATIOS)[0]
        self._anchors=self._anchors.to(cfg.device)
        #generate_anchors_pre(height,width,feat_stride,anchor_scales,anchor_ratios)
        self._num_anchors = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS)
        self.normalize = torchvision.transforms.Normalize(cfg.TRAIN.IMG_MEAN,cfg.TRAIN.IMG_STD)
        self._num_classes = num_classes
        self.trainable_net = trainable_net(self._num_anchors,num_classes).to(cfg.device)
        
    def _softmax_layer(self,bottom):
        bottom_shape=bottom.shape
        res=nn.functional.softmax(bottom.reshape(-1,2),dim=1)
        return res.reshape(bottom_shape)
    
    def _proposal_top_layer(self,rpn_cls_prob,rpn_bbox_pred):
        #proposal_top_layer(rpn_cls_prob,rpn_bbox_pred,im_info,anchors,num_anchors)
        return proposal_top_layer(rpn_cls_prob,rpn_bbox_pred,self._im_info,
                                  self._anchors, self._num_anchors)
    
    def _proposal_layer(self,rpn_cls_prob,rpn_bbox_pred):
        #proposal_layer(rpn_cls_prob,rpn_bbox_pred,im_info,anchors,num_anchors)
        return proposal_layer(rpn_cls_prob,rpn_bbox_pred,self._im_info,
                              self._anchors,self._num_anchors)

    def _proposal_layer_test(self,rpn_cls_prob,rpn_bbox_pred):
        #proposal_layer(rpn_cls_prob,rpn_bbox_pred,im_info,anchors,num_anchors)
        return proposal_layer_test(rpn_cls_prob,rpn_bbox_pred,self._im_info,
                                   self._anchors,self._num_anchors)
    
    def _roi_pool_layer(self,bottom,rois):
        return torchvision.ops.roi_pool(bottom,rois,cfg.POOLING_SIZE,1/self._feat_stride)
    
    def _anchor_target_layer(self,rpn_cls_score,gt_boxes):
        #anchor_target_layer(rpn_cls_score,gt_boxes,im_info,all_anchors,num_anchors)
        t=anchor_target_layer(rpn_cls_score,gt_boxes,self._im_info,
                                   self._anchors,self._num_anchors)
        
        self._anchor_targets['rpn_labels'] = t[0]
        self._anchor_targets['rpn_bbox_targets'] = t[1]
        self._anchor_targets['rpn_bbox_inside_weights'] = t[2]
        self._anchor_targets['rpn_bbox_outside_weights'] = t[3]
    
    def _proposal_target_layer(self,rois,roi_scores,gt_boxes):
        #proposal_target_layer(rpn_rois,rpn_scores,gt_boxes,_num_classes)
        t=proposal_target_layer(rois,roi_scores,gt_boxes,self._num_classes)
        #t = (rois,roi_scores,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights)
        rois=t[0]
        self._proposal_targets['rois'] = rois
        self._proposal_targets['labels'] = t[2].to(torch.int32)
        self._proposal_targets['bbox_targets'] = t[3]
        self._proposal_targets['bbox_inside_weights'] = t[4]
        self._proposal_targets['bbox_outside_weights'] = t[5]
        return rois,t[1]
        
    def _smooth_l1_loss(self,bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights,sigma=1.0,dim=[1]):
        '''
        Inputs: 
            bbox_pred - Predicted regression co-efficients, Shape - (1 x h x w x num_anchors*4) or (-1 x 4*num_classes)
                        ,form - [x_ctr,y_ctr,w,h]
            bbox_targets - Actual regression co-efficients, Shape - (1 x h x w x num_anchors*4) or (-1 x 4*num_classes)
                           ,form - [x_ctr,y_ctr,w,h]
            bbox_inside_weights - Will have fg sample rows = [1,1,1,1], rest 0 
            bbox_outside_weights - similar to bbox_inside_weights, 
                                   sometimes it is divided by num_examples
        '''
        sigma_2=sigma**2
        box_diff=bbox_pred-bbox_targets 
        #Only fg_samples regression co-efficients contribute to the loss
        in_box_diff=bbox_inside_weights*box_diff 
        abs_box_diff=torch.abs(in_box_diff)
        #If diff value is less than 1/sigma_2, l2 loss will be used, else, l1 loss  
        smoothL1_sign=(abs_box_diff<(1/sigma_2)).to(torch.float32).requires_grad_(False) #-1 x 4
        in_loss_box=smoothL1_sign*((in_box_diff**2)*(sigma_2/2))+(1-smoothL1_sign)*(abs_box_diff-0.5/sigma_2)
        out_loss_box=bbox_outside_weights*in_loss_box
        loss_box=out_loss_box.sum(axis=dim).mean()
        return loss_box
    
    def cal_loss(self,sigma_rpn=3.0):
        #RPN class loss
        rpn_cls_score = self._predictions['rpn_cls_score'].reshape(-1,2)
        rpn_label = self._anchor_targets['rpn_labels']
        rpn_select = torch.where(rpn_label>-1)[0]
        rpn_cls_score = rpn_cls_score[rpn_select,:]
        rpn_label = rpn_label[rpn_select]
        rpn_cross_entropy = self.CELoss(rpn_cls_score,rpn_label.to(torch.long))
        #RPN bbox loss
        rpn_bbox_pred = self._predictions['rpn_bbox_pred']
        rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                            rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
        #RCNN class loss
        cls_score = self._predictions["cls_score"]
        label = self._proposal_targets["labels"].reshape(-1)
        cross_entropy = self.CELoss(cls_score.reshape(-1,self._num_classes),label.to(torch.long))
        #RCNN, bbox loss
        bbox_pred = self._predictions['bbox_pred']
        bbox_targets = self._proposal_targets['bbox_targets']
        bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
        bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
        loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, 
                                        bbox_inside_weights, bbox_outside_weights)
        #Final loss
        loss = cross_entropy + loss_box + rpn_loss_box + rpn_cross_entropy
        return loss
    
    def forward_train(self,img,gt_boxes):
        '''
        img - Shape: 1 x 3 x img_height x img_width, values: [0,1]
        gt_boxes - Shape: 1 x 5, form: [x_tl,y_tl,x_br,y_br,class_num]
        '''
        #Get img and gt_boxes on the GPU if available
        img,gt_boxes = img.to(cfg.device),gt_boxes.to(cfg.device)

        img=torch.mul(img,255)
        normalized_img = self.normalize(img)
        feature_maps = self.trainable_net.extract_features(normalized_img)
        #feature maps shape: 1 x num_channels x h_fm x w_fm
        rpn_cls_score,rpn_bbox_pred = self.trainable_net.rpn_forward(feature_maps)
        #rpn_cls_score shape: 1 x 2*num_anchors x h_fm x w_fm
        #rpn_bbox_pred shape: 1 x 4*num_anchors x h_fm x w_fm
        rpn_cls_score = rpn_cls_score.permute(0,2,3,1)
        #rpn_cls_score shape: 1 x h_fm x w_fm x 2*num_anchors
        rpn_bbox_pred = rpn_bbox_pred.permute(0,2,3,1)
        #rpn_bbox_pred shape: 1 x h_fm x w_fm x 4*num_anchors
        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._anchor_target_layer(rpn_cls_score,gt_boxes)
        rpn_cls_prob = self._softmax_layer(rpn_cls_score)
        blob,scores = self._proposal_layer(rpn_cls_prob,rpn_bbox_pred)
        rois,rois_scores = self._proposal_target_layer(blob,scores,gt_boxes)
        frois = self._roi_pool_layer(feature_maps,rois)
        frois = frois.reshape(frois.shape[0],-1)
        #frois shape: 1 x num_channels * cfg.POOLING_SIZE * cfg.POOLING_SIZE
        self._predictions["cls_score"],self._predictions["bbox_pred"] = self.trainable_net.get_output(frois)
        return self.cal_loss()

    def forward_test(self,img):
        img=img.to(cfg.device)
        img=torch.mul(img,255)
        normalized_img = self.normalize(img)
        self.trainable_net.eval()

        with torch.no_grad():
            feature_maps = self.trainable_net.extract_features(normalized_img)
            #feature maps shape: 1 x num_channels x h_fm x w_fm
            rpn_cls_score,rpn_bbox_pred = self.trainable_net.rpn_forward(feature_maps)
            #rpn_cls_score shape: 1 x 2*num_anchors x h_fm x w_fm
            #rpn_bbox_pred shape: 1 x 4*num_anchors x h_fm x w_fm
            rpn_cls_score = rpn_cls_score.permute(0,2,3,1)
            #rpn_cls_score shape: 1 x h_fm x w_fm x 2*num_anchors
            rpn_bbox_pred = rpn_bbox_pred.permute(0,2,3,1)
            #rpn_bbox_pred shape: 1 x h_fm x w_fm x 4*num_anchors
            rpn_cls_prob = self._softmax_layer(rpn_cls_score)
            rois = self._proposal_layer_test(rpn_cls_prob,rpn_bbox_pred)[0]
            #[0,x_tl,y_tl,x_br,y_br]
            frois = self._roi_pool_layer(feature_maps,rois)
            frois = frois.reshape(frois.shape[0],-1)
            #frois shape: selected x num_channels x cfg.POOLING_SIZE x cfg.POOLING_SIZE
            cls_scores,bbox_preds = self.trainable_net.get_output(frois)
            #cls_scores: selected x num_classes
            #bbox_preds: selected x num_classes*4

            #Convert unnormalized log probabilities to probabilities
            cls_probs = self._softmax_layer(cls_scores)
            #Get the classes assigned by the network to each roi
            confident_classes = cls_probs.argmax(axis=1)
            #Eliminate background boxes
            keep = torch.where(confident_classes>0)[0]
            if (len(keep)):
                confident_classes,bbox_preds = confident_classes[keep],bbox_preds[keep]
                cls_probs = cls_probs[keep,confident_classes]
            else:
                return []

            deltas = torch.zeros((len(keep),4),dtype=bbox_preds.dtype,device=bbox_preds.device)
          
            for idx,clss in enumerate(confident_classes):
                start = clss*4 
                deltas[idx,:] = bbox_preds[idx,start:start+4]

            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                deltas = (deltas*torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS,dtype=deltas.dtype,device=deltas.device) +
                                 torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS,dtype=deltas.dtype,device=deltas.device))

            final_boxes = bbox_transform_inv(rois[keep,1:],deltas)
            #final_boxes: len(keep) x 4, [x_tl,y_tl,x_br,y_br]
            
            #To reduce redundant predictions
            keep=torchvision.ops.nms(final_boxes,cls_probs.to(final_boxes.dtype),iou_threshold=cfg.TEST.NMS)
            confident_classes=confident_classes[keep]
            cls_probs=cls_probs[keep]
            final_boxes=final_boxes[keep]
            
            self.trainable_net.train()
            
            return torch.hstack([confident_classes.to(final_boxes.dtype).unsqueeze(-1), 
                                 cls_probs.to(final_boxes.dtype).unsqueeze(-1),
                                 final_boxes]).tolist() # selected x 6, form: [class_num, confidence, x_tl, y_tl, x_br, y_br]

    def cal_mAP(self,loader,iou_threshold=0.5,limit=float('inf')):
        #mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20)
        
        true_boxes,pred_boxes,count = [],[],0
      
        for test_idx,(img,gt_boxes) in enumerate(loader):
            img=img.unsqueeze(0)
            gt_boxes=[gt_boxes]
            #Add correct lists to true_boxes, correct format : [img_id, class_number, probability, x_tl, y_tl, x_br, y_br]
            for gt_box in gt_boxes:
                true_boxes.append([test_idx,gt_box[4],1,gt_box[0],gt_box[1],gt_box[2],gt_box[3]])

            #Add correct lists to pred_boxes, correct format : [img_id, class_number, probability, x_tl, y_tl, x_br, y_br]
            predictions = self.forward_test(img)
            for i in range(len(predictions)):
                predictions[i] = [test_idx] + predictions[i]
                pred_boxes.append(predictions[i])
            
            count+=1
            if (count==limit):
                break

        return mean_average_precision(pred_boxes,true_boxes,iou_threshold=iou_threshold).item()

cfg = edict()

#
# Training options
#
cfg.TRAIN = edict()

# Initial learning rate
cfg.TRAIN.LEARNING_RATE = 0.00001

# Momentum
cfg.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
cfg.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
cfg.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
#cfg.TRAIN.STEPSIZE = 30000

# Whether to double the learning rate for bias
#cfg.TRAIN.DOUBLE_BIAS = True

# Whether to add ground truth boxes to the pool when sampling regions
cfg.TRAIN.USE_GT = False #Set back to False

# Scale to use during training (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
#cfg.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
#cfg.TRAIN.MAX_SIZE = 1000

# Minibatch size (number of regions of interest [ROIs])
cfg.TRAIN.BATCH_SIZE = 256

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
cfg.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
cfg.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
cfg.TRAIN.BG_THRESH_HI = 0.5
cfg.TRAIN.BG_THRESH_LO = 0.0

# Use horizontally-flipped images during training?
#cfg.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
#cfg.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
#cfg.TRAIN.BBOX_THRESH = 0.5

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
cfg.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
cfg.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
cfg.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
cfg.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# IOU >= thresh: positive example
cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
cfg.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
cfg.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
cfg.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
cfg.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
cfg.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
cfg.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
cfg.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# __C.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

#cfg.TRAIN.FUSE = True
#cfg.TRAIN.USE_NOISE = False
#cfg.TRAIN.USE_NOISE_AUG = False
#cfg.TRAIN.USE_JPG_AUG = False

#
# Testing options
#
cfg.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
#cfg.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
#cfg.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
cfg.TEST.NMS = 0.2

## NMS threshold used on RPN proposals
cfg.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
cfg.TEST.RPN_PRE_NMS_TOP_N = 6000

## Number of top scoring boxes to keep after applying NMS to RPN proposals
cfg.TEST.RPN_POST_NMS_TOP_N = 100

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# __C.TEST.RPN_MIN_SIZE = 16

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
#cfg.TEST.RPN_TOP_N = 5000

#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
#cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# Size of the pooled region after RoI pooling
cfg.POOLING_SIZE = 7

# Anchor scales for RPN
cfg.ANCHOR_SCALES = torch.tensor([8,16,32,64])

# Anchor ratios for RPN
cfg.ANCHOR_RATIOS = torch.tensor([0.5,1,2])

cfg.TRAIN.IMG_MEAN = [122.7717, 115.9465, 102.9801] #[0.485, 0.456, 0.406]
cfg.TRAIN.IMG_STD = [1,1,1] #[0.229, 0.224, 0.225]

cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def display(timg,gt_boxes):
    nimg=timg.permute(1,2,0).numpy()
    fig,ax=plt.subplots()
    ax.imshow(nimg)
    for gt_box in gt_boxes:
        x_tl,y_tl,x_br,y_br,_=gt_box
        rect=Rectangle((x_tl,y_tl), x_br-x_tl, y_br-y_tl,
                       linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

class getData(Dataset):
    def __init__(self,path,img_height=800,img_width=800,train=True):
        file='/train_filter_single.txt' if train else '/test_filter_single.txt'
        self.path=path
        self.transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((img_height,img_width))])
        self.img_height,self.img_width=img_height,img_width
        with open(path+file,'r') as fileref:
            self.lst=fileref.readlines()
            
    def __getitem__(self,idx):
        #Assuming one gt box per idx
        s=self.lst[idx]
        fn,x_tl,y_tl,x_br,y_br,_=s.strip().split(" ") #[Filename, x_tl, y_tl, x_br, y_br, Don't_care]
        x_tl,y_tl,x_br,y_br=float(x_tl),float(y_tl),float(x_br),float(y_br)

        img=PIL.Image.open(self.path+'/'+fn).convert('RGB')

        #Convert [x_tl, y_tl, x_br, y_br] to [x_ctr, y_ctr, width, height]
        #,height=x_br-x_tl,y_br-y_tl
        #gt_box=[x_tl+width/2, y_tl+height/2, width, height, 1]
        
        #Get scales
        cur_width,cur_height=img.size
        height_scale,width_scale=self.img_height/cur_height,self.img_width/cur_width

        #Create a gt_box list
        gt_box=[x_tl*width_scale,y_tl*height_scale,x_br*width_scale,y_br*height_scale,1]

        #Add class label, dataset contains all tampered images
        return self.transforms(img),gt_box
        
    def __len__(self):
        return len(self.lst)

def collate_fn(lst):
    return lst[0]



net=Network(im_info=[(800,800)], num_classes=2) #Number of fms = 512
checkpoint=torch.load("model/trained_20.pt",map_location=cfg.device)
net.trainable_net.load_state_dict(checkpoint["model_state_dict"])
net.trainable_net.eval()

def save_result(timg,gt_boxes,path):
    nimg=timg.permute(1,2,0).numpy()
    fig,ax=plt.subplots()
    ax.imshow(nimg)
    for gt_box in gt_boxes:
        x_tl,y_tl,x_br,y_br,_=gt_box
        rect=Rectangle((x_tl,y_tl), x_br-x_tl, y_br-y_tl,
                       linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig(path)
    plt.close(fig)

def get_and_save_result(img_path,file_path):
    if os.path.exists('static/final_output.png'):
        os.remove('static/final_output.png')

    img=PIL.Image.open(img_path).convert('RGB')

    timg=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((800,800))])(img)
    preds=net.forward_test(timg.unsqueeze(0))

    for i in range(len(preds)):
        preds[i] = [preds[i][2],preds[i][3],preds[i][4],preds[i][5],preds[i][0]]

    save_result(timg,preds,file_path)



