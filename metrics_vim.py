import math
import os
import argparse

import numpy as np
import cv2

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

def MAD(pred, true, weight):
    mad = np.abs(pred.cpu().numpy() - true.cpu().numpy()).mean() * weight
    return mad

def MSE(pred, true, weight):
    mse = ((pred.cpu().numpy() - true.cpu().numpy()) ** 2).mean() * weight
    return mse

def dtSSD(pred, pred_tm1, true, true_tm1, weight):
    dtssd = ((pred - pred_tm1) - (true - true_tm1)) ** 2
    dtssd = dtssd.mean().cpu().numpy()
    dtssd = np.sqrt(dtssd) * weight
    return dtssd

def maskiou(gt, output):
    assert gt.shape == output.shape
    gt_binary = gt.detach().clone()
    output_binary = output.detach().clone()
    gt_binary[gt_binary>0.5] = 1.0
    gt_binary[gt_binary<=0.5] = 0.0
    output_binary[output_binary>0.5] = 1.0
    output_binary[output_binary<=0.5] = 0.0
    inter = gt_binary * output_binary
    area_gt = gt_binary[gt_binary==1.0].shape[0]
    area_output = output_binary[output_binary==1.0].shape[0]
    area_inter= inter[inter==1.0].shape[0]
    if area_gt + area_output - area_inter <= 0:
        iou = 0
    else:
        iou = area_inter / (area_gt + area_output - area_inter)
    return iou
    
def matching(gt_clip, output_clip):
    cost_mask = torch.cdist(gt_clip.flatten(1), output_clip.flatten(1), p=1)
    indices = linear_sum_assignment(cost_mask.cpu())
    return indices

def tracking(gt_clip, output_clip, indices_clip):
    tracking_scores = []
    for t in range(gt_clip.shape[1]):
        cost_mask_t = torch.cdist(gt_clip[:,t].flatten(1), output_clip[:,t].flatten(1), p=1)
        indices_t = linear_sum_assignment(cost_mask_t.cpu())
        if False in (indices_clip[1] == indices_t[1]):
            tracking_scores.append(0.0)
        else:
            tracking_scores.append(1.0)
    return np.mean(tracking_scores)

def TP_tracking(gt_clip, output_clip, TP_set):
    tracking_scores = []
    for t in range(gt_clip.shape[1]):
        cost_mask_t = torch.cdist(gt_clip[:,t].flatten(1), output_clip[:,t].flatten(1), p=1)
        indices_t = linear_sum_assignment(cost_mask_t.cpu())
        TP_tracking_scores = []
        for row, col in TP_set:
            per_frame_row_index = indices_t[0]
            TP_row_index = np.where(per_frame_row_index==row)[0]
            if len(TP_row_index) == 0:
                TP_tracking_scores.append(0.0)
            else:
                per_frame_col_index = indices_t[1]
                TP_col_index = per_frame_col_index[TP_row_index[0]]
                if TP_col_index != col:
                    TP_tracking_scores.append(0.0)
                else:
                    TP_tracking_scores.append(1.0)
        tracking_scores.append(np.mean(TP_tracking_scores))
    return np.mean(tracking_scores)

def recognition(gt_clip, output_clip, indices, thres=0.5):
    row_ind, col_ind = indices
    
    TP = 0
    FP = max(output_clip.shape[0] - gt_clip.shape[0], 0)
    FN = max(0, gt_clip.shape[0] - output_clip.shape[0])
    TP_set = []
    iou_weights = []
    for row, col in zip(row_ind, col_ind):
        gt_clip_ins = gt_clip[row]
        output_clip_ins = output_clip[col]
        mask_ious = []
        for t in range(gt_clip_ins.shape[0]):
            gt_clip_ins_frame= gt_clip_ins[t]
            output_clip_ins_frame = output_clip_ins[t]
            mask_iou = maskiou(gt_clip_ins_frame, output_clip_ins_frame)
            mask_ious.append(mask_iou)
        mask_iou = np.mean(mask_ious)
        if mask_iou > thres:
            iou_weights.append(mask_iou)
            TP += 1
            TP_set.append((row, col))
        else:
            FP += 1
            FN += 1
    recognition_score = TP / (TP + FP * 0.5 + FN *0.5)
    if len(iou_weights) != 0:
        recognition_score = recognition_score * np.mean(iou_weights)
    return recognition_score, TP_set

def similarity(gt_clip, output_clip, TP_set, weight=20, error='MSE'):
    matting_scores = []
    for row, col in TP_set:
        gt_clip_ins = gt_clip[row]
        output_clip_ins = output_clip[col]
        similarity_metrics = []
        pred_pha_tm1 = None
        true_pha_tm1 = None
        for t in range(gt_clip_ins.shape[0]):
            gt_clip_ins_frame= gt_clip_ins[t]
            output_clip_ins_frame = output_clip_ins[t]
            if error == 'MSE':
                similarity_metric = 1.0 - min(1.0, MSE(output_clip_ins_frame, gt_clip_ins_frame, weight))
            elif error == 'MAD':
                similarity_metric = 1.0 - min(1.0, MAD(output_clip_ins_frame, gt_clip_ins_frame, weight))
            elif error == 'dtSSD':
                if pred_pha_tm1 is None and true_pha_tm1 is None:
                    similarity_metric = 1.0
                else:
                    similarity_metric = 1.0 - min(1.0, dtSSD(output_clip_ins_frame, pred_pha_tm1, gt_clip_ins_frame, true_pha_tm1, weight))
                pred_pha_tm1 = output_clip_ins_frame
                true_pha_tm1 = gt_clip_ins_frame
            similarity_metrics.append(similarity_metric)

        matting_scores.append(np.mean(similarity_metrics))
        
    matting_score = np.mean(matting_scores)
    return matting_score

transform = T.Compose([
    T.ToTensor()
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='outputs/')
    parser.add_argument('--gt-dir', type=str, default='~/data/VIM50')
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--weight', type=float, default=50)
    parser.add_argument('--error', type=str, default='MSE')
    args = parser.parse_args()
    ##### read in clips and save alpha mattes into matrix
    output_path = args.output_dir
    gt_path = args.gt_dir
    recognitions = []
    trackings = []
    mattings = []
    VMQs = []
    for clip in sorted(os.listdir(gt_path)):
        gt_clip_path = os.path.join(gt_path, clip, 'pha')
        img_set = []
        for ins in sorted(os.listdir(gt_clip_path)):
            ins_img_set = []
            for frame in sorted(os.listdir(os.path.join(gt_clip_path, ins))):
                im = Image.open(os.path.join(gt_clip_path, ins, frame))
                w, h = im.size
                ins_img_set.append(transform(im).unsqueeze(0).unsqueeze(0).cuda())                
            img_set.append(torch.cat(ins_img_set, 1))
        gt_img = torch.cat(img_set, 0)
        ### gt_img [num_ins, num_frame, 1, H, W]
        output_clip_path = os.path.join(output_path, clip)
        img_set = []
        for ins in sorted(os.listdir(output_clip_path)):
            ins_img_set = []
            for frame in sorted(os.listdir(os.path.join(output_clip_path, ins))):
                im = Image.open(os.path.join(output_clip_path, ins, frame))
                w, h = im.size
                ins_img_set.append(transform(im).unsqueeze(0).unsqueeze(0).cuda())                
            img_set.append(torch.cat(ins_img_set, 1))
        output_img = torch.cat(img_set, 0)
        ### output_img [num_ins, num_frame, 1, H, W]
        assert gt_img.shape[-2:] == output_img.shape[-2:]

        #### run L1-distance based one-to-one matching and return indices
        indices = matching(gt_img, output_img)

        #print(tracking_score)
        #### TP, FP, FN computed based on mask-iou matrix
        recognition_score, TP_set = recognition(gt_img, output_img, indices, args.iou)
        recognitions.append(recognition_score)

        #### run per-frame matching vs per-clip matching and return tracking score
        if recognition_score == 0.0:
            tracking_score = 0.0
        else:
            tracking_score = TP_tracking(gt_img, output_img, TP_set)
        trackings.append(tracking_score)

        #### similar score computation
        if recognition_score == 0.0:
            matting_score = 0.0
        else:
            matting_score = similarity(gt_img, output_img, TP_set, args.weight, args.error)
        mattings.append(matting_score)

        VMQ = tracking_score * recognition_score * matting_score
        VMQs.append(VMQ)
        print('clip:', clip, 'RS:', recognition_score, 'TS:', tracking_score, 'MS:', matting_score, 'VMQ:', VMQ)
    print('Finals:')
    print('RQ:', np.round(np.mean(recognitions), 4))
    print('TQ:', np.round(np.mean(trackings), 4))
    print('MQ:', np.round(np.mean(mattings), 4))
    print('VMQ:', np.round(np.mean(trackings)*np.mean(recognitions)*np.mean(mattings), 4))
