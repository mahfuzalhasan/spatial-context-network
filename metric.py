import torch 
# import torchmetrics
# from torchmetrics import Dice
# from torchmetrics.functional import dice
import numpy as np

def dice_score_by_data_torch(target, pred, threshold=0.5, smooth = 1e-6):
    # Calculating PCS threshold
    thr_pcs = np.log(1/threshold - 1 + smooth)
    num = pred.size(0)
    # Probability Correction Strategy --> Shallow Attention Network for Polyp Segmentation
    pred[torch.where(pred > thr_pcs)] /= (pred > thr_pcs).float().mean()
    pred[torch.where(pred < thr_pcs)] /= (pred < thr_pcs).float().mean()
    pred = pred.sigmoid()
    if threshold is not None:
        pred = (pred > threshold).float() 
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum(dim=1)
    dice_score = (2. * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
    return dice_score

def recall(y_true, y_pred, threshold=0.5, smooth=1e-6):
    num_samples = y_true.size(0)
    # Calculating PCS threshold
    thr_pcs = np.log(1/threshold - 1 + smooth)
    num = y_pred.size(0)
    # Probability Correction Strategy --> Shallow Attention Network for Polyp Segmentation
    y_pred[torch.where(y_pred > thr_pcs)] /= (y_pred > thr_pcs).float().mean()
    y_pred[torch.where(y_pred < thr_pcs)] /= (y_pred < thr_pcs).float().mean()
    y_pred = y_pred.sigmoid()
    if threshold is not None:
        y_pred = (y_pred > threshold).float()
    y_true_f = y_true.view(num_samples, -1)
    y_pred_f = y_pred.view(num_samples, -1)

    # true positives, false positives, false negatives
    tp = (y_true_f * y_pred_f).sum(dim=1)
    fn = (y_true_f * (1 - y_pred_f)).sum(dim=1)
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall


def dice_coeff(pred, target, threshold = None, smooth = 1e-6):
    if threshold is not None:
        pred = (pred > threshold).float()  #FocusNet
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum(dim=1)
    dice_score = (2. * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth) 
    return dice_score

## From https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy/script
def jaccard_index(outputs, labels, smooth=1e-6):
    
    intersection = (outputs.long() & labels.long()).float().sum((1, 2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs.long() | labels.long()).float().sum((1, 2, 3))         # Will be zzero if both are 0
    
    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    return iou 


def dice_value(predicted, targets, threshold=0.5, epsilon=1e-6):
    # compute Dice components
    predicted = (predicted > threshold).float()
    intersection = torch.sum(predicted * targets, (1, 2, 3))
    cardinal = torch.sum(predicted + targets, (1, 2, 3))
    return torch.mean((2 * intersection) / (cardinal + epsilon)).item()


if __name__=='__main__':
    B = 8
    C = 1
    H = 384
    W = 384
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    val_1 = dice_value(pred, target, threshold=0.7)
    val_2 = dice_coeff(pred, target, threshold=0.7)
    print('each dim: ',val_1)
    print('flatten: ',torch.mean(val_2))