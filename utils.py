import os
import random
import numpy as np 
import cv2
import scipy.ndimage as ndimage
import nrrd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from skimage import measure, morphology, segmentation
from skimage.transform import resize
from sklearn.cluster import KMeans
# import seaborn as sns
import torch
import pickle

from augmentation import Augmentation




def multistep_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, lr_decay_epoch, max_epoch, gamma=0.1):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    flag = False
    for i in range(len(lr_decay_epoch)):
        if epoch == lr_decay_epoch[i]:
            flag = True
            break

    if flag == True:
        lr = init_lr * gamma**(i+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else:
        return optimizer.param_groups[0]['lr']

    return lr


def noramlize(img):
    # return (255.0*(img - np.min(img))/(np.max(img) - np.min(img))).astype('uint8')
    return np.uint8(img)

def window_image(img, window_center=40, window_width=400, intercept=0, slope=1, rescale=True):
    # transform to hu
    # 16 bit
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    # [-160, 240] --> 400 ... signed 8 bit --> [-127, 128]
    img_min = window_center - window_width//2 #minimum HU level -160
    img_max = window_center + window_width//2 #maximum HU level  240
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level

    # 16 bit
    # normalization for 16 bit --> [0, 1]
    img_norm = (img - img_min) / (img_max - img_min)
    #print('img_norm type: ',img_norm.dtype)
    
    # 8-bit
    #[0, 255] --> [0, 1]
    img_8_bit = (img_norm*255.0).astype('uint8') 
    #print('img_8_bit type: ',img_8_bit.dtype)

    return img_8_bit, img_norm

def bounding_box_detection(img, mask, lung_mask):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # find the contours
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_cnt = None
    max_x, max_y, max_w, max_h = 0,0,0,0

    for cnt in contours:
        # compute the bounding rectangle of the contour
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
            max_x, max_y, max_w, max_h = x,y,w,h
    img = img[max_y:max_y+max_h, max_x:max_x+max_w]
    mask = mask[max_y:max_y+max_h, max_x:max_x+max_w]
    lung_mask = lung_mask[max_y:max_y+max_h, max_x:max_x+max_w]
    return img, mask, lung_mask

def lungs_filtration(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    mask = np.zeros_like(image, dtype=np.uint16)
    mask[image <= -400] = 1
    # Remove out of boady air from the mask
    _, _, num_axial_slices = mask.shape
    out_of_body_segment = mask[0, 0, 0]
    body_air = []
    for i in range(num_axial_slices):
        msk = mask[:, :, i]
        msk, _ = ndimage.label(msk)
        out_of_body_segment = msk[0, 0] 
        msk[msk == out_of_body_segment] = 0
        msk[msk > 0] = 1
        msk = msk.astype(np.uint8)
        # Remove airway
        msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel)
        msk = segmentation.clear_border(msk)
        body_air.append(msk)
    mask = np.dstack(body_air)
    mask = mask.astype(np.uint8)
    return mask


def generate_markers(image):
    '''
    Courtesy: https://www.kaggle.com/code/ankasor/improved-lung-segmentation-using-watershed
    '''
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = (marker_internal_labels > 0).astype('uint8')*255
    
    return marker_internal

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_confusion_matrix(predictions, actual_labels, labels=None):
    predictions = (np.array(predictions) >= 0.5).astype('uint8')
    cm = confusion_matrix(predictions, actual_labels, labels=labels)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('conf.png')


def save_model(state, epoch, save_dir, best=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
    if best:
        save_file_path = os.path.join(save_dir, 'model_best.pth')
    torch.save(state, save_file_path)