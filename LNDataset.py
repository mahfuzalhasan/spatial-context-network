import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import gc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from platform import system
import os

from skimage.measure import label, regionprops
from scipy import signal
from scipy.stats import multivariate_normal

import torchvision



os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models

from tqdm import tqdm_notebook as tqdm
import platform
import random
import warnings
import pickle
import albumentations as A
import math

from config import *
from augmentation import Augmentation
from utils import *

warnings.filterwarnings('ignore')

def load_data(save_path, split='train'):
	save_path = os.path.join(save_path, split)
	pos_samples = pickle.load(open(os.path.join(save_path, 'lymph_node.pkl'), 'rb'))
	neg_samples = pickle.load(open(os.path.join(save_path, 'no_lymph_node.pkl'), 'rb'))
	samples = []
	samples.extend(pos_samples)
	if split=='train':
		random.shuffle(neg_samples)
		random.shuffle(neg_samples)
		neg_samples = neg_samples[:len(pos_samples)]
	samples.extend(neg_samples)
	return samples

def data_module_creation(train_set, val_set, test_set):
	aug = Augmentation()
	train_ds = LNDataset(train_set, dim=sz, num_class=num_class,
	transforms=aug)
	valid_ds = LNDataset(val_set, dim=sz, num_class=num_class)
	test_ds = LNDataset(test_set, dim=sz, num_class=num_class)
	print(len(train_ds), len(valid_ds), len(test_ds))
	data_module = LNDataModule(train_ds, valid_ds, test_ds, batch_size=batch_size)
	return train_ds, valid_ds, test_ds, data_module

class LNDataset(Dataset):
    def __init__(self, image_ids, dim=256, num_class=25, transforms=None, 
                 heatmap_on = heatmap_prediction):
        super().__init__()
        self.image_ids = image_ids
        self.dim = dim
        self.num_class = num_class
        self.transforms = transforms
        self.heatmap_on = heatmap_on
        self.sigma = 10
        self.transform_val = A.Compose(
            [
                A.CenterCrop (height = dim, width = dim)
            ],
            additional_targets = {'image0':'image', 'image1':'image'}
        )
        random.shuffle(self.image_ids)

    def convert_to_tensor(self, img):
        img_torch = torch.from_numpy(img)
        img_torch = img_torch.type(torch.FloatTensor)
        img_torch = img_torch.permute(-1, 0, 1)
        return img_torch

    def region_finding(self, mask):
        center_map = np.zeros((mask.shape[0], mask.shape[1]))
        mask[mask<255] = 0
        #mask = mask/255
        #mask[mask<1] = 0
        #mask = mask.astype('uint8')
        #print(f'mask max:{np.amax(mask)} min:{np.amin(mask)}')
        label_img = label(mask)
        region = regionprops(label_img)
        #print(f'regions:{len(region)}')
        for props in region:
            y0, x0 = props.centroid
            y0 = math.ceil(y0)
            x0 = math.ceil(x0)
            #print('centriod: ',y0, x0)
            center_map[y0, x0] = 1
        return center_map

    def gaussian_heatmap(self, mask, idx):
        #print(f'img:{idx}')
        center_map = self.region_finding(mask.astype('uint8'))
        h, w = center_map.shape
        H1 = torch.linspace(1, h, h)
        W1 = torch.linspace(1, w, w)
        [H, W] = torch.meshgrid(H1, W1)
        xs, ys = np.where(center_map==1)
        if len(xs)==0 or len(ys)==0:
            return torch.from_numpy(center_map)
        else:
            #print(f'img:{idx} centroids:{xs}{ys}')
            HH = H - xs[0]
            WW = W - ys[0]
            cube = HH*HH + WW*WW
            cube /= (2. * self.sigma * self.sigma)
            cube = torch.exp(-cube)
            #cube[cube<0.01] = 0
            #cube = cube.unsqueeze(0)
            #torchvision.utils.save_image(cube, './history_dir/output_check/'+str(idx)+'_'+str(0)+'_h.jpg', normalize=True, nrow=1, range=(0, 1))
            for i,x in enumerate(xs):
                if i==0:
                    continue
                HH = H - xs[i]
                WW = W - ys[i]

                tmp_cube = HH*HH + WW*WW
                tmp_cube /= (2. * self.sigma * self.sigma)
                tmp_cube = torch.exp(-tmp_cube)
                tmp_cube[tmp_cube<0.01] = 0
                #tmp_cube = tmp_cube.unsqueeze(0)
                #torchvision.utils.save_image(tmp_cube, './history_dir/output_check/'+str(idx)+'_'+str(i)+'_h.jpg', normalize=True, nrow=1, range=(0, 1))
                cube = torch.add(cube, tmp_cube)
            cube[cube<0.01] = 0
        return cube

    def points_to_gaussian_heatmap(self, centers, height, width, scale):
        gaussians = []
        for y,x in centers:
            s = np.eye(2)*scale
            g = multivariate_normal(mean=(x,y), cov=s)
            gaussians.append(g)
        return gaussians

        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_8_bit = cv2.imread(image_id)
        file_path = image_id.replace('/images/', '/files/').replace('.jpg','.npy')
        image = np.load(file_path)
        
        mask = cv2.imread(image_id.replace('/images/', '/masks/'))
        
        image_8_bit = image_8_bit[:,:,0]
        mask = mask[:,:,0]
        
        if self.transforms is not None:
            image, mask, image_8_bit = self.transforms.generation(image, mask, image_8_bit)
        else:
            transformed = self.transform_val(image=image, image0=mask, image1=image_8_bit)
            image, mask, image_8_bit = transformed['image'], transformed['image0'], transformed['image1']
        th, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)

        if self.heatmap_on:
            heatmap = self.gaussian_heatmap(mask, idx)
            max_hmap = torch.max(heatmap)
            min_hmap = torch.min(heatmap)
            heatmap = (heatmap - min_hmap)/(max_hmap - min_hmap + 1e-4)
            heatmap = heatmap.unsqueeze(0)
            heatmap = heatmap.type(torch.FloatTensor)
            # print("heatmap: ",heatmap.shape)
            #heatmap = heatmap.permute(-1, 0, 1)


        image = np.expand_dims(image, axis=2)
        img_tensor = self.convert_to_tensor(image)

        image_8_bit = image_8_bit/255.0
        image_8_bit = np.expand_dims(image_8_bit, axis=2)
        img_8_bit_tensor = self.convert_to_tensor(image_8_bit)
        
        
        mask = mask/255
        mask = np.expand_dims(mask, axis=2)
        mask_tensor = self.convert_to_tensor(mask)
        
        patient_id = image_id.split('/')[-3]

        #if self.labels is not None:
        mask_value = np.count_nonzero(mask)
        if mask_value>0:
            labels = 1
        else:
            labels = 0

        target = self.onehot(self.num_class, labels) 
        if self.heatmap_on:
            return img_8_bit_tensor, img_tensor, mask_tensor, target, heatmap
        else:
            return patient_id, img_8_bit_tensor, img_tensor, mask_tensor, target

        

    def __len__(self):
        return len(self.image_ids)
    
    def onehot(self, num_class, target):
        vec = torch.zeros(num_class, dtype=torch.float32)
        #print(vec, target)
        vec[target] = 1.
        return vec
    
    def get_labels(self):
        return list(self.labels)


class LNDataModule():
    def __init__(self, train_ds, valid_ds, test_ds, 
    batch_size=32, sampler=None, shuffle=True, num_workers=4):
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        if "Windows" in system():
            self.num_workers = 0
        else:
            self.num_workers = num_workers
        self.sampler = sampler

    def train_dataloader(self):
        if self.sampler is not None:
            sampler = self.sampler(labels=self.train_ds.get_labels(), mode="upsampling")
            train_loader = DataLoader(self.train_ds,batch_size=self.batch_size, 
            sampler= sampler, shuffle=False, drop_last=True,
            num_workers=self.num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle, 
            drop_last=True,
            num_workers=self.num_workers, pin_memory=True)
            print("train dataloader: ",len(train_loader))
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.valid_ds,batch_size=self.batch_size, drop_last=True,
        shuffle=True,
         num_workers=self.num_workers, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        if self.test_ds is not None:
            test_loader = DataLoader(self.test_ds,batch_size=1, 
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True)
        return test_loader

if __name__=="__main__":
    data_path = "../DATA/lymph_node/ct_221/_window"
    save_path = f"{data_path}/split/full"
    os.makedirs('./history_dir/output_check/', exist_ok=True)
    
    train_set = load_data(save_path)
    val_set = load_data(save_path, split='val')
    test_set = load_data(save_path, split='test')
    train_ds, valid_ds, test_ds, data_module = data_module_creation(train_set, val_set, test_set)

    for i in range(len(train_ds)):
        valid_ds.__getitem__(i)
        if i==100:
            exit()
        