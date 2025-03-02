import os
import numpy as np
import glob
from functools import partial
import gc
from matplotlib import pyplot as plt
from matplotlib.pyplot import axis
from datetime import datetime
import shutil
import warnings
import pickle

#### Third party libraries
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")


import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from test_module import test_seg
from config import *
from LNDataset import LNDataset, LNDataModule
from utils import *
from model.s_net import s_net

from losses import FocusNetLoss
from augmentation import Augmentation

seed_everything(SEED)

def load_data(save_path, split='train'):
	save_path = os.path.join(save_path, split)
	pos_samples = pickle.load(open(os.path.join(save_path, 'lymph_node.pkl'), 'rb'))
	neg_samples = pickle.load(open(os.path.join(save_path, 'no_lymph_node.pkl'), 'rb'))
	print(f'total:{len(neg_samples)+len(pos_samples)}')
	samples = []
	samples.extend(pos_samples)
	# Balanced Sampling for Training
	if split=='train':
		random.shuffle(neg_samples)
		# random.shuffle(neg_samples)
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
		


def main():

	save_path = f'./data/split/five_fold/fold_{val_fold}'
	print(f'data path:{save_path}')
	criterion = FocusNetLoss()	#wBCE and wIOU

	### model generation
	model = s_net(1, 1)
	# model = caranet()
	device_ids = [3]
	model.to(f'cuda:{device_ids[0]}', non_blocking=True)
	if torch.cuda.is_available():
		model = nn.DataParallel(model, device_ids = device_ids)
		model.to(f'cuda:{device_ids[0]}', non_blocking=True)
	#### Loading Pretrained Weights
	saved_states = torch.load(saved_model_path)
	# print('keys in saved states: ',saved_states['model'].keys())
	# exit()
	model.load_state_dict(saved_states['model'])
	print('########## Model Loaded ###########')

	index = saved_model_path.rindex('/')
	run_id = saved_model_path[index-20:index]
	print("test run id: ",run_id)

	train_set = load_data(save_path)
	val_set = load_data(save_path, split='val')
	test_set = load_data(save_path, split='test')
	# exit()
	_, _, _, data_module = data_module_creation(train_set, val_set, test_set)

	print("test dataloader: ",len(data_module.test_dataloader()))
	test_loss, test_dice, test_recall, avg_wbce_l, avg_wiou_l = test_seg(data_module.test_dataloader(), model, criterion, run_id, mixed_precision=True, device='cuda', train=False)
	print(f"Test Loss: {test_loss :.4f} Dice: {test_dice :.4f} test_recall:{test_recall}")
   
if __name__== '__main__':
	main()
