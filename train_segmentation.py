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

from train_module import train_val_seg
from config import *
from LNDataset import LNDataset, LNDataModule
from utils import *
from model.s_net import s_net

from losses import FocusNetLoss
from augmentation import Augmentation

seed_everything(SEED)

""" Balanced Sampling for Training
9k pos samples with lhymph nodes 
9k neg samples from 14k neg samples ---> 9k neg sample

# 1. smaller objects --> imbalance in each image
# 2. no lymoh node images >>>>> lymph node images ---> imbalance in the dataset
# by equal sampling --> tried to remove 2nd level of imbalance """

def load_data(save_path, split='train'):
    save_path = os.path.join(save_path, split)
    pos_samples = pickle.load(open(os.path.join(save_path, 'lymph_node.pkl'), 'rb'))
    neg_samples = pickle.load(open(os.path.join(save_path, 'no_lymph_node.pkl'), 'rb'))
    samples = []
    samples.extend(pos_samples)

    print(f'######## split:{split} ###########')
    print(f'pos:{len(pos_samples)}')
    if split=='train':
        random.shuffle(neg_samples)
        print(f'neg:{len(neg_samples)}')
        # random.shuffle(neg_samples)
        neg_samples = neg_samples[:len(pos_samples)]
    print(f'spit:{split} selected neg:{len(neg_samples)}')
    samples.extend(neg_samples)
    if split != 'test':
        random.shuffle(samples)
    
    return samples

def data_module_creation(train_set, val_set, test_set):
    aug = Augmentation()
    train_ds = LNDataset(train_set, dim=sz, num_class=num_class, transforms=aug)
    valid_ds = LNDataset(val_set, dim=sz, num_class=num_class)
    test_ds = LNDataset(test_set, dim=sz, num_class=num_class)
    print(len(train_ds), len(valid_ds), len(test_ds))
    data_module = LNDataModule(train_ds, valid_ds, test_ds, batch_size=batch_size)
    return train_ds, valid_ds, test_ds, data_module
        
def main():
    save_path = f'./data/split/five_fold/fold_{val_fold}'
    print(f'############## data path:{save_path} ##########')
    optim = torch.optim.AdamW
    criterion = FocusNetLoss()	# Binary focal and tversky loss
    ### model generation
    model = s_net(1, 1)
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids = device_ids)
        model.to(f'cuda:{device_ids[0]}', non_blocking=True)

    start_epoch = 0
    if resume:
        print('##########loading pre-trained model###########')
        saved_states = torch.load(resume_path)
        model.load_state_dict(saved_states['model'], strict=False)
        start_epoch = saved_states['epoch'] + 1
        print('##########Done###########')

    trainable_params = model.parameters()

    plist = [
        {'params': trainable_params,  'lr': learning_rate}
    ]

    optimizer = optim(trainable_params, learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7)

    timestamp = datetime.datetime.now().strftime('%m-%d-%y_%H%M%S')
    run_id = f"fold_{val_fold}_{timestamp}"
    print(f"run_id:{run_id}")

    writer = SummaryWriter(os.path.join('./history_dir/logs', str(run_id)))

    best_dice = 0.0
    patience_counter = 0

    # training starts
    #[ 0 , 200[ --> 0-199
    for epoch in range(start_epoch, n_epochs):
        train_set = load_data(save_path)
        val_set = load_data(save_path, split='val')
        test_set = load_data(save_path, split='test')
        _, _, _, data_module = data_module_creation(train_set, val_set, test_set)
        #exit()

        train_loss, train_dice, train_recall, t_fl, t_tl, t_wbce_l, t_wiou_l, model = train_val_seg(epoch, data_module.train_dataloader(), model, optimizer, criterion, run_id, device='cuda')

        valid_loss, valid_dice, valid_recall, v_fl, v_tl, v_wbce_l, v_wiou_l = train_val_seg(epoch, data_module.val_dataloader(), model, optimizer, criterion, run_id, device='cuda', train=False)

        print("#"*10)
        print(f"Epoch {epoch} Report:")
        print(f"Validation Loss: {valid_loss :.4f} dice: {valid_dice :.4f} recall:{valid_recall:.4f}")
        lr_scheduler.step(valid_dice)

        # save model after every 5 epoch
        if epoch % 5 == 0 and epoch >0:
            state = {'model': model.module.state_dict(), 
            'optim': optimizer.state_dict(), 
            'scheduler':lr_scheduler.state_dict(),
            'valid_loss':valid_loss, 
            'valid_dice':valid_dice,
            'valid_recall':valid_recall,
            'epoch':epoch}
            save_model(state, epoch, os.path.join(model_dir, str(run_id)))
           
        if valid_dice > best_dice:
            best_dice = valid_dice
            state = {'model': model.module.state_dict(), 
            'optim': optimizer.state_dict(),
            'scheduler':lr_scheduler.state_dict(),
            'valid_loss':valid_loss, 
            'valid_dice':valid_dice, 
            'epoch':epoch}
            print(f'$$$$$$$$$ best from epoch: {epoch} dice:{best_dice} $$$$$$$$$')
            save_model(state, epoch, os.path.join(model_dir, str(run_id)), best=True)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'patience counter:{patience_counter}')
            if patience_counter == 30:
                print(f'dice score didnt improve for 30 epochs')
                break

        writer.add_scalar('Training Total Loss', train_loss, epoch)
        writer.add_scalar('Train wBCE Loss', t_wbce_l, epoch)
        writer.add_scalar('Train wiou Loss', t_wiou_l, epoch)
        writer.add_scalar('Train Focal Loss', t_fl, epoch)
        writer.add_scalar('Train Tversky Loss', t_tl, epoch)
        writer.add_scalar('Training Avg D_Coeff', train_dice, epoch)
        writer.add_scalar('Training Recall', train_recall, epoch)

        writer.add_scalar('Valid Total Loss', valid_loss, epoch)
        writer.add_scalar('Valid wBCE Loss', v_wbce_l, epoch)
        writer.add_scalar('Valid wiou Loss', v_wiou_l, epoch)
        writer.add_scalar('Validation Focal Loss', v_fl, epoch)
        writer.add_scalar('Validation Tversky Loss', v_tl, epoch)
        writer.add_scalar('Valid Avg D_Coeff', valid_dice, epoch)
        writer.add_scalar('Valid Recall', valid_recall, epoch)
        print("#"*10,"\n")
   
if __name__== '__main__':
    main()
    
