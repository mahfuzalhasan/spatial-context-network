import os
import numpy as np
import natsort
import random
import pickle
import copy

def file_save(data, save_path, file_name):
    if not os.path.exists(os.path.join(save_path, file_name)):
        with open(os.path.join(save_path, file_name), 'wb') as f:
            pickle.dump(data, f)


def split_save(split_ids, data_path, save_path, train=False, val=False):
    
    pos_sample_file = "lymph_node.pkl"
    neg_sample_file = "no_lymph_node.pkl"
    
    if train:
        folder = "train"
    elif val:
        folder="val"
    else:
        folder="test"
        
    lymph_node_data = []
    no_lymph_node_data = []
    
    for pat_id in split_ids:
        #print('pat id: ',pat_id)
        img_folder = os.path.join(data_path, pat_id, 'images')
        label_dict = pickle.load(open(os.path.join(data_path, pat_id, 'labels.pkl'), 'rb'))
        img_paths = os.listdir(img_folder)
        for i, img_id in enumerate(img_paths):
            img_path = os.path.join(img_folder, img_id)
            slice_num = int(img_id[:img_id.rindex('.')])
            label = label_dict[slice_num]       # slice --> 0/1
            #print(f'slice_num:{slice_num} label:{label}')
            if label==1:
                lymph_node_data.append(img_path)
            else:
                no_lymph_node_data.append(img_path)
    save_path = os.path.join(save_path, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f'{folder}::LN slices:{len(lymph_node_data)} blank:{len(no_lymph_node_data)}')
    file_save(lymph_node_data, save_path, pos_sample_file)
    file_save(no_lymph_node_data, save_path, neg_sample_file)


def create_split(patient_ids, start, end, data_path, save_path):
    # based on patient id/ patient number
    patient_ids_copy = copy.deepcopy(patient_ids)
    
    split_ids = patient_ids_copy[200:]
    # print(f'split:{split_ids}')
    # exit()
    split_save(split_ids, data_path, save_path, train=False, val=False)
    
    split_ids = patient_ids_copy[start:end]
    split_save(split_ids, data_path, save_path, val=True)
    print('val:{val_ids:{split_ids}}')
    
    del patient_ids_copy[start:end]
    print('train:{len(patient_ids_copy)}')
    split_save(patient_ids_copy, data_path, save_path, train=True)

if __name__=='__main__':
    num_slices = 0
    data_path = f"./data/ln_ct_{num_slices}_npz"
    save_path = "./data/split/five_fold"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    patient_ids = os.listdir(data_path)
    patient_ids = natsort.natsorted(patient_ids)
    print(patient_ids)
    fold = 5
    data_per_fold = 40      # Total 200 data for training--> each val fold --> 40 data
    for i in range(1, fold+1):
        start = (i-1)*data_per_fold
        end = (i-1)*data_per_fold+data_per_fold
        fold_save_path = os.path.join(save_path, f'fold_{i}')
        if not os.path.exists(fold_save_path):
            os.makedirs(fold_save_path)
        create_split(patient_ids, start, end, data_path, fold_save_path)