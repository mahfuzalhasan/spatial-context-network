from multiprocessing import Pool
from sklearn.model_selection import GroupKFold, StratifiedKFold
from config import *
from utils import *
from tqdm import tqdm as T
import numpy as np
import natsort
import cv2
import nrrd
import pandas as pd

data_dir = "./data/ln_ct/"
label_dict = {'patient_id':[], 'slice_num':[], 'label':[]}
patient_ids = os.listdir(data_dir)


# for pat_id in T(patient_ids):
def data_processing(args):
	pat_id, num_slices = args
	new_data_dir = f"./data/ln_ct_{num_slices}_npz"	# Slices to Crop --> 0 for 2D Training. >0 for 2.5D training
	pat_id = str(pat_id)
	try:
		data_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'IM00' in f][0]
	except:
		print(f"Problem with {pat_id}")    
	seg_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'Segmentation' in f][0]

	img_pat_id = nrrd.read(os.path.join(data_dir, pat_id, data_file))[0]
	mask_pat_id = nrrd.read(os.path.join(data_dir, pat_id, seg_file))[0]
	padded_data = np.pad(img_pat_id, ((0, 0), (0, 0), (num_slices, num_slices)), mode='constant')
	os.makedirs(os.path.join(new_data_dir, pat_id, 'images'), exist_ok=True)
	os.makedirs(os.path.join(new_data_dir, pat_id, 'masks'), exist_ok=True)
	for i in range(num_slices, padded_data.shape[-1] - num_slices):
		_, img = window_image(padded_data[:,:,i-num_slices:i+num_slices+1], 40, 400, 0, 1)
		mask = mask_pat_id[:, :, i - num_slices]
		mask[mask>0] = 1
		label_dict['patient_id'].append(pat_id)
		label_dict['slice_num'].append(i)
		if np.sum(mask) == 0:
			label_dict['label'].append(0)
		else: label_dict['label'].append(1)
		np.savez(os.path.join(new_data_dir, pat_id, f'images/{i-num_slices}'), img)
		np.savez(os.path.join(new_data_dir, pat_id, f'masks/{i-num_slices}'), mask)
		# cv2.imwrite(os.path.join(new_data_dir, pat_id, f'images/{i}.png'), img)
		# cv2.imwrite(os.path.join(new_data_dir, pat_id, f'masks/{i}.png'), mask)
	return label_dict

def datapath(patient_id, slice_num):return f"{patient_id}/images/{slice_num}.npz"

if __name__ == '__main__':
	num_slices = 0
	args_list = [(patient_id, num_slices) for patient_id in patient_ids]
	with Pool(16) as p:
		results = list(T(p.imap(data_processing, args_list), total=len(patient_ids), colour='red'))
	# 	print(len(results))
	# # Merge results iteratively using a loop:
	# merged_label_dict = {}

	# for d in results:
	# 	for key, value in d.items():
	# 		if key in merged_label_dict:
	# 			merged_label_dict[key].extend(value)
	# 		else:
	# 			merged_label_dict[key] = value

	# df = pd.DataFrame(merged_label_dict).drop_duplicates()
	# df.to_csv(f"{new_data_dir}/labels.csv", index=False)
	# gkf = GroupKFold(n_splits=5)
	# patient_ids = df.explode('patient_id')['patient_id'].unique().tolist()
	# # patient_ids.sort()                    #sort alphabetically
	# patient_ids = natsort.natsorted(patient_ids)        # sort numerically
	# train_pat_ids = patient_ids[:200]
	# test_pat_ids = patient_ids[200:]
	# print(f'test_ids:{test_pat_ids}')
	# df['path'] = list(map(datapath, df['patient_id'], df['slice_num']))   
	# df['path'] = df['path'].map(lambda x: f"{new_data_dir}/{x}")
	# train_df = df[df["patient_id"].isin(train_pat_ids)].reset_index(drop=True)
	# test_df = df[df["patient_id"].isin(test_pat_ids)].reset_index(drop=True)

	# # for i, (train_index, val_index) in enumerate(gkf.split(train_df['path'], train_df['label'], groups=train_df['patient_id'])):
	# #     train_idx = train_index
	# #     val_idx = val_index
	# #     train_df.loc[val_idx, 'fold'] = i

	# pat_id_dict = {}

	# for i, pat_id in enumerate(train_pat_ids):
	# 	pat_id_dict.update({pat_id:i*5//len(train_pat_ids)})

	# train_df['fold_patient'] = train_df['patient_id'].map(lambda x: pat_id_dict[x])

	# # train_df['fold'] = train_df['fold'].astype('int')
	# train_df['fold_patient'] = train_df['fold_patient'].astype('int')
	# # df['fold'] = df['patient_id'].map(lambda x: 0 if x in train_pat_ids else 1)
	# print(train_df.head(20))

	# train_df.to_csv(f"{new_data_dir}/train_labels.csv", index=False)
	# test_df.to_csv(f"{new_data_dir}/test_labels.csv", index=False)
	
	# for fold in range(5):
	# 	df = pd.read_csv(f"{new_data_dir}/train_labels.csv").drop_duplicates()
	# 	train_df = df[(df['fold_patient'] != fold)] 
	# 	valid_df = df[df['fold_patient'] == fold]
	# 	test_df = pd.read_csv(f"{new_data_dir}/test_labels.csv").drop_duplicates()
	# 	print(len(train_df), len(valid_df), len(test_df))

	# 	train_pos = train_df[train_df['label'] == 1] 
	# 	train_neg = train_df[train_df['label'] == 0] 

	# 	valid_pos = valid_df[valid_df['label'] == 1] 
	# 	valid_neg = valid_df[valid_df['label'] == 0] 

	# 	test_pos = test_df[test_df['label'] == 1] 
	# 	test_neg = test_df[test_df['label'] == 0]

	# 	print(f'train:::: pos:{len(train_pos)} neg:{len(train_neg)}')
	# 	print(f'valid:::: pos:{len(valid_pos)} neg:{len(valid_neg)}')
	# 	print(f'test:::: pos:{len(test_pos)} neg:{len(test_neg)}')
	
	
	