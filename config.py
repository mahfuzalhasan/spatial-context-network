import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import cv2
import albumentations as A

data_dir = './data/ct_221'
new_data_dir = f"{data_dir}_window" #./data/ct_221_window
patient_ids = os.listdir(data_dir)
model_dir = "model_dir"
history_dir = "history_dir"
device_ids = [0, 1, 2]
kernel_size = 13

heatmap_prediction = False

resume = False
resume_path = "./model_dir/23-05-02_0440/model_64.pth"
only_heatmap_train = False



display_train_value = 200
display_valid_value = 60
plot_train_img = 50
plot_val_img = 15
partial_map = False

display = 100

#######################Testing
#######CaraNet
# saved_model_path = "./model_dir/23-02-09_1815/model_45.pth"
#######FocusNet
#saved_model_path = "./model_dir/23-02-02_0100/model_119.pth"
#saved_model_path = "./model_dir/23-02-07_2136/model_129.pth"
#saved_model_path = "./model_dir/23-02-16_0140/model_2.pth"
# saved_model_path = "./model_dir/23-02-16_2156/model_10.pth"
#saved_model_path = "./model_dir/23-03-09_1818/model_18.pth"
# saved_model_path = "./model_dir/23-03-20_0632/model_70.pth" # w/o ra attention
# saved_model_path = "./model_dir/23-03-20_1212/model_34.pth"
# saved_model_path = "./model_dir/23-04-02_1901/model_66.pth"
# saved_model_path = "./model_dir/23-05-02_0440/model_64.pth"
# saved_model_path = "./model_dir/23-05-04_2119/model_58.pth"   #with heatmap
# saved_model_path = "./model_dir/23-05-06_1811/model_45.pth"   #with heatmap

# saved_model_path = "./model_dir/23-05-07_0551/model_45.pth" #with localization finetune

# saved_model_path = "./model_dir/24-03-05_2121/model_best.pth"
# saved_model_path = "./model_dir/24-03-06_1156/model_best.pth"
saved_model_path = "./model_dir/23-04-02_1901/model_66.pth"


# saved_model_path = "./model_dir/24-03-07_1701/model_best.pth"   #fold 0
# saved_model_path = "./model_dir/24-03-10_2258/model_best.pth"   #fold 1
# saved_model_path = "./model_dir/24-03-12_1802/model_best.pth"   #fold 2
# saved_model_path = "./model_dir/24-03-13_1251/model_best.pth"   #fold 3
# saved_model_path = "./model_dir/24-03-13_2147/model_best.pth"   #fold 4


SEED = 2022
num_class = 2
learning_rate = 5e-5
n_epochs = 150
sz = 384
n_fold = 5
val_fold = 1
mixed_precision = 0
pretrained_model = 'resnet50'
model_name = f"{pretrained_model}"
batch_size = 16
output_dir = "./history_dir/outputs"
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

#configure augmentations
aug_dict = {
 'noisy':A.GaussNoise(var_limit=(10, 10), p=0.4),
 'blur':A.Blur(p=0.3),
 'shift_scale_rot_zoom_out':A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.25, 
 border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), rotate_limit=15, p=0.5),
 'discontinued': A.Cutout(num_holes=40, max_h_size=4, max_w_size=4, fill_value=255, p=0.3),
 'zoom_in':A.RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=0.6),
 }
aug_list = list(aug_dict.values())

train_aug = A.Compose([
    A.OneOf( 
        aug_list,
    p=0.3),
    A.Resize(sz, sz, always_apply=True),
    ],    
      )
# train_aug = None
val_aug = A.Compose([A.Resize(sz, sz, p=1, always_apply=True)])
# val_aug = None
