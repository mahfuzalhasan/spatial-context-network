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
# saved_model_path = "./model_dir/caranet/model_best.pth"
#######FocusNet
saved_model_path = "./model_dir/snet/model_best.pth"


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