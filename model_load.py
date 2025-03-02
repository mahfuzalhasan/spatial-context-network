import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.s_net import s_net

print(f'torch version: {torch.__version__}')




def load_model(num_channel, num_classes, saved_model_path, device_ids):
    model = s_net(channel, num_classes, se=True, norm='bn')
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids = device_ids)
        model.to(f'cuda:{device_ids[0]}', non_blocking=True)
        
    #### Loading Pretrained Weights
    saved_states = torch.load(saved_model_path)
    model.load_state_dict(saved_states['model'])
    return model


if __name__=="__main__":
    
    channel = 1
    num_classes = 1
    saved_model_path = "./trained_models/model_best.pth"
    device_ids = [0, 1]

    model = load_model(channel, num_classes, saved_model_path, device_ids)
    
    B = 8
    C = 1
    H = 384
    W = 384
    inputs = torch.randn(B, C, H, W)
    inputs = inputs.cuda()
    outputs = model(inputs)
    print(f'output:{outputs[0].size()} partial_1:{outputs[1].size()} partial_2:{outputs[2].size()}')
