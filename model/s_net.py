import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)
model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)

from net import *


# s_net fro Focus_Net
class s_net(nn.Module):
    """share weights before the last conv layer"""
    def __init__(self, channel, num_classes, se=True, reduction=2, norm='bn'):
        super(s_net, self).__init__()
        # downsample twice
        self.conv1x = inconv(channel, 32, norm=norm)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2x = self._make_layer(
            conv_block, 32, 48, 2, se=se, stride=1, reduction=reduction, norm=norm)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv4x = self._make_layer(
            conv_block, 48, 64, 2, se=se, stride=1, reduction=reduction, norm=norm)
        self.conv4xd2 = self._make_layer(
            conv_block, 64, 64, 2, se=se, stride=1, reduction=reduction, norm=norm, dilation_rate=(2, 2))

        # DenseASPP
        current_num_feature = 64
        d_feature0 = 64
        d_feature1 = 32
        dropout0 = 0
        self.ASPP_1 = DenseASPPBlock(input_num=current_num_feature, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(3, 3), drop_out=dropout0, norm=norm)

        self.ASPP_2 = DenseASPPBlock(input_num=current_num_feature+d_feature1*1, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(6, 6), drop_out=dropout0, norm=norm)

        self.ASPP_3 = DenseASPPBlock(input_num=current_num_feature+d_feature1*2, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(12, 12), drop_out=dropout0, norm=norm)

        self.ASPP_4 = DenseASPPBlock(input_num=current_num_feature+d_feature1*3, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(18, 18), drop_out=dropout0, norm=norm)
        current_num_feature = current_num_feature + 4 * d_feature1

        # upsample
        self.up1 = up_block(in_ch=current_num_feature,
              out_ch=48, se=se, reduction=reduction, norm=norm)
        self.literal1 = nn.Conv2d(48, 48, 3, padding=1)

        self.up2 = up_block(in_ch=48, out_ch=32, se=se, reduction=reduction, norm=norm)
        self.literal2 = nn.Conv2d(32, 32, 3, padding=1)

        # output branch
        self.out_conv = nn.Conv2d(32, num_classes, 1, 1)

    def forward(self, x, label=False):
        # encoder
        o1 = self.conv1x(x)
        
        o2 = self.maxpool1(o1)
        o2 = self.conv2x(o2)
        
        o3 = self.maxpool2(o2)
        o3 = self.conv4x(o3)
        
        o4 = self.conv4xd2(o3)
        
        # DenseASPP
        aspp1 = self.ASPP_1(o4)
        feature = torch.cat((aspp1, o4), dim=1)
        
        aspp2 = self.ASPP_2(feature)
        feature = torch.cat((aspp2, feature), dim=1)

        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp4 = self.ASPP_4(feature)
        feature = torch.cat((aspp4, feature), dim=1)
        
        # Decoder
        out, ra_out1 = self.up1(feature, self.literal1(o2))
        ra_out1 = F.interpolate(ra_out1, scale_factor=2, mode='bilinear')
        
        feature_map, ra_out2 = self.up2(out, self.literal2(o1))
        out = self.out_conv(feature_map)

        return out, ra_out1, ra_out2

    def _make_layer(self, block, in_ch, out_ch, num_blocks, se=True, stride=1, reduction=2, dilation_rate=1, norm='bn'):
        layers = []
        layers.append(block(in_ch, out_ch, se=se, stride=stride,
                            reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        for i in range(num_blocks-1):
            layers.append(block(out_ch, out_ch, se=se, stride=1,
                                reduction=reduction, dilation_rate=dilation_rate, norm=norm))

        return nn.Sequential(*layers)

if __name__=='__main__':
	model = s_net(1, 1, se=True, norm='bn')
	print(torch.cuda.is_available())
	print(torch.__version__)

	device_ids = [3]
	saved_model_path = "/home/UFAD/mdmahfuzalhasan/Documents/temp/lymph_node/23-04-02_1901_model_66.pth"
	model.to(f'cuda:{device_ids[0]}', non_blocking=True)
	if torch.cuda.is_available():
		model = nn.DataParallel(model, device_ids = device_ids)
		model.to(f'cuda:{device_ids[0]}', non_blocking=True)
	#### Loading Pretrained Weights
	saved_states = torch.load(saved_model_path)
	model.load_state_dict(saved_states['model'])
	####print(network)
	#exit()
	# model.cuda()
	B = 2
	C = 1
	H = 384
	W = 384
	inputs = torch.randn(B, C, H, W)
	inputs = inputs.cuda()
	outputs = model(inputs)
	# print(outputs.size())
	print(f'output:{outputs[0].size()} partial_1:{outputs[1].size()} partial_2:{outputs[2].size()}')
	# receptive_field_dict = receptive_field(network, (1, 384, 384), device="cpu")
	# print(receptive_field_dict)
	# print(f'output:{outputs[0].size()} partial_1:{outputs[1].size()} partial_2:{outputs[2].size()}')
	##print(f'output:{output.size()}')



