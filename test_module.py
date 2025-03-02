import os
import time
import sys
import gc
import numpy as np
from collections import OrderedDict
import csv

sys.path.append('..')

import os
from metric import dice_value, dice_coeff, dice_score_by_data_torch, recall
from utils import *
from config import *
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import gc
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from collections import OrderedDict

import torch
import torch.nn.functional as F

from visualizer import write_img



def structure_loss(pred, mask):
	avg_pooling = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
	neg_part_base = 1
	
	#omitting
	weit =  neg_part_base + 5*avg_pooling  
														
	bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
	wbce = (weit*bce)
	wbce = wbce.sum(dim=(2, 3))/weit.sum(dim=(2, 3))
	
	pred = torch.sigmoid(pred)
	inter = ((pred * mask)*weit).sum(dim=(2, 3))
	union = ((pred + mask)*weit).sum(dim=(2, 3))
	wiou = 1 - ((inter + 1)/(union - inter+1))
	
	m_wbce = wbce.mean()
	m_iou = wiou.mean()

	return m_wbce, m_iou


def test_seg(dataloader, model, criterion, run_id, mixed_precision=True, device='cuda', train=True):
	
	t1 = time.time()
	patient_ids = []
	dice_scores = []
	recall_scores = []

	wbce_loss = []
	wiou_loss = []
	losses = []

	focal_losses = []
	tversky_losses = []

	dice_threshold = 0.7
	plot_img = 1
	
	model.eval()
		
	with torch.set_grad_enabled(train):
		for idx, (pat_id, inp_8_bit, inputs, labels, cls_labels) in enumerate(dataloader):
			inputs = inputs.to(f'cuda:{model.device_ids[0]}', non_blocking=True) #16-bit
			labels = labels.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
			inp_8_bit = inp_8_bit.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
			patient_ids.append(pat_id)
			outputs, lateral_map_2, lateral_map_1 = model(inputs)
			# I have never trained model on fl
			# fl --> focal loss
			# tl --> Tversky Loss
			fl, tl = criterion(outputs, labels)
			wbce2, wiou2 = structure_loss(lateral_map_2, labels)
			wbce1, wiou1 = structure_loss(lateral_map_1, labels)

			wbce = wbce2+wbce1
			wiou = wiou2+wiou1
			
			loss = tl + 2*wbce + wiou

			wbce_loss.append(2*wbce.item())
			wiou_loss.append(wiou.item())
			focal_losses.append(fl.item())
			tversky_losses.append(tl.item())

			losses.append(loss.item())


			if outputs.shape[1]>1:
				predictions = torch.nn.functional.softmax(outputs, dim=1)
				pred_labels = torch.argmax(predictions, dim=1) 
				pred_labels = pred_labels.float()
				outputs_prob = torch.unsqueeze(pred_labels, dim=1)
			else:
				outputs_prob = (torch.sigmoid(outputs)>dice_threshold).float()
				if partial_map:
					pd_outputs = (torch.sigmoid(lateral_map_1)>dice_threshold).float()
			#####
			if idx%plot_img==0:
				if partial_map:
					visuals = OrderedDict([('input', inp_8_bit[0:8, :, :, :]),
											('mask', labels[0:8, :, :, :]),
											('output', outputs_prob[0:8, :, :, :]),
											('partial_d', pd_outputs[0:8, :, :, :])])
				else:
					visuals = OrderedDict([('input', inp_8_bit[0:8, :, :, :]),
										('mask', labels[0:8, :, :, :]),
										('output', outputs_prob[0:8, :, :, :])])
								
				write_img(visuals, run_id, "all", idx, test=True, caranet=False, heatmap_prediction=False)					

			elapsed = int(time.time() - t1)
			eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
			### Dice
			dice_val = dice_score_by_data_torch(labels, outputs, threshold=dice_threshold).detach().cpu().numpy()
			dice_scores.extend(dice_val) ###[]
			### Dice done

			### Recall
			recall_score_batch = recall(labels, outputs, threshold = dice_threshold).detach().cpu().numpy()
			recall_scores.extend(recall_score_batch)
			### Recall Done
			
			display = 500

			if idx%display==0:
				dice_running_avg = np.mean(dice_scores)
				recall_running_avg = np.mean(recall_scores)
				msg = f'Progress: [{idx}/{len(dataloader)}] loss: {(np.mean(losses)):.4f} wbce_l:{np.mean(wbce_loss):.4f}  wiou_l:{np.mean(wiou_loss):.4f} Dice {dice_running_avg:.4f} Recall:{recall_running_avg}'
				print(msg)

	# WBCE and WIOU Loss
	avg_loss = np.mean(losses)
	avg_wbce_l = np.mean(wbce_loss)
	avg_wiou_l = np.mean(wiou_loss)

	mean_dice = np.mean(dice_scores)
	mean_recall = np.mean(recall_scores)

	msg = f'Final loss: {avg_loss:.4f} wbce_l:{avg_wbce_l:.4f}  wiou_l:{avg_wiou_l:.4f} Dice {mean_dice:.4f} Recall:{mean_recall:.4f} '
	print(msg)
	return patient_ids, avg_loss, mean_dice, mean_recall, avg_wbce_l, avg_wiou_l


