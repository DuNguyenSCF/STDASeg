import os
import sys
import argparse
from . import ramps

import numpy as np
import torch

from medpy import metric # import module metric from medpy package


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

# Preprocessing input
def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

def to_tensor_img(x, **kwargs):
    return torch.from_numpy((x.astype(np.float32)).transpose(2, 0, 1))

def to_tensor_msk(x, **kwargs):
    return torch.from_numpy((x.astype(np.uint8))) # WxH

def to_tensor_msk_test(x, **kwargs):
    return torch.from_numpy(np.expand_dims(x.astype(np.uint8), axis=0)) # 1xWxH


# def get_current_consistency_weight(epoch):
#     # Consistency ramp-up from https://arxiv.org/abs/1610.02242
#     return args_STDASeg.consistency * ramps.sigmoid_rampup(epoch, args_STDASeg.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
        
def calculate_metric_percase_mod(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum() == 0:
        if pred.sum() == 0:
            return 1, 1
        else:
            return 0, 0
    else:
        if pred.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return dice, hd95
        else:
            return 0, 0
    
def test_single_volume_mod(image, label, net, classes):
    
    image = image.float().cuda()
    label = label.squeeze(0).numpy()
    
    net.eval()
    with torch.no_grad():
        prediction = torch.argmax(torch.softmax(
            net(image), dim=1), dim=1).squeeze(0)
        prediction = prediction.cpu().detach().numpy()
        
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_mod(
            prediction == i, label == i))
    return metric_list


class AverageMeter(object):
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count+=n
        self.avg = self.sum/self.count
        