import os
from glob import glob
import random
import shutil
import sys

import argparse
import logging
import itertools
import functools
import h5py

from typing import Any, Optional, Tuple
from abc import ABC, abstractmethod
import errno

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid


import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom

from config import get_config
import augmentations
from augmentations.ctaugment import OPS
import albumentations as albu
# from dataloaders import utils
from dataloaders.crack_datasets import get_training_augmentation, Train_Supervised_DataSets, Val_DataSets


from networks.net_factory import net_factory
from modules_models.Swinv2_Unet import swin_unet
from networks.vit_seg_modeling import VisionTransformer as TransU
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.SemiCrack import projectors, classifier
from networks.discriminator import FCDiscriminator


from utils import args_main_Supervised, helper_functions, losses, metrics, ramps
from utils.helper_functions import calculate_metric_percase_mod, test_single_volume_mod
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss

from tensorboardX import SummaryWriter
import time
from tqdm import tqdm

args = args_main_Supervised.initialize_Supervised_train_args()
# print(args)

def create_model(args, model):
    
    if args.model == 'TransU':
        print(f'Creating {args.model} model...!')
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.patch_size[0] / args.vit_patches_size), int(args.patch_size[0] / args.vit_patches_size))
            model = TransU(config_vit, img_size=args.patch_size[0], num_classes=config_vit.n_classes).cuda()
            model.load_from(weights=np.load(config_vit.pretrained_path))
    elif args.model == 'SwinU':
        print(f'Creating {args.model} model...!')
        if args.patch_size[0] == 256:
            model = swin_unet(size="swinv2_small_window8_256", img_size=args.patch_size[0]).cuda()
        elif args.patch_size[0] == 512:
            model = swin_unet(size="swinv2_base_window16_256", img_size=args.patch_size[0]).cuda()
    elif args.model == 'unet_mod':
        print(f'Creating {args.model} model...!')
        model = net_factory(net_type=args.model, in_chns=3, class_num=args.num_classes, weights = None)
    else: # for CNN models with pre-trained weights for encoder
        if args.load_pretrained_weights:
            model = net_factory(net_type=args.model, in_chns=3, class_num=args.num_classes, weights=args.encoder_weights)
        else:
            model = net_factory(net_type=args.model, in_chns=3, class_num=args.num_classes, weights = None)
            
    return model

# args.load_pretrained_weights = True
# model = create_model(args, args.model)

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

# Preprocessing data
preprocessing_params = {}
preprocessing_params["input_space"] = "RGB"
preprocessing_params["input_range"] = [0,1]
if args.model == "TransU" or args.model == "SwinU":
    preprocessing_params["mean"] = [0.485, 0.456, 0.406]
    preprocessing_params["std"] = [0.229, 0.224, 0.225]

elif args.load_pretrained_weights: 
    preprocessing_params["mean"] = [0.485, 0.456, 0.406]
    preprocessing_params["std"] = [0.229, 0.224, 0.225]
else:
    preprocessing_params["mean"] = None # [0.485, 0.456, 0.406]
    preprocessing_params["std"] = None # [0.229, 0.224, 0.225]
    
preprocessing_fn = functools.partial(helper_functions.preprocess_input, **preprocessing_params)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=helper_functions.to_tensor_img, mask=helper_functions.to_tensor_msk),
    ]
    return albu.Compose(_transform)

# sample_trans_data = Train_Supervised_DataSets(
#     base_dir = args.root_path,
#     train_filenames = args.train_filenames,
#     preprocessing=get_preprocessing(preprocessing_fn),
#     transform = get_training_augmentation(args)
# )
# idx = random.randint(0,len(sample_trans_data))
# sample_trans = sample_trans_data[idx]
# fig, axs = plt.subplots(2,2,figsize=(6,6), squeeze=True)
# print(sample_trans['name'])
# axs[0,0].imshow(sample_trans['image_ori'])
# axs[0,1].imshow(sample_trans['mask_ori'])
# axs[1,0].imshow(torch.movedim(sample_trans['image'],0, -1))
# axs[1,1].imshow(sample_trans['mask'])

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    model = create_model(args, args.model)
    
    if args.train_transform:
        train_transforms = get_training_augmentation(args)
        print('Train data will be transformed','\n')
    else:
        train_transforms = None
        print('Train data will not be transformed','\n')
    if args.val_transform:
        val_transforms = get_training_augmentation(args)
        print('Valid data will be transformed','\n')
    else:
        val_transforms = None
        print('Valid data will not be transformed','\n')
        
    
    
    db_train = Train_Supervised_DataSets(
                    base_dir = args.root_path,
                    train_filenames = args.train_filenames,
                    preprocessing=get_preprocessing(preprocessing_fn),
                    transform = train_transforms
                )
    db_val = Val_DataSets(
                    base_dir = args.root_path,
                    val_filenames = args.val_filenames,
                    preprocessing=get_preprocessing(preprocessing_fn),
                    transform = val_transforms
                )
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 100 == 0:
                image = volume_batch[0, :, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % args.save_checkpoint == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_mod(
                        sampled_batch["image"], sampled_batch["mask"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    if args.save_best_ckpt_with_iter:
                        save_mode_path = os.path.join(snapshot_path,
                                                      'iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance, 4)))
                        torch.save(model.state_dict(), save_mode_path)
                    
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best)
                    print('Saved best checkpoint at iter {}!'.format(iter_num))

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            # if iter_num % 5000 == 0: # save model every 5000 iters
            #     save_mode_path = os.path.join(
            #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    print('-'*100)
    print('Start training...!')
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    train(args, snapshot_path)
    
    torch.cuda.empty_cache()