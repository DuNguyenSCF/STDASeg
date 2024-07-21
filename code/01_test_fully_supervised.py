import os
from glob import glob
import random
import shutil
import sys
# sys.path.insert(1, '/data/gpfs/projects/punim1699/data_dunguyen/01_Paper_2/Codes/Modules_Models')
import argparse
import logging
import itertools
import functools
import h5py

from typing import Any, Optional, Tuple
from abc import ABC, abstractmethod
import errno

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset
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
# from dataloaders.crack_datasets import Test_Trg_DataSets


from networks.net_factory import net_factory
from modules_models.Swinv2_Unet import swin_unet
from networks.vit_seg_modeling import VisionTransformer as TransU
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.SemiCrack import projectors, classifier
from networks.discriminator import FCDiscriminator


from utils import args_main_Supervised, ramps, helper_functions
from utils.helper_functions import calculate_metric_percase_mod, test_single_volume_mod, AverageMeter
from utils.crack_metrics_for_testing import AIU, ODS, EarlyStopping, Precision, Recall, Dice_Score, IoU
from PRE_RE_F1_IoU_over_thresholds import mask_normalize, compute_pre_rec_iou, compute_PRE_REC_FM_IoU_of_methods, plot_save_pr_curves, plot_save_fm_curves, plot_save_iou_curves
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss

from patchify import patchify, unpatchify


from tensorboardX import SummaryWriter
from time import time
from datetime import datetime
from tqdm import tqdm
now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

arg_parser = argparse.ArgumentParser()
arg_parser = args_main_Supervised.add_test_args(arg_parser)
args = arg_parser.parse_args()

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

# Preprocessing data for testing
preprocessing_params = {}
preprocessing_params["input_space"] = "RGB"
preprocessing_params["input_range"] = [0,1]
if args.preprocess:
    print("Testing data will be normalized!")
    preprocessing_params["mean"] = [0.485, 0.456, 0.406]
    preprocessing_params["std"] = [0.229, 0.224, 0.225]
else:
    print("Testing data will NOT be normalized!")
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
        albu.Lambda(image=helper_functions.to_tensor_img, mask=helper_functions.to_tensor_msk_test), # different from training
    ]
    return albu.Compose(_transform)

class Test_Trg_DataSets(Dataset):
    def __init__(
        self,
        trg_base_dir=None,
        trg_fullsize_filenames=None,
        trg_crop_filenames=None,
        test_full = True,
        preprocessing = None,
    ):
        self._trg_base_dir = trg_base_dir
        self.trg_fullsize_filenames = trg_fullsize_filenames
        self.trg_crop_filenames = trg_crop_filenames
        self.sample_list = []
        self.test_full = test_full
        self.preprocessing = preprocessing
        if self.test_full:
            with open(self._trg_base_dir + f"/{self.trg_fullsize_filenames}", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

            print("total {} samples".format(len(self.sample_list)))
        else:
            with open(self._trg_base_dir + f"/{self.trg_crop_filenames}", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

            print("total {} samples".format(len(self.sample_list)))
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.test_full:
            image = np.array(Image.open(self._trg_base_dir + "/{}".format(case)))
            mask = np.array(Image.open(self._trg_base_dir + "/{}".format(case.replace('images','masks').replace('.jpg','.png'))).convert('L'))
            mask = (mask/255)
        else:
            image = np.array(Image.open(self._trg_base_dir + "/{}".format(case)))
            mask = np.array(Image.open(self._trg_base_dir + "/{}".format(case.replace('images','masks').replace('.jpg','.png'))).convert('L'))
            mask = (mask/255)
        
        samples = {"image_ori": image, "mask_ori": mask}
        sample_pre = self.preprocessing(image=samples['image_ori'], mask=samples['mask_ori'])
        samples['image'] = sample_pre['image']
        samples['mask'] = sample_pre['mask']
        samples['idx'] = idx
        samples['name'] = case.split('/')[-1].split('.')[0]
            
        # image = torch.from_numpy((image/255).astype(np.float32).transpose(2, 0, 1))
        # mask = torch.from_numpy(np.expand_dims(mask.astype(np.uint8), axis=0))
        # samples = {"image": image, "mask": mask}
        # samples['idx'] = idx
        # samples['name'] = case.split('/')[-1].split('.')[0]
        
        return samples


def test(args, IoU, Dice_Score):
        
    avg_meter = AverageMeter()

    trg_test_data = Test_Trg_DataSets(
        trg_base_dir = args.root_path,
        trg_fullsize_filenames=args.trg_fullsize_filenames,
        trg_crop_filenames=args.trg_crop_filenames,
        test_full = args.test_full,
        preprocessing = get_preprocessing(preprocessing_fn)
    )

    trg_testloader = DataLoader(
                    trg_test_data,
                    batch_size=args.batch_size, 
                    shuffle=False,)
    trg_testloader_iter = enumerate(trg_testloader)
    _, trg_test_img = trg_testloader_iter.__next__()
    trg_test_img['image'].shape, trg_test_img['mask'].shape, trg_test_img['name']

    num_classes = args.num_classes
    snapshot_path = "../model/{}_{}/{}".format(
            args.exp, args.labeled_num, args.model)

    model = create_model(args, args.model)

    print('Loading checkpoint....!')

    model = model.cuda()
    checkpoint_path = os.path.join(snapshot_path,args.checkpoint)
    model.load_state_dict(torch.load(checkpoint_path))

    method = "{}_{}_{}_iters_{}".format(args.exp.split('/')[-1], args.labeled_num, args.model, args.iters)
    saved_fullscale_preds_dir = os.path.join(args.results_dir,f'destination_fullscale_masks/{args.method}')
    saved_cropoverlaped_preds_dir = os.path.join(args.results_dir,f'destination_cropoverlaped_masks/{args.method}')
    saved_crop_preds_dir = os.path.join(args.results_dir,f'destination_crop{args.patch_size[0]}_masks/{args.method}')
    saved_acc_dir = os.path.join(args.results_dir,f'Pre_Rec_Fm_IoU_Curve/{args.method}')

    # Metric selection
    iou_score = IoU(threshold=0.5, from_logits=False)
    dice_coef = Dice_Score(threshold=0.5, from_logits=False)

    model.eval()
    if args.test_full:
        print('Testing on fullsize images....!')
        start = time()
        # Write results of all full-scale image
        ftext = open('Test_logs/{}_Test_Thrh05_Fullscale.txt'.format(method),'a+')
        ftext.write('*'*50 +"\n")
        # f.write('\n')
        ftext.write(date_time + ': ' + method + '_' + args.checkpoint.split('.pth')[0] + '\n')
        ftext.write('*'*50 + '\n')

        saved_preds_folders = os.path.join(saved_fullscale_preds_dir,method)
        os.makedirs(saved_preds_folders, exist_ok=True)
        files = glob(saved_preds_folders + '/*')
        if files == []:
            print('\nThe destination mask folder is ready for saving mask')
        else:
            for f in files:
                os.remove(f)
            print('\nRemoved all files in destination_mask folder!!!')


        with torch.no_grad():
            for _, test_sampled_batch in tqdm(enumerate(trg_testloader)):
                test_img_batch = test_sampled_batch['image']
                test_msk_batch = test_sampled_batch['mask']
                unpatch_arr = np.zeros((1, args.overlap_h, args.overlap_w, num_classes, args.patch_size[0], args.patch_size[1]))
                inp_arr = test_img_batch.squeeze().numpy() # size [3, 3456, 5184]
                patches_inp = patchify(inp_arr, (3, args.patch_size[0], args.patch_size[1]), step=args.patch_size[0]-args.overlap_pixels)
                for i in range(0, args.overlap_h):
                    for j in range(0, args.overlap_w):
                        patches_inp_tensor = torch.tensor(patches_inp[0,i,j,:,:,:]).unsqueeze(0).cuda()
                        pred = model(patches_inp_tensor)
                        for c_i in range(num_classes):
                            unpatch_arr[0,i,j,c_i,:,:] = pred.squeeze(0).detach().cpu().numpy()[c_i,:,:]
                unpatch_output = unpatchify(unpatch_arr, (num_classes, args.fullsize_h, args.fullsize_w))
                output = torch.tensor(unpatch_output)
                output = torch.softmax(output, dim=0)
                output = output[1].unsqueeze(0).unsqueeze(0).cuda()
                target = test_msk_batch.cuda()
                # output.shape, target.shape
                dice_score = dice_coef(output, target)
                ftext.write('%s ------ %.4f \n'%(test_sampled_batch['name'][0], dice_score))

                avg_meter.update(dice_score, test_img_batch.size(0))
                # output = torch.sigmoid(output).cpu().numpy()
                output = output.cpu().numpy()
                for i in range(len(output)):
                    img = output[i,0]
                    img_PIL = Image.fromarray((img*255).astype(np.uint8)).convert('RGB')
                    img_PIL.save(os.path.join(saved_preds_folders, test_sampled_batch['name'][0] +'.png'))

        ftext.write('Mean dice score: %.4f \n'%(avg_meter.avg))
        ftext.write('Took: %.2f to inference \n'%(time() - start))
        ftext.close()
        print('Took ', time() - start)
        print('DICE: %.4f' % avg_meter.avg)
    elif args.test_crop_overlaped:
        print('Testing on croped images with overlaped regions....!')
        start = time()
        # Write results of all full-scale image
        ftext = open('Test_logs/{}_Test_Thrh05_CropOverlaped.txt'.format(method),'a+')
        ftext.write('*'*50 +"\n")
        # f.write('\n')
        ftext.write(date_time + ': ' + method + '_' + args.checkpoint.split('.pth')[0] + '\n')
        ftext.write('*'*50 + '\n')

        saved_preds_folders = os.path.join(saved_cropoverlaped_preds_dir,method)
        os.makedirs(saved_preds_folders, exist_ok=True)
        files = glob(saved_preds_folders + '/*')
        if files == []:
            print('\nThe destination mask folder is ready for saving mask')
        else:
            for f in files:
                os.remove(f)
            print('\nRemoved all files in destination_mask folder!!!')


        with torch.no_grad():
            for _, test_sampled_batch in tqdm(enumerate(trg_testloader)):
                test_img_batch = test_sampled_batch['image']
                test_msk_batch = test_sampled_batch['mask']
                unpatch_arr = np.zeros((1, args.overlap_h, args.overlap_w, num_classes, args.patch_size[0], args.patch_size[1]))
                inp_arr = test_img_batch.squeeze().numpy() # size [3, 3456, 5184]
                patches_inp = patchify(inp_arr, (3, args.patch_size[0], args.patch_size[1]), step=args.patch_size[0]-args.overlap_pixels)
                for i in range(0, args.overlap_h):
                    for j in range(0, args.overlap_w):
                        patches_inp_tensor = torch.tensor(patches_inp[0,i,j,:,:,:]).unsqueeze(0).cuda()
                        pred = model(patches_inp_tensor)
                        for c_i in range(num_classes):
                            unpatch_arr[0,i,j,c_i,:,:] = pred.squeeze(0).detach().cpu().numpy()[c_i,:,:]
                unpatch_output = unpatchify(unpatch_arr, (num_classes, args.fullsize_h, args.fullsize_w))
                output = torch.tensor(unpatch_output)
                output = torch.softmax(output, dim=0)
                output = output[1].unsqueeze(0).unsqueeze(0).cuda()
                target = test_msk_batch.cuda()
                # output.shape, target.shape
                dice_score = dice_coef(output, target)
                ftext.write('%s ------ %.4f \n'%(test_sampled_batch['name'][0], dice_score))

                avg_meter.update(dice_score, test_img_batch.size(0))
                # output = torch.sigmoid(output).cpu().numpy()
                output = output.cpu().numpy()
                for i in range(len(output)):
                    img = output[i,0]
                    img_PIL = Image.fromarray((img*255).astype(np.uint8)).convert('RGB')
                    img_PIL.save(os.path.join(saved_preds_folders, test_sampled_batch['name'][0] +'.png'))

        ftext.write('Mean dice score: %.4f \n'%(avg_meter.avg))
        ftext.write('Took: %.2f to inference \n'%(time() - start))
        ftext.close()
        print('Took ', time() - start)
        print('DICE: %.4f' % avg_meter.avg)
    else:
        print('Testing on cropped images...!')
        start = time()
        # Write results of all full-scale image
        ftext = open('Test_logs/{}_Test_Thrh05_Crop{}.txt'.format(method, args.patch_size[0]),'a+')
        ftext.write('*'*50 +"\n")
        # f.write('\n')
        ftext.write(date_time + ': ' + method + '_' + args.checkpoint.split('.pth')[0] + '\n')
        ftext.write('*'*50 + '\n')

        saved_preds_folders = os.path.join(saved_crop_preds_dir,method)
        os.makedirs(saved_preds_folders, exist_ok=True)
        files = glob(saved_preds_folders + '/*')
        if files == []:
            print('\nThe destination mask folder is ready for saving mask')
        else:
            for f in files:
                os.remove(f)
            print('\nRemoved all files in destination_mask folder!!!')

        with torch.no_grad():
            for _, test_sampled_batch in tqdm(enumerate(trg_testloader)):
                test_img_batch = test_sampled_batch['image']
                test_msk_batch = test_sampled_batch['mask']
                inp = test_img_batch.cuda()
                target = test_msk_batch.cuda()
                pred = model(inp)
                output = pred.squeeze(0).detach().cpu()
                output = torch.softmax(output, dim=0)
                output = output[1].unsqueeze(0).unsqueeze(0)

                dice_score = dice_coef(output, target)
                ftext.write('%s ------ %.4f \n'%(test_sampled_batch['name'][0], dice_score))

                avg_meter.update(dice_score, test_img_batch.size(0))
                # output = torch.sigmoid(output).cpu().numpy()
                output = output.cpu().numpy()
                for i in range(len(output)):
                    img = output[i,0]
                    img_PIL = Image.fromarray((img*255).astype(np.uint8)).convert('RGB')
                    img_PIL.save(os.path.join(saved_preds_folders, test_sampled_batch['name'][0] + '.png'))

        ftext.write('Mean dice score: %.4f \n'%(avg_meter.avg))
        ftext.write('Took: %.2f to inference \n'%(time() - start))
        ftext.close()
        print('Took ', time() - start)
        print('DICE: %.4f' % avg_meter.avg)

    print('-'*100)
    print('PR, RE, F1, Iou over thresholds')

    mask_folder = [method]

    if args.test_full:
        gt_dir = args.root_path + '/' + args.fullsize_gt
    else:
        gt_dir = args.root_path + '/' + args.crop_gt

    gt_name_list = list(sorted(glob(gt_dir+'/*.png')))
    mask_dir_lists = [saved_preds_folders+'/'] # Change if merge predicted masks
    if args.test_full:
        save_dir = os.path.join(saved_acc_dir,method+'/fullscale/')
        os.makedirs(save_dir, exist_ok=True)
    elif args.test_crop_overlaped:
        save_dir = os.path.join(saved_acc_dir,method+'/cropoverlaped/')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.join(saved_acc_dir,method+f'/crop{args.patch_size[0]}/')
        os.makedirs(save_dir, exist_ok=True)

    print("------2. Compute the Precision, Recall and F-measure of Methods------")
    PRE, REC, FM, IoU, OIS_FM, gt2rs_fm = compute_PRE_REC_FM_IoU_of_methods(gt_name_list,mask_dir_lists,beta=1.0)

    mybins=np.arange(0, 256)
    thrh_range = np.array(mybins[0:-1]).astype('float')/255.0

    samples = np.zeros((OIS_FM.shape[0]),).astype('str')
    max_F1 = np.zeros((OIS_FM.shape[0]),)
    thrh = np.zeros((OIS_FM.shape[0]),)

    for i in range(0, OIS_FM.shape[0]):
        reverseF1 = OIS_FM[i][0][::-1]
        max_F1[i] = np.max(reverseF1)
        thrh[i] = thrh_range[np.argmax(reverseF1)]
        samples[i] = gt_name_list[i].split('/')[-1]
        # break

    # Write csv files
    PRE_REC_FM_IoU_history = {'PRE':PRE.reshape(255,), 'REC':REC.reshape(255,), 'FM':FM.reshape(255,), 'IoU':IoU.reshape(255,)}
    df = pd.DataFrame.from_dict(PRE_REC_FM_IoU_history)
    df.to_csv(os.path.join(save_dir, 'PRE_REC_FM_IoU_{}_accuracy.csv'.format(method)),index = False, header=True)

    OIS_history = {'Samples':samples, 'max_F1':max_F1, 'Threshold':thrh}
    df = pd.DataFrame.from_dict(OIS_history)
    df.to_csv(os.path.join(save_dir, 'OIS_{}_accuracy.csv'.format(method)),index = False, header=True)

    for i in range(0,FM.shape[0]):
        print(">>", mask_folder[i],":", "num_rs/num_gt-> %d/%d,"%(int(gt2rs_fm[i][0]),len(gt_name_list)), "maxF->%.3f, "%(np.max(FM,1)[i]), "meanF->%.3f, "%(np.mean(FM,1)[i]))
    print('\n')
    for i in range(0,IoU.shape[0]):
        print(">>", mask_folder[i],":", "num_rs/num_gt-> %d/%d,"%(int(gt2rs_fm[i][0]),len(gt_name_list)), "maxIoU->%.3f, "%(np.max(IoU,1)[i]), "meanIoU->%.3f, "%(np.mean(IoU,1)[i]))
    print('\n')

    print(">>", mask_folder[i],":", "OIS->%.3f, "%(np.mean(max_F1)))
    print('\n')

    print(">>", mask_folder[i],":", "mean_threshold->%.3f, "%(np.mean(thrh)), "var_threshold->%.3f, "%(np.var(thrh)))
    print('\n')


    # Plot
    method_names = [method]
    lineSylClr = ['r-'] # curve style, same size with rs_dirs
    linewidth = [1] # line width, same size with rs_dirs
    # xrange = (0,1.0), # the showing range of x-axis
    # yrange = (0,1.0), # the showing range of y-axis
    dataset_name = mask_folder[0]


    plot_save_pr_curves(PRE=PRE, REC=REC, method_names=method_names, 
                       lineSylClr=lineSylClr,
                       linewidth=linewidth,
                       xrange=(0.0, 1.0),
                       yrange=(0.0, 1.0),
                       dataset_name=dataset_name,
                       save_dir=save_dir,
                       save_fmt='.png')

    plot_save_fm_curves(FM=FM, mybins=np.arange(0, 256),
                       method_names=method_names,
                       lineSylClr=lineSylClr,
                       linewidth=linewidth,
                       xrange=(0.0, 1.0),
                       yrange=(0.0, 1.0),
                       dataset_name=dataset_name,
                       save_dir=save_dir,
                       save_fmt='.png')

    plot_save_iou_curves(IoU=IoU, mybins=np.arange(0, 256),
                       method_names=method_names,
                       lineSylClr=lineSylClr,
                       linewidth=linewidth,
                       xrange=(0.0, 1.0),
                       yrange=(0.0, 1.0),
                       dataset_name=dataset_name,
                       save_dir=save_dir,
                       save_fmt='.png')


    print('Finish testing process for this case !!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    torch.cuda.empty_cache()
    
    
if __name__ == '__main__':
    print('-'*100)
    print('Start testing...!')
    test(args, IoU, Dice_Score)
