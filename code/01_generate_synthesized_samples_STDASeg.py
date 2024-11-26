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

from tqdm import tqdm

# from dataloaders import utils
from utils import args_main_STDASeg, helper_functions
from dataloaders import crack_datasets_STDASeg
from dataloaders.crack_datasets_STDASeg import get_syn_training_augmentation, get_trg_training_augmentation, Train_Src_DataSets, Train_CP_DataSets, Train_Dove_DataSets, Train_FPIE_DataSets, Train_Trg_DataSets, Val_DataSets 



arg_parser = argparse.ArgumentParser()
arg_parser = args_main_STDASeg.add_generate_syndata_STDASeg_args(arg_parser)
args = arg_parser.parse_args()




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

# Normalization
if args.load_pretrained_weights:
    preprocessing_params["mean"] = [0.485, 0.456, 0.406]
    preprocessing_params["std"] = [0.229, 0.224, 0.225]

else:
    preprocessing_params["mean"] = None # [0.485, 0.456, 0.406]
    preprocessing_params["std"] = None # [0.229, 0.224, 0.225]

# Denormalization for saving pseudo labels
if args.load_pretrained_weights:
    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
else:
    denormalize = transforms.Normalize(
        mean=[0, 0, 0],
        std=[0, 0, 0]
    )
    
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

def create_syn_sample_folder(args):
    print(f'Saving some synthesized samples for training using {args.syn_method} method...!')
    save_syn_sample_path = f'{args.saved_syn_samples}/{args.syn_method}'
    if os.path.exists(save_syn_sample_path):
        # Iterate over each file in the directory
        print('Removing the exsisting files...!')
        for filename in os.listdir(save_syn_sample_path):
            file_path = os.path.join(save_syn_sample_path, filename)
            # Check if it's a file and not a subdirectory
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
    else:
        print('New folder is created for saving samples...!')
        os.makedirs(save_syn_sample_path, exist_ok=True)
        
    return save_syn_sample_path

#########################################################################
#############  Generate synthesized samples for checking  ###############
#########################################################################
if args.syn_concrete_data:
    print("Synthesizing concrete crack data...!")
elif args.syn_pav_data:
    print("Synthesizing pavment crack data...!")
else:
    print("Using source data for training...!")


# Generate synthesized images
def generate_syn_samples(args):
    if args.syn_method == 'cutpaste':
        syn_method = args.syn_method
        print(f"Start synthesizing image using {args.syn_method}...!")
        # Generate some synthesized samples for first look

        save_syn_sample_path = create_syn_sample_folder(args)

        sample_trans_data = Train_CP_DataSets(
            base_dir=args.root_path,
            train_src_filenames = args.train_syn_src_filenames,
            train_trg_filenames = args.train_syn_trg_filenames,
            syn_method = syn_method,
            preprocessing=get_preprocessing(preprocessing_fn),
            transform = None,
            shuffle = False
        )

        for idx in tqdm(range(0, args.saved_syn_samples)):
            sample_trans = sample_trans_data[idx]
            name = sample_trans['name']
            img = sample_trans['image']
            msk = sample_trans['mask']
            fig, axs = plt.subplots(1,3,figsize=(9,6), squeeze=True)
            axs[0].imshow(sample_trans['crk_image'])
            axs[1].imshow(sample_trans['non_crk'])
            axs[2].imshow(torch.movedim(denormalize(sample_trans['image']), 0, -1))
            for i in range(0, 3):
                axs[i].set_axis_off()
                # axs[i].set_title(img_titles[i], loc='center', y=1.05)
            plt.subplots_adjust(wspace=0.05, hspace=0)
            plt.savefig(f"{save_syn_sample_path}/CP_{name}.jpg", bbox_inches='tight')
            plt.close()
            # break

    elif args.syn_method == 'fpie':
        syn_method = args.syn_method
        print(f"Start synthesizing image using {args.syn_method}...!")
        # Generate some synthesized samples for first look

        save_syn_sample_path = create_syn_sample_folder(args)

        sample_trans_data = Train_FPIE_DataSets(
            base_dir=args.root_path,
            train_src_filenames = args.train_syn_src_filenames,
            train_trg_filenames = args.train_syn_trg_filenames,
            syn_method = args.syn_method,
            proc = crack_datasets_STDASeg.proc,
            properties = crack_datasets_STDASeg.properties, # a dic of properties for syn data
            preprocessing=get_preprocessing(preprocessing_fn),
            transform = None
        ) 

        for idx in tqdm(range(0, args.saved_syn_samples)):
            sample_trans = sample_trans_data[idx]
            name = sample_trans['name']
            img = sample_trans['image']
            msk = sample_trans['mask']
            fig, axs = plt.subplots(1,3,figsize=(9,6), squeeze=True)
            axs[0].imshow(sample_trans['crk_image'])
            axs[1].imshow(sample_trans['noncrk_image'])
            axs[2].imshow(torch.movedim(denormalize(sample_trans['image']), 0, -1))
            for i in range(0, 3):
                axs[i].set_axis_off()
                # axs[i].set_title(img_titles[i], loc='center', y=1.05)
            plt.subplots_adjust(wspace=0.05, hspace=0)
            plt.savefig(f"{save_syn_sample_path}/FPIE_{name}.jpg", bbox_inches='tight')
            plt.close()
            # break

    elif args.syn_method == 'dove':
        syn_method = crack_datasets_STDASeg.Dove_G.eval()
        print(f"Start synthesizing image using {args.syn_method}...!")
        # Generate some synthesized samples for first look

        save_syn_sample_path = create_syn_sample_folder(args)
        sample_trans_data = Train_Dove_DataSets(
            base_dir=args.root_path,
            train_src_filenames = args.train_syn_src_filenames,
            train_trg_filenames = args.train_syn_trg_filenames,
            syn_method = syn_method,
            preprocessing=get_preprocessing(preprocessing_fn),
            transform = None,
            dove_preprocess = crack_datasets_STDASeg.config['preprocess'],
            dove_load_size = crack_datasets_STDASeg.config['crop_size'],
            dove_crop_size = crack_datasets_STDASeg.config['crop_size'], 
            dove_no_flip = True,
            )

        for idx in tqdm(range(0, args.saved_syn_samples)):
            sample_trans = sample_trans_data[idx]
            name = sample_trans['name']
            img = sample_trans['image']
            msk = sample_trans['mask']
            fig, axs = plt.subplots(1,3,figsize=(9,6), squeeze=True)
            axs[0].imshow(sample_trans['crk_image'])
            axs[1].imshow(sample_trans['noncrk_image'])
            axs[2].imshow(torch.movedim(denormalize(sample_trans['image']), 0, -1))
            for i in range(0, 3):
                axs[i].set_axis_off()
                # axs[i].set_title(img_titles[i], loc='center', y=1.05)
            plt.subplots_adjust(wspace=0.05, hspace=0)
            plt.savefig(f"{save_syn_sample_path}/Dove_{name}.jpg", bbox_inches='tight')
            plt.close()
            # break

    else:
        syn_method = 'uda'
        print(f"Source data is directly used for training STDASeg in the UDA manner...!")
        

    
if __name__ == "__main__":
    print('-'*100)
    
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
    
    print(f"Start synthesizing data using {args.syn_method} method...!")
    
    generate_syn_samples(args)
        
    
    torch.cuda.empty_cache()