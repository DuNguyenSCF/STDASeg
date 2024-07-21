import os
import sys
# sys.path.append('../utils')
import argparse
from utils import args_main_Supervised, helper_functions
# import helper_functions

import albumentations as albu
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

import functools

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom

# arg_parser = argparse.ArgumentParser()
# arg_parser = args_main_supervised_training.add_train_args(arg_parser)
# args = arg_parser.parse_args("")

args = args_main_Supervised.initialize_Supervised_train_args()


# # Preprocessing data
# preprocessing_params = {}
# preprocessing_params["input_space"] = "RGB"
# preprocessing_params["input_range"] = [0,1]
# if args.model == "TransU" or args.model == "SwinU":
#     preprocessing_params["mean"] = [0.485, 0.456, 0.406]
#     preprocessing_params["std"] = [0.229, 0.224, 0.225]

# elif args.load_pretrained_weights: 
#     preprocessing_params["mean"] = [0.485, 0.456, 0.406]
#     preprocessing_params["std"] = [0.229, 0.224, 0.225]
# else:
#     preprocessing_params["mean"] = None # [0.485, 0.456, 0.406]
#     preprocessing_params["std"] = None # [0.229, 0.224, 0.225]
    
# preprocessing_fn = functools.partial(helper_functions.preprocess_input, **preprocessing_params)

# def get_preprocessing(preprocessing_fn):
#     """Construct preprocessing transform

#     Args:
#         preprocessing_fn (callbale): data normalization function
#             (can be specific for each pretrained neural network)
#     Return:
#         transform: albumentations.Compose

#     """

#     _transform = [
#         albu.Lambda(image=preprocessing_fn),
#         albu.Lambda(image=helper_functions.to_tensor_img, mask=helper_functions.to_tensor_msk),
#     ]
#     return albu.Compose(_transform)


# Data augmentation   
def get_training_augmentation(args):
    train_transform = [
        albu.RandomResizedCrop(args.patch_size[0], args.patch_size[1], scale=(0.8, 1), 
                               ratio=(0.95, 1.12), 
                               interpolation=0, always_apply=True, p=0.5),
        
        albu.OneOf(
            [
                albu.HorizontalFlip(p=0.5),
                albu.Rotate(limit=[-20,20], p=0.5),
                albu.Transpose(p=0.5),
            ]
        ),
        
    ]
    return albu.Compose(train_transform)


class Train_Supervised_DataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        train_filenames = None,
        num=None,
        preprocessing = None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.train_filenames = train_filenames
        self.sample_list = []
        self.preprocessing = preprocessing
        self.transform = transform

        # For src 1
        with open(self._base_dir + f"/{self.train_filenames}", "r") as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {} samples".format(len(self.sample_list)))
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = np.array(Image.open(self._base_dir + "/{}".format(case)))
        mask = np.array(Image.open(self._base_dir + "/{}".format(case.replace('images','masks').replace('.jpg','.png'))).convert('L'))
        mask = (mask/255)
        samples = {"image_ori": image, "mask_ori": mask}
        
        if self.transform:
            sample_trans = self.transform(image=samples['image_ori'], mask=samples['mask_ori'])
            # image, mask = sample_trans['image'], sample_trans['mask']
            
            if self.preprocessing:
                sample_pre = self.preprocessing(image=sample_trans['image'], mask=sample_trans['mask'])
        else:
            sample_pre = self.preprocessing(image=samples['image_ori'], mask=samples['mask_ori'])
        
        samples['image'] = sample_pre['image']
        samples['mask'] = sample_pre['mask']
        samples['idx'] = idx
        samples['name'] = case.split('/')[-1].split('.')[0]
        
        return samples
    
    
class Val_DataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        val_filenames = None,
        num=None,
        preprocessing = None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.val_filenames = val_filenames
        self.sample_list = []
        self.preprocessing = preprocessing
        self.transform = transform

        # For src 1
        with open(self._base_dir + f"/{self.val_filenames}", "r") as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {} samples".format(len(self.sample_list)))
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = np.array(Image.open(self._base_dir + "/{}".format(case)))
        mask = np.array(Image.open(self._base_dir + "/{}".format(case.replace('images','masks').replace('.jpg','.png'))).convert('L'))
        mask = (mask/255)
        samples = {"image_ori": image, "mask_ori": mask}
        
        if self.transform:
            sample_trans = self.transform(image=samples['image_ori'], mask=samples['mask_ori'])
            # image, mask = sample_trans['image'], sample_trans['mask']
            
            if self.preprocessing:
                sample_pre = self.preprocessing(image=sample_trans['image'], mask=sample_trans['mask'])
        else:
            sample_pre = self.preprocessing(image=samples['image_ori'], mask=samples['mask_ori'])
        
        samples['image'] = sample_pre['image']
        samples['mask'] = sample_pre['mask']
        samples['idx'] = idx
        samples['name'] = case.split('/')[-1].split('.')[0]
        
        return samples



