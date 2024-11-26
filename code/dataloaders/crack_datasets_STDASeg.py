import os
import sys
# sys.path.append('../utils')
import argparse
from utils import args_main_STDASeg, helper_functions

import random

import albumentations as albu
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import torchvision.transforms.functional as TF

import functools

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
import cv2

# Modules for FPIE and Dove methods
import code
import modules_models
import dove
from dove import network, dove_network, util
import Fast_Poisson_Image_Editing
from Fast_Poisson_Image_Editing import fpie
from Fast_Poisson_Image_Editing.syn_fpie import EquSolver, GridSolver, BaseProcessor, EquProcessor, GridProcessor 
from fpie import np_solver
from fpie.process import ALL_BACKEND, CPU_COUNT, DEFAULT_BACKEND

# arg_parser = argparse.ArgumentParser()
# arg_parser = args_main_STDASeg_training.add_train_STDASeg_args(arg_parser)
# args = arg_parser.parse_args()

args = args_main_STDASeg.initialize_STDASeg_train_args()


# args.syn_concrete_data = True # for debug, comment when run training as this argument will be assigned via python script

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

# Data augmentation   
def get_syn_training_augmentation():
    train_transform = [
        albu.RandomResizedCrop(args.patch_size[0], args.patch_size[1], scale=(0.8, 1), 
                               ratio=(0.95, 1.12), 
                               interpolation=0, always_apply=True, p=0.2),
        
        albu.OneOf(
            [
                albu.HorizontalFlip(p=0.5),
                albu.Rotate(limit=[-20,20], p=0.5),
                albu.Transpose(p=0.5),
            ],
            p=1,
        ),

        albu.PadIfNeeded(min_height=args.patch_size[0], min_width=args.patch_size[1], always_apply=True, border_mode=0),
        
        albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),

        albu.GaussNoise(p=0.2),

        albu.OneOf(
            [
                albu.Sharpen(p=0.5),
                albu.Blur(blur_limit=3, p=0.3),
                albu.MotionBlur(blur_limit=3, p=0.3),
            ],
            p=0.3,
        ),

    ]
    return albu.Compose(train_transform)

def get_trg_training_augmentation():
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


###########################################
############ FPIE method ##################
###########################################
if args.syn_method == 'fpie':
    CPU_COUNT = 24
    DEFAULT_BACKEND = "numpy"
    ALL_BACKEND = ["numpy"]

    try:
        from fpie import numba_solver
        ALL_BACKEND += ["numba"]
        DEFAULT_BACKEND = "numba"
    except ImportError:
        numba_solver = None  # type: ignore

    try:
        from fpie import taichi_solver
        ALL_BACKEND += ["taichi-cpu", "taichi-gpu"]
        DEFAULT_BACKEND = "taichi-cpu"
    except ImportError:
        taichi_solver = None  # type: ignore

    try:
        from fpie import core_gcc  # type: ignore
        DEFAULT_BACKEND = "gcc"
        ALL_BACKEND.append("gcc")
    except ImportError:
        core_gcc = None

    try:
        from fpie import core_openmp  # type: ignore
        DEFAULT_BACKEND = "openmp"
        ALL_BACKEND.append("openmp")
    except ImportError:
        core_openmp = None

    try:
        from mpi4py import MPI

        from fpie import core_mpi  # type: ignore
        ALL_BACKEND.append("mpi")
    except ImportError:
        MPI = None  # type: ignore
        core_mpi = None

    try:
        from fpie import core_cuda  # type: ignore
        DEFAULT_BACKEND = "cuda"
        ALL_BACKEND.append("cuda")
    except ImportError:
        core_cuda = None
        
        
###################################################
##Config for syn data ############################
#################################################
config = {}
if args.syn_method == 'cutpaste':
    print("No configuration is required for CP method...!")
if args.syn_method == 'fpie':
    print("Get configuration for FPIE method...!")
    config['v'] = True # action="store_true", help="show the version and exit"
    config['check-backend'] = True # action="store_true", help="print all available backends"
    config['gen_type'] = 'cli'
    config['b'] = DEFAULT_BACKEND
    config['c'] = CPU_COUNT
    config['z'] = 1024 # help="cuda block size (only for equ solver)"
    config['method'] = 'equ' # ["equ", "grid"], help="how to parallelize computation"

    if config['gen_type'] == 'cli':
        config['h0'] = 0 # help="mask position (height) on source image", default=0
        config['w0'] = 0 # help="mask position (width) on source image", default=0

        config['h1'] = 0 # "mask position (height) on target image", default=0
        config['w1'] = 0 # "mask position (width) on target image", default=0

        config['p'] = 0 # help="output result every P iteration", default=0

    config['g'] = 'src' # choices=["max", "src", "avg"], help="how to calculate gradient for PIE"
    config['n'] = 10000 # help="how many iteration would you perfer, the more the better"



    config['mpi-sync-interval'] = 100 # help="MPI sync iteration interval", if "mpi" in ALL_BACKEND:

    config['grid-x'] = 8 # help="x axis stride for grid solver",
    config['grid-y'] = 8 # help="y axis stride for grid solver"

    size = args.patch_size[0] # 256

    proc: BaseProcessor

    if config['method'] == 'equ':
        proc = EquProcessor(
          config['g'],
          config['b'],
          config['c'],
          config['mpi-sync-interval'],
          config['z'],
        )
    else:
        proc = GridProcessor(
          config['g'],
          config['b'],
          config['c'],
          config['mpi-sync-interval'],
          config['z'],
          config['grid-x'] ,
          config['grid-y'],
        )

    properties = {
            'h0': config['h0'],
            'w0': config['w0'],
            'h1': config['h1'],
            'w1': config['w1'],
            'n' : config['n'],
            'p' : config['p'],
            'size': args.patch_size[0]
                 }
elif args.syn_method == 'dove':
    print("Get configuration for DOVE method...!")
    
    config['data_root_dir'] = '/data/gpfs/projects/punim1699/data_dunguyen/Chundata_bridgecrack' # where source data located

    if args.syn_concrete_data:
        print("Syn concrete data using pretrained model on concrete composite dataset...!")
        config['dataset'] = 'Chun'
        config['checkpoint'] = 'Chun_generator_param.pkl'
        config['ckpt_path'] = 'pretrained_ckpt/Dove_Gen_concrete_cracks_512'
        # config['note'] = 'Testcase_03_imgharmonize_gen_cracks_512_results'
    elif args.syn_pav_data:
        print("Syn pavement data using pretrained model on pavement composite dataset...!")
        config['dataset'] = 'Pav_self_collect'
        config['checkpoint'] = 'Pav_self_collect_generator_param.pkl'
        config['ckpt_path'] = 'pretrained_ckpt/Dove_Gen_pavement_cracks_512'
        # config['note'] = 'Pav_self_collect_imgharmonize_gen_cracks_results'
    
    config['test_batch_size'] = 1
    config['num_workers'] = 4
    config['input_size'] = 512
    config['preprocess'] = 'none' #'resize_and_crop' #'scale_width_and_crop' or #none
    config['resize_scale'] = 572 # resize scale (0 is false)
    config['crop_size'] = 512
    config['no_fliplr'] = True
    config['inp_c'] = 4
    config['out_c'] = 3
    config['ngf'] = 64
    config['ndf'] = 64
    config['netG'] = 's2ad' # 'resnet_9blocks', 'resnet_6blocks', 'unet_128', 'unet_256'
    config['norm'] = 'instance'
    config['use_dropout'] = True
    config['init_type'] = 'normal'
    config['init_gain'] = 0.02
    config['gpu_ids'] = [0]
    config['netD'] = 'basic'
    config['n_layers_D'] = 3
    config['gan_mode'] = 'wgangp'
    gan_mode = config['gan_mode']
    config['gp_ratio'] = 1.0
    config['lambda_a'] = 1.0
    config['lambda_v'] = 1.0

    config['lrG'] = 0.0002
    config['lrD'] = 0.0002
    config['beta1'] = 0.5
    config['beta2'] = 0.999
    config['G_lr_policy'] = 'linear'
    config['D_lr_policy'] = 'linear'
    config['n_epochs'] = 100 # for linear (LambdaLR) or CosineAnnealingLR lr scheduler
    config['n_epochs_decay'] = 100 # for linear lr scheduler

    config['lr_decay_iters'] = 100 # for step lr scheduler

    config['mode'] = 'min' # for ReduceLROnPlateau lr scheduler
    config['patience'] = 5 # for ReduceLROnPlateau lr scheduler
    config['factor'] = 0.2 # for ReduceLROnPlateau lr scheduler

    config['milestones']='1,2' # for multiStep lr scheduler
    config['gamma']=2/3 # for multiStep lr scheduler

    config['train_epoch'] = config['n_epochs'] + config['n_epochs_decay']
    config['L1_lambda'] = 100

#     config['save_dir'] = '/data/gpfs/projects/punim1699/data_dunguyen/Codes_github/pytorch-pix2pix-master_znxlwm/pytorch-pix2pix-master'
#     config['save_root'] = 'results'
#     # config['note'] = 'Testcase_03_imgharmonize_gen_cracks_results'

#     root = config['save_dir'] + '/' + config['note'] + '_' + config['save_root'] + '/'
#     model = config['dataset'] +'_'
#     if not os.path.isdir(root):
#         os.mkdir(root)
#     fixed_results_path = root + 'fixed_results'
#     if not os.path.isdir(fixed_results_path):
#         os.mkdir(fixed_results_path)
        
###################################################################
if args.syn_method == 'dove':    
    # Transform for dove 
    def get_params(preprocess, load_size, crop_size, size):
        w, h = size
        new_h = h
        new_w = w
        if preprocess == 'resize_and_crop':
            new_h = new_w = load_size
        elif preprocess == 'scale_width_and_crop':
            new_w = opt.load_size
            new_h = opt.load_size * h // w

        x = random.randint(0, np.maximum(0, new_w - crop_size))
        y = random.randint(0, np.maximum(0, new_h - crop_size))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def get_transform(preprocess, load_size, crop_size, no_flip, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))
        if 'resize' in preprocess:
            osize = [load_size, load_size]
            transform_list.append(transforms.Resize(osize, method))
        elif 'scale_width' in preprocess:
            transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)))

        if 'crop' in preprocess:
            if params is None:
                transform_list.append(transforms.RandomCrop(crop_size))
            else:
                transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

        if preprocess == 'none':
            transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

        if not no_flip:
            if params is None:
                transform_list.append(transforms.RandomHorizontalFlip())
            elif params['flip']:
                transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)


    def __transforms2pil_resize(method):
        mapper = {transforms.InterpolationMode.BILINEAR: Image.Resampling.BILINEAR,
                  transforms.InterpolationMode.BICUBIC: Image.Resampling.BICUBIC,
                  transforms.InterpolationMode.NEAREST: Image.Resampling.NEAREST,
                  transforms.InterpolationMode.LANCZOS: Image.Resampling.LANCZOS,}
        return mapper[method]


    def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
        method = __transforms2pil_resize(method)
        ow, oh = img.size
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        if h == oh and w == ow:
            return img

        __print_size_warning(ow, oh, w, h)
        return img.resize((w, h), method)


    def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
        method = __transforms2pil_resize(method)
        ow, oh = img.size
        if ow == target_size and oh >= crop_size:
            return img
        w = target_size
        h = int(max(target_size * oh / ow, crop_size))
        return img.resize((w, h), method)


    def __crop(img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img


    def __flip(img, flip):
        if flip:
            return img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        return img


    def __print_size_warning(ow, oh, w, h):
        """Print warning information about image size(only print once)"""
        if not hasattr(__print_size_warning, 'has_printed'):
            print("The image size needs to be a multiple of 4. "
                  "The loaded image size was (%d, %d), so it was adjusted to "
                  "(%d, %d). This adjustment will be done to all images "
                  "whose sizes are not multiples of 4" % (ow, oh, w, h))
            __print_size_warning.has_printed = True

    Dove_G = dove_network.define_G(
        input_nc = config['inp_c'], 
        output_nc = config['out_c'], 
        ngf = config['ngf'] ,
        netG = config['netG'] ,
        norm=config['norm'],
        use_dropout=config['use_dropout'],
        init_type=config['init_type'],
        init_gain=config['init_gain'],
        gpu_ids=config['gpu_ids'],
    )
    # Dove_G.cuda()
    print("Loading pre-trained weights for Dove generator...!")
    # print(config)
    
    Dove_G.load_state_dict(torch.load(os.path.join(config['ckpt_path'],config['checkpoint'])))
    # Dove_G.load_state_dict(torch.load(root + config['checkpoint']))
    # DoveNet_syn_method = Dove_G.eval()
    # print("Ready to generate syn dove images!")
    
####################################################################
##################### Source dataset ###############################
####################################################################   

class Train_Src_DataSets(Dataset):
    def __init__(
        self,
        max_samples=1000,
        base_dir=None,
        train_filenames = None,
        num=None,
        preprocessing = None,
        transform=None,
    ):
        
        self.num_src_samples = max_samples
        self._base_dir = base_dir
        self.train_filenames = train_filenames
        self.sample_list = []
        self.src_sample_list = []
        self.preprocessing = preprocessing
        self.transform = transform

        # For src 1
        with open(self._base_dir + f"/{self.train_filenames}", "r") as f1:
            self.src_sample_list = f1.readlines()
        self.src_sample_list = [item.replace("\n", "") for item in self.src_sample_list]
        print("There are {} labeled samples".format(len(self.src_sample_list)))
        
        for i in range(self.num_src_samples):
            self.sample_list.append(self.src_sample_list[i%len(self.src_sample_list)])

        print("Total {} labeled samples for training after alignment".format(len(self.sample_list)))
    
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
    
####################################################################
#### Dataset for cutpaste method #######################################
####################################################################

class Train_CP_DataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        train_src_filenames = None,
        train_trg_filenames = None,
        syn_method = None,
        preprocessing = None,
        transform=None,
        shuffle = False
        # dove_model = None
    ):
        self._base_dir = base_dir
        self.train_src_filenames = train_src_filenames
        self.train_trg_filenames = train_trg_filenames
        self.syn_method = syn_method
        self.src_sample_list = []
        self.len_src_samples = None
        self.trg_sample_list = []
        self.len_trg_samples = None
        self.src_sample_list_aligned = []
        self.trg_sample_list_aligned = []
        self.len_data = None
        self.preprocessing = preprocessing
        self.transform = transform
        self.shuffle = shuffle
        # self.dove_model = dove_model

        # For trg noncrack image
        with open(self._base_dir + f"/{self.train_trg_filenames}", "r") as f1:
            self.trg_sample_list = f1.readlines()
        self.trg_sample_list = [item.replace("\n", "") for item in self.trg_sample_list]
        self.len_trg_samples = len(self.trg_sample_list)
        print("The number of noncrack samples: {}".format(self.len_trg_samples))
        # For src crack image
        with open(self._base_dir + f"/{self.train_src_filenames}", "r") as f1:
            self.src_sample_list = f1.readlines()
        self.src_sample_list = [item.replace("\n", "") for item in self.src_sample_list]
        self.len_src_samples = len(self.src_sample_list)
        print("The number of crack samples: {}".format(self.len_src_samples))
        
        self.len_data = max(self.len_trg_samples, self.len_src_samples)
        for i in range(self.len_data):
            self.trg_sample_list_aligned.append(self.trg_sample_list[i%len(self.trg_sample_list)])
            self.src_sample_list_aligned.append(self.src_sample_list[i%len(self.src_sample_list)])

        print("Total {} noncrack samples".format(len(self.trg_sample_list_aligned)))
        print("Total {} crack samples".format(len(self.src_sample_list_aligned)))
        
    
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        if self.shuffle:
            random.shuffle(self.src_sample_list_aligned)
        src_case = self.src_sample_list_aligned[idx]
        crk_image = np.array(Image.open(self._base_dir + "/{}".format(src_case)))
        crk_mask = np.array(Image.open(self._base_dir + "/{}".format(src_case.replace('images','masks').replace('.jpg','.png'))).convert('L'))
        crk_mask_ = np.array(Image.open(self._base_dir + "/{}".format(src_case.replace('images','masks').replace('.jpg','.png')))) / 255
        # crk_mask_dialated = Image.open(self._base_dir + "/{}".format(src_case.replace('images','masks_dilated').replace('.jpg','.png')))
        crk_mask_inv_arr = 1 - crk_mask_ # to get background region
        
        trg_case = self.trg_sample_list_aligned[idx]
        noncrk_image = np.array(Image.open(self._base_dir + "/{}".format(trg_case)))
        
        fg_inp_arr = (crk_image * crk_mask_).astype(np.uint8)# only get crack region for input
        bg_inp_arr = (noncrk_image * crk_mask_inv_arr).astype(np.uint8)
        comp_inp = (fg_inp_arr+bg_inp_arr)
        
        samples = {
                "crk_image": crk_image, # WxHx3
                "crk_mask": crk_mask/255, # WxH
                "non_crk": noncrk_image,
                "bg_region": bg_inp_arr, # WxHx3
                "fg_region": fg_inp_arr, # WxHx3
                "comp_inp": comp_inp, # WxHx3
                  }
        
        
        if self.transform:
            if self.syn_method:
                sample_trans = self.transform(image=samples['comp_inp'], mask=samples["crk_mask"])
            else:
                sample_trans = self.transform(image=samples['crk_image'], mask=samples["crk_mask"])
            # image, mask = sample_trans['image'], sample_trans['mask']
            
            if self.preprocessing:
                sample_pre = self.preprocessing(image=sample_trans['image'], mask=sample_trans['mask'])
        
        elif self.syn_method:
            sample_pre = self.preprocessing(image=samples['comp_inp'], mask=samples["crk_mask"])
        
        else:
            sample_pre = self.preprocessing(image=samples['crk_image'], mask=samples["crk_mask"])
        
        samples['image'] = sample_pre['image']
        samples['mask'] = sample_pre['mask']
        samples['idx'] = idx
        samples['name'] = src_case.split('/')[-1].split('.')[0] + '_and_' + trg_case.split('/')[-1].split('.')[0]
        
        return samples
    

####################################################################
#### Dataset for Dove method #######################################
####################################################################

class Train_Dove_DataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        train_src_filenames = None,
        train_trg_filenames = None,
        syn_method = None,
        preprocessing = None,
        transform=None,
        dove_preprocess = None,
        dove_load_size = None,
        dove_crop_size = None, 
        dove_no_flip = None,
        shuffle = False
        # dove_model = None
    ):
        self._base_dir = base_dir
        self.train_src_filenames = train_src_filenames
        self.train_trg_filenames = train_trg_filenames
        self.syn_method = syn_method
        self.src_sample_list = []
        self.len_src_samples = None
        self.trg_sample_list = []
        self.len_trg_samples = None
        self.src_sample_list_aligned = []
        self.trg_sample_list_aligned = []
        self.len_data = None
        self.preprocessing = preprocessing
        self.transform = transform
        
        self.dove_preprocess = dove_preprocess
        self.dove_load_size = dove_load_size
        self.dove_crop_size = dove_crop_size
        self.dove_no_flip = dove_no_flip
        
        self.shuffle = shuffle
        # self.dove_model = dove_model

        # For trg noncrack image
        with open(self._base_dir + f"/{self.train_trg_filenames}", "r") as f1:
            self.trg_sample_list = f1.readlines()
        self.trg_sample_list = [item.replace("\n", "") for item in self.trg_sample_list]
        self.len_trg_samples = len(self.trg_sample_list)
        print("The number of noncrack samples: {}".format(self.len_trg_samples))
        # For src crack image
        with open(self._base_dir + f"/{self.train_src_filenames}", "r") as f1:
            self.src_sample_list = f1.readlines()
        self.src_sample_list = [item.replace("\n", "") for item in self.src_sample_list]
        self.len_src_samples = len(self.src_sample_list)
        print("The number of crack samples: {}".format(self.len_src_samples))
        
        self.len_data = max(self.len_trg_samples, self.len_src_samples)
        for i in range(self.len_data):
            self.trg_sample_list_aligned.append(self.trg_sample_list[i%len(self.trg_sample_list)])
            self.src_sample_list_aligned.append(self.src_sample_list[i%len(self.src_sample_list)])

        print("Total {} noncrack samples".format(len(self.trg_sample_list_aligned)))
        print("Total {} crack samples".format(len(self.src_sample_list_aligned)))
        
    
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        if self.shuffle:
            random.shuffle(self.src_sample_list_aligned)
        src_case = self.src_sample_list_aligned[idx]
        crk_image = np.array(Image.open(self._base_dir + "/{}".format(src_case)))
        crk_mask = np.array(Image.open(self._base_dir + "/{}".format(src_case.replace('images','masks').replace('.jpg','.png'))).convert('L'))
        crk_mask_dialated = Image.open(self._base_dir + "/{}".format(src_case.replace('images','masks_dilated').replace('.jpg','.png')))
        crk_mask_dialated_arr = np.array(crk_mask_dialated) / 255 # to cover crack region
        crk_mask_dialated_inv_arr = 1 - crk_mask_dialated_arr # to get background region
        
        trg_case = self.trg_sample_list_aligned[idx]
        noncrk_image = np.array(Image.open(self._base_dir + "/{}".format(trg_case)))
        
        fg_inp_arr = (crk_image * crk_mask_dialated_arr).astype(np.uint8)# only ger crack region for input
        bg_inp_arr = (noncrk_image * crk_mask_dialated_inv_arr).astype(np.uint8)
        comp_inp_ = Image.fromarray(fg_inp_arr+bg_inp_arr)
        
        # mask = (mask/255)
        
        
        dove_transform_params = get_params(self.dove_preprocess, self.dove_load_size, self.dove_crop_size, comp_inp_.size)
        comp_inp_transform = get_transform(self.dove_preprocess, self.dove_load_size, self.dove_crop_size, self.dove_no_flip, dove_transform_params)
        msk_inp = TF.to_tensor(crk_mask_dialated.convert('1')).unsqueeze(0)
        comp_inp = comp_inp_transform(comp_inp_).unsqueeze(0)
        
        samples = {
                "crk_image": crk_image, # WxHx3
                "crk_mask": crk_mask/255, # WxH
                "crk_mask_dialated": np.array(crk_mask_dialated), # WxHx3
                "noncrk_image": noncrk_image, # WxHx3
                "comp_inp": np.array(comp_inp_), # 1x3xWxH
                "msk_inp": msk_inp # 1x1xWxH
                  }
        
        
        if self.syn_method:
            with torch.no_grad():
                inputs = torch.cat([comp_inp, msk_inp], 1)
                inputs = Variable(inputs.cuda(non_blocking=True))
                syn_crk_image = self.syn_method(inputs)
                syn_crk_image = (syn_crk_image[0].cpu().data.numpy().transpose(1, 2, 0) + 1)/2
                syn_crk_image = (syn_crk_image*255).astype(np.uint8)
                samples['syn_crk_image'] = syn_crk_image
            # print('Successfully synthesized image!')
        
        if self.transform:
            if self.syn_method:
                sample_trans = self.transform(image=samples['syn_crk_image'], mask=samples["crk_mask"])
            else:
                sample_trans = self.transform(image=samples['crk_image'], mask=samples["crk_mask"])
            # image, mask = sample_trans['image'], sample_trans['mask']
            
            if self.preprocessing:
                sample_pre = self.preprocessing(image=sample_trans['image'], mask=sample_trans['mask'])
        
        elif self.syn_method:
            sample_pre = self.preprocessing(image=samples['syn_crk_image'], mask=samples["crk_mask"])
        
        else:
            sample_pre = self.preprocessing(image=samples['crk_image'], mask=samples["crk_mask"])
        
        samples['image'] = sample_pre['image']
        samples['mask'] = sample_pre['mask']
        samples['idx'] = idx
        samples['name'] = src_case.split('/')[-1].split('.')[0] + '_and_' + trg_case.split('/')[-1].split('.')[0]
        
        return samples
    
####################################################################
#### Dataset for FPIE method #######################################
####################################################################    

class Train_FPIE_DataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        train_src_filenames = None,
        train_trg_filenames = None,
        syn_method = None,
        proc = None,
        properties = None, # a dic of properties for syn data
        preprocessing = None,
        transform=None,
        shuffle = False
    ):
        self._base_dir = base_dir
        self.train_src_filenames = train_src_filenames
        self.train_trg_filenames = train_trg_filenames
        self.syn_method = syn_method
        self.proc = proc
        self.properties = properties
        self.src_sample_list = []
        self.len_src_samples = None
        self.trg_sample_list = []
        self.len_trg_samples = None
        self.src_sample_list_aligned = []
        self.trg_sample_list_aligned = []
        self.len_data = None
        self.preprocessing = preprocessing
        self.transform = transform
        self.shuffle = shuffle

        # For trg noncrack image
        with open(self._base_dir + f"/{self.train_trg_filenames}", "r") as f1:
            self.trg_sample_list = f1.readlines()
        self.trg_sample_list = [item.replace("\n", "") for item in self.trg_sample_list]
        self.len_trg_samples = len(self.trg_sample_list)
        print("The number of noncrack samples: {}".format(self.len_trg_samples))
        # For src crack image
        with open(self._base_dir + f"/{self.train_src_filenames}", "r") as f1:
            self.src_sample_list = f1.readlines()
        self.src_sample_list = [item.replace("\n", "") for item in self.src_sample_list]
        self.len_src_samples = len(self.src_sample_list)
        print("The number of crack samples: {}".format(self.len_src_samples))
        
        self.len_data = max(self.len_trg_samples, self.len_src_samples)
        for i in range(self.len_data):
            self.trg_sample_list_aligned.append(self.trg_sample_list[i%len(self.trg_sample_list)])
            self.src_sample_list_aligned.append(self.src_sample_list[i%len(self.src_sample_list)])

        print("Total {} noncrack samples".format(len(self.trg_sample_list_aligned)))
        print("Total {} crack samples".format(len(self.src_sample_list_aligned)))
        
    
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        if self.shuffle:
            random.shuffle(self.src_sample_list_aligned)
        src_case = self.src_sample_list_aligned[idx]
        crk_image = np.array(Image.open(self._base_dir + "/{}".format(src_case)))
        crk_mask = np.array(Image.open(self._base_dir + "/{}".format(src_case.replace('images','masks').replace('.jpg','.png'))).convert('L'))
        crk_mask_dialated = np.array(Image.open(self._base_dir + "/{}".format(src_case.replace('images','masks_dilated').replace('.jpg','.png'))))
        
        trg_case = self.trg_sample_list_aligned[idx]
        noncrk_image = np.array(Image.open(self._base_dir + "/{}".format(trg_case)))
        
        # mask = (mask/255)
        samples = {
                "crk_image": crk_image, # WxHx3
                "crk_mask": crk_mask/255, # WxH
                "crk_mask_dialated": crk_mask_dialated, # WxHx3
                "noncrk_image": noncrk_image # WxHx3
                  }
        if self.syn_method:
            src = samples["crk_image"]
            mask = samples["crk_mask_dialated"]
            tgt = samples["noncrk_image"]

            src = cv2.copyMakeBorder(src, 9, 9, 9, 9, cv2.BORDER_CONSTANT, None, value = 0)
            mask = cv2.copyMakeBorder(mask, 9, 9, 9, 9, cv2.BORDER_CONSTANT, None, value = 0)
            tgt = cv2.copyMakeBorder(tgt, 9, 9, 9, 9, cv2.BORDER_CONSTANT, None, value = 0)
            
            if mask.sum() == 0:
                syn_result = tgt[9:9+self.properties['size'], 9:9+self.properties['size']]
            else:
                n = self.proc.reset(src, mask, tgt, 
                                    (self.properties['h0'], self.properties['w0']), 
                                    (self.properties['h1'], self.properties['w1']))
                self.proc.sync()
                
                if self.proc.root:
                    result = tgt
                if self.properties['p'] == 0:
                    self.properties['p'] = self.properties['n']
                    
                for i in range(0, self.properties['n'], self.properties['p']):
                    if self.proc.root:
                        result, err = self.proc.step(self.properties['p'])  # type: ignore
                        # print(f"Iter {i + config['p']}, abs error {err}")
                        if i + self.properties['p'] < self.properties['n']:
                            syn_result = result
                            # write_image(f"iter{i + config['p']:05d}.png", result)
                    else:
                        self.proc.step(self.properties['p'])
                if self.proc.root:
                    syn_result = result[9:9+self.properties['size'], 9:9+self.properties['size']]
                    
            samples['syn_crk_image'] = syn_result

        
        if self.transform:
            if self.syn_method:
                sample_trans = self.transform(image=samples['syn_crk_image'], mask=samples["crk_mask"])
            else:
                sample_trans = self.transform(image=samples['crk_image'], mask=samples["crk_mask"])
            # image, mask = sample_trans['image'], sample_trans['mask']
            
            if self.preprocessing:
                sample_pre = self.preprocessing(image=sample_trans['image'], mask=sample_trans['mask'])
        
        elif self.syn_method:
            sample_pre = self.preprocessing(image=samples['syn_crk_image'], mask=samples["crk_mask"])
        
        else:
            sample_pre = self.preprocessing(image=samples['crk_image'], mask=samples["crk_mask"])
        
        samples['image'] = sample_pre['image']
        samples['mask'] = sample_pre['mask']
        samples['idx'] = idx
        samples['name'] = src_case.split('/')[-1].split('.')[0] + '_and_' + trg_case.split('/')[-1].split('.')[0]
        
        return samples

class Train_Trg_DataSets(Dataset):
    def __init__(
        self,
        num_src_samples=1000,
        base_dir=None,
        train_filenames = None,
        num=None,
        preprocessing = None,
        transform=None,
    ):
        self.num_src_samples = num_src_samples
        self._base_dir = base_dir
        self.train_filenames = train_filenames
        self.sample_list = []
        self.trg_sample_list = []
        self.preprocessing = preprocessing
        self.transform = transform

        with open(self._base_dir + f"/{self.train_filenames}", "r") as f1:
            self.trg_sample_list = f1.readlines()
        self.trg_sample_list = [item.replace("\n", "") for item in self.trg_sample_list]

        for i in range(self.num_src_samples):
            self.sample_list.append(self.trg_sample_list[i%len(self.trg_sample_list)])

        print("Total {} samples on the target after alignment".format(len(self.sample_list)))
    
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
        val_syn_filenames = None,
        val_trg_filenames = None,
        preprocessing = None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.val_syn_filenames = val_syn_filenames
        self.val_trg_filenames = val_trg_filenames
        self.sample_list = []
        self.sample_list_syn = []
        self.sample_list_trg = []
        self.preprocessing = preprocessing
        self.transform = transform
        
        # For src
        if self.val_syn_filenames:
            with open(self._base_dir + f"/{self.val_syn_filenames}", "r") as f1:
                self.sample_list_syn = f1.readlines()
            self.sample_list_syn = [item.replace("\n", "") for item in self.sample_list_syn]
            
        # For trg
        if self.val_trg_filenames:
            with open(self._base_dir + f"/{self.val_trg_filenames}", "r") as f1:
                self.sample_list_trg = f1.readlines()
            self.sample_list_trg = [item.replace("\n", "") for item in self.sample_list_trg]
        
        self.sample_list = self.sample_list_syn + self.sample_list_trg
        print("Total {} valid samples of source labeled data".format(len(self.sample_list_syn)))
        print("Total {} valid samples of target labeled data".format(len(self.sample_list_trg)))

        print("Total {} samples used for validating".format(len(self.sample_list)))
    
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
    
# Difine syn_method that used to synthesizing data during the training process

if args.syn_concrete_data:
    print("Synthesizing concrete crack data...!")
elif args.syn_pav_data:
    print("Synthesizing pavment crack data...!")
else:
    print("Using source data for training...!")

if args.syn_method == 'cutpaste':
    syn_method = args.syn_method
    print(f"Start synthesizing image using {args.syn_method}...!")

elif args.syn_method == 'fpie':
    syn_method = args.syn_method
    print(f"Start synthesizing image using {args.syn_method}...!")

elif args.syn_method == 'dove':
    # syn_method = crack_datasets_STDASeg.DoveNet_syn_method
    syn_method = Dove_G.eval()
    print(f"Start synthesizing image using {args.syn_method}...!")
        
else:
    syn_method = 'uda'
    print(f"Source data is directly used for training STDASeg in the UDA manner...!")
    
    
#### Data loader###########################\

# Define transform on source domain
if args.train_syn_transform:
    train_syn_transform = get_syn_training_augmentation()
    print('Train syn data will be transformed','\n')
else:
    train_syn_transform = None
    print('Train data will not be transformed','\n')

# Define transform on target domain
if args.train_trg_transform:
    train_trg_transform = get_trg_training_augmentation()
    print('Train trg data will be transformed','\n')
else:
    train_trg_transform = None
    print('Train trg data will not be transformed','\n')

if args.val_transform:
    val_transforms = get_trg_training_augmentation()
    print('Valid data will be transformed','\n')
else:
    val_transforms = None
    print('Valid data will not be transformed','\n')

# Dataset with respect to each domain mixing strategy

if args.syn_method == 'cutpaste':
    src_train_data = Train_CP_DataSets(
                                            base_dir=args.root_path,
                                            train_src_filenames = args.train_syn_src_filenames,
                                            train_trg_filenames = args.train_syn_trg_filenames,
                                            syn_method = syn_method,
                                            preprocessing=get_preprocessing(preprocessing_fn),
                                            transform = train_syn_transform,
                                            shuffle = False
                                        )
    src_trainloader = DataLoader(
                                            src_train_data,
                                            batch_size=args.batch_size - args.labeled_bs, 
                                            shuffle=True, 
                                            num_workers=6, 
                                            pin_memory=True,
                                            # worker_init_fn=worker_init_fn
                                )
    
elif args.syn_method == 'fpie':
    src_train_data = Train_FPIE_DataSets(
                                            base_dir=args.root_path,
                                            train_src_filenames = args.train_syn_src_filenames,
                                            train_trg_filenames = args.train_syn_trg_filenames,
                                            syn_method = args.syn_method,
                                            proc = proc,
                                            properties = properties, # a dic of properties for syn data
                                            preprocessing=get_preprocessing(preprocessing_fn),
                                            transform = train_syn_transform,
                                            shuffle = False
                                        )
    src_trainloader = DataLoader(
                                            src_train_data,
                                            batch_size=args.batch_size - args.labeled_bs, 
                                            shuffle=True, 
                                            num_workers=6, 
                                            pin_memory=True,
                                            # worker_init_fn=worker_init_fn
                                )

elif args.syn_method == 'dove':
    src_train_data = Train_Dove_DataSets(
                                            base_dir=args.root_path,
                                            train_src_filenames = args.train_syn_src_filenames,
                                            train_trg_filenames = args.train_syn_trg_filenames,
                                            syn_method = syn_method,
                                            preprocessing=get_preprocessing(preprocessing_fn),
                                            transform = train_syn_transform,
                                            dove_preprocess = config['preprocess'],
                                            dove_load_size = config['crop_size'],
                                            dove_crop_size = config['crop_size'], 
                                            dove_no_flip = True,
                                            shuffle = False
                                        )
    src_trainloader = DataLoader(
                                            src_train_data,
                                            batch_size=args.batch_size - args.labeled_bs, 
                                            shuffle=True, 
                                            # num_workers=6, 
                                            pin_memory=True,
                                            # worker_init_fn=worker_init_fn
                                )
    
else:
    src_sample_list = []
    with open(args.root_path + f"/{args.train_syn_src_filenames}", "r") as f1:
        src_sample_list = f1.readlines()
    src_sample_list = [item.replace("\n", "") for item in src_sample_list]

    print("Total {} samples from the source".format(len(src_sample_list)))

    trg_sample_list = []
    with open(args.root_path + f"/{args.train_trg_filenames}", "r") as f1:
        trg_sample_list = f1.readlines()
    trg_sample_list = [item.replace("\n", "") for item in trg_sample_list]

    print("Total {} samples from the target".format(len(trg_sample_list)))

    max_samples = max(len(src_sample_list), len(trg_sample_list))
    print("Total {} samples for training".format(max_samples))

    src_train_data = Train_Src_DataSets(
                                            max_samples = max_samples,
                                            base_dir = args.root_path,
                                            train_filenames = args.train_syn_src_filenames,
                                            preprocessing=get_preprocessing(preprocessing_fn),
                                            transform = train_syn_transform
                                        )
    src_trainloader = DataLoader(
                                            src_train_data,
                                            batch_size=args.batch_size - args.labeled_bs, 
                                            shuffle=True, 
                                            num_workers=6, 
                                            pin_memory=True,
                                            # worker_init_fn=worker_init_fn
                                )
    
# print(len(src_trainloader))
# src_trainloader_iter = enumerate(src_trainloader)
# _, src_data = src_trainloader_iter.__next__()
# print(f"src_train_batch_data_shape: {src_data['image'].shape}")


num_src_samples = len(src_train_data)

trg_train_data = Train_Trg_DataSets(
                                            num_src_samples = num_src_samples,
                                            base_dir = args.root_path,
                                            train_filenames = args.train_trg_filenames,
                                            preprocessing=get_preprocessing(preprocessing_fn),
                                            transform = train_trg_transform
                                        )

trg_trainloader = DataLoader(
                                            trg_train_data,
                                            batch_size=args.batch_size - args.labeled_bs, 
                                            shuffle=True, 
                                            # num_workers=6, 
                                            pin_memory=True,
                                            # worker_init_fn=worker_init_fn
                                )

# print(len(trg_trainloader))
# trg_trainloader_iter = enumerate(trg_trainloader)
# _, trg_data = trg_trainloader_iter.__next__()
# trg_data['image'].shape
# print(f"trg_train_batch_data_shape: {trg_data['image'].shape}")

if args.valid_on_trg_domain:
    print("Data for validating is from targert domain.")
    val_data = Val_DataSets(
                            base_dir=args.root_path,
                            # val_syn_filenames = args.val_syn_filenames,
                            val_syn_filenames = None,
                            val_trg_filenames = args.val_trg_filenames,
                            preprocessing=get_preprocessing(preprocessing_fn),
                            transform = val_transforms
                                    )
    
    valloader = DataLoader(
                            val_data,
                            batch_size=args.val_batch_size, 
                            shuffle=False, 
                            # num_workers=4, 
                            pin_memory=True,
                            # worker_init_fn=worker_init_fn
                )

else:
    print("Please set valid_on_trg_domain == True!!!")
    val_data = Val_DataSets(
                            base_dir=args.root_path,
                            # val_syn_filenames = args.val_syn_filenames,
                            val_syn_filenames = None,
                            val_trg_filenames = args.val_trg_filenames,
                            preprocessing=get_preprocessing(preprocessing_fn),
                            transform = val_transforms
                                    )
    
    valloader = DataLoader(
                            val_data,
                            batch_size=args.val_batch_size, 
                            shuffle=False, 
                            # num_workers=4, 
                            pin_memory=True,
                            # worker_init_fn=worker_init_fn
                )

# valloader_iter = enumerate(valloader)
# _, trg_valdata = valloader_iter.__next__()
# trg_valdata['image'].shape
# print(f"trg_val_batch_data_shape: {trg_valdata['image'].shape}")




