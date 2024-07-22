import sys
sys.path.insert(0, '/data/gpfs/projects/punim1699/data_dunguyen/Codes_github/SSL4MIS/code/modules_models/')
import time
from networks.efficientunet import Effi_UNet
from networks.enet import ENet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_mod, UNet_DS, UNet_URPC, UNet_CCT
from networks.STRNet import STRNet, CTCNet
# from networks.SemiCrack import CCTNet
import argparse
from networks.vision_transformer import SwinUnet as SwinU
from networks.config import get_config
from networks.nnunet import initialize_network
import numpy as np
from networks.vit_seg_modeling import VisionTransformer as TransU
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

import modules_models
from modules_models.Unet_SeResnext_model import Unet_SeResnext
from modules_models.Unet_Vgg_model import Unet_Vgg
from modules_models.Unet_EfficicentNet_model import Unet_EfficientNet
from modules_models.Unet_MobileNet_model import Unet_mobilenet_v2
from modules_models.Unet_MixTransformer_model import Unet_SegFormer
from modules_models.DeeplabV3_SeResnext_model import DeepLabV3Plus_SeResnext
from modules_models.Linknet_SeResnext_model import Linknet
from modules_models.Swinv2_Unet import swin_unet



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Supervision_CNN_Trans2D', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[512, 512],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# For ViT
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

args = parser.parse_args("")
# args = parser.parse_args()
config = get_config(args)


# def net_factory(net_type="unet", in_chns=1, class_num=3):
#     if net_type == "unet":
#         net = UNet(in_chns=in_chns, class_num=class_num).cuda()
#     if net_type == "unet_mod":
#         net = UNet_mod(in_chns=in_chns, class_num=class_num).cuda()
#     elif net_type == "enet":
#         net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
#     elif net_type == "unet_ds":
#         net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
#     elif net_type == "unet_cct":
#         net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
#     elif net_type == "unet_urpc":
#         net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
#     elif net_type == "efficient_unet":
#         net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
#                         in_channels=in_chns, classes=class_num).cuda()
#     elif net_type == "ViT_Seg":
#         net = ViT_seg(config, img_size=args.patch_size,
#                       num_classes=args.num_classes).cuda()
#     elif net_type == "pnet":
#         net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
#     elif net_type == "nnUNet":
#         net = initialize_network(num_classes=class_num).cuda()
#     else:
#         net = None
#     return net


def net_factory(net_type="unet", in_chns=1, class_num=2, weights = None):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
        
    elif net_type == "unet_mod":
        print(f'Creating {net_type} model...!')
        net = UNet_mod(in_chns=in_chns, class_num=class_num).cuda()
        
    elif net_type == "seresnext_unet":
        print(f'Creating {net_type} model...!')
        net = Unet_SeResnext(encoder_name='se_resnext50_32x4d', encoder_weights = weights, in_channels=in_chns, classes=class_num).cuda()
        
    elif net_type == "vgg_unet":
        print(f'Creating {net_type} model...!')
        net = Unet_Vgg(encoder_name='vgg19', encoder_weights = weights, in_channels=in_chns, classes=class_num).cuda()
        
    elif net_type == "efficient_unet":
        print(f'Creating {net_type} model...!')
        net = Unet_EfficientNet(encoder_name='efficientnet-b5', encoder_weights = weights, in_channels=in_chns, classes=class_num).cuda()
        
    elif net_type == "mobi_unet":
        print(f'Creating {net_type} model...!')
        net = Unet_mobilenet_v2(encoder_name='mobilenet_v2', encoder_weights = weights, in_channels=in_chns, classes=class_num).cuda()
        
    elif net_type == "deeplabv3":
        print(f'Creating {net_type} model...!')
        net = DeepLabV3Plus_SeResnext(encoder_name='se_resnext50_32x4d', encoder_weights = weights, in_channels=in_chns, classes=class_num).cuda()
        
    elif net_type == "linknet":
        print(f'Creating {net_type} model...!')
        net = DeepLabV3Plus_SeResnext(encoder_name='se_resnext50_32x4d', encoder_weights = weights, in_channels=in_chns, classes=class_num).cuda()
        
    elif net_type == "enet":
        net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()

    # elif net_type == "efficient_unet":
    #     net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
    #                     in_channels=in_chns, classes=class_num).cuda()
        
    # elif net_type == "SwinU":
    #     print(f'Creating {net_type} model...!')
    #     net = SwinU(config, img_size=args.patch_size[0],
    #                   num_classes=args.num_classes).cuda()
    #     net.load_from(config)
    elif net_type == "ctcn":
        print(f'Creating {net_type} model...!')
        net = CTCNet().cuda()
        
    elif net_type == "SwinU":
        print(f'Creating {net_type} model...!')
        if args.patch_size[0] == 256:
            net = swin_unet(size="swinv2_small_window8_256", img_size=args.patch_size[0]).cuda()
        elif args.patch_size[0] == 512:
            net = swin_unet(size="swinv2_base_window16_256", img_size=args.patch_size[0]).cuda()
        
    elif net_type == "TransU":
        print(f'Creating {net_type} model...!')
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.patch_size[0] / args.vit_patches_size), int(args.patch_size[0] / args.vit_patches_size))
        net = TransU(config_vit, img_size=args.patch_size[0], num_classes=config_vit.n_classes).cuda()
        # net.load_from(weights=np.load(config_vit.pretrained_path))
        
    elif net_type == "segformer_unet":
        print(f'Creating {net_type} model...!')
        net = Unet_SegFormer(encoder_name='mit_b5', encoder_weights = None, in_channels=in_chns, classes=class_num).cuda()
        
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    else:
        net = None
    return net
