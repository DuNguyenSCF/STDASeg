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
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
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

from tensorboardX import SummaryWriter
import time
from tqdm import tqdm

# from dataloaders import utils
from utils import args_main_STDASeg, helper_functions, losses, metrics, ramps
from dataloaders import crack_datasets_STDASeg
from dataloaders.crack_datasets_STDASeg import src_trainloader,  trg_trainloader, valloader

from networks.net_factory import net_factory
from modules_models.Swinv2_Unet import swin_unet
from networks.vit_seg_modeling import VisionTransformer as TransU
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.SemiCrack import projectors, classifier
from networks.discriminator import FCDiscriminator



from utils.helper_functions import calculate_metric_percase_mod, test_single_volume_mod
from utils.losses import IW_MaxSquareloss, discrepancy_calc



# arg_parser = argparse.ArgumentParser()
# arg_parser = args_main_STDASeg_training.add_train_STDASeg_args(arg_parser)
# args = arg_parser.parse_args()

args = args_main_STDASeg.initialize_STDASeg_train_args()

def create_model(args, type_model):
    
    if type_model == 'TransU':
        print(f'Creating {type_model} model...!')
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.patch_size[0] / args.vit_patches_size), int(args.patch_size[0] / args.vit_patches_size))
            model = TransU(config_vit, img_size=args.patch_size[0], num_classes=config_vit.n_classes).cuda()
            model.load_from(weights=np.load(config_vit.pretrained_path))
    elif type_model == 'SwinU':
        print(f'Creating {type_model} model...!')
        if args.patch_size[0] == 256:
            model = swin_unet(size="swinv2_small_window8_256", img_size=args.patch_size[0]).cuda()
        elif args.patch_size[0] == 512:
            model = swin_unet(size="swinv2_base_window16_256", img_size=args.patch_size[0]).cuda()
    elif type_model == 'unet_mod':
        print(f'Creating {type_model} model...!')
        model = net_factory(net_type=type_model, in_chns=3, class_num=args.num_classes, weights = None)
    else: # for CNN models with pre-trained weights for encoder
        if args.load_pretrained_weights:
            model = net_factory(net_type=type_model, in_chns=3, class_num=args.num_classes, weights=args.encoder_weights)
        else:
            model = net_factory(net_type=type_model, in_chns=3, class_num=args.num_classes, weights = None)
            
    return model

# args.load_pretrained_weights = True
# model = create_model(args, args.model)
        
#########################################################################
#######################  Loading models and training  ###################
#########################################################################
model1 = create_model(args, args.cnn_model)
model2 = create_model(args, args.trans_model)

if args.checkpoint_1.endswith('.pth'):
    snapshot_path_1 = "../model/{}".format(args.checkpoint_1_base_dir)
    checkpoint_path_1 = os.path.join(snapshot_path_1,args.checkpoint_1)
    model1.load_state_dict(torch.load(checkpoint_path_1))
    print('Loading checkpoint at epoch 50th for model 1...!')
else:
    # model1 = xavier_normal_init_weight(model1)
    print('Using pre-trained weights for model 1...!')

if args.checkpoint_2.endswith('.pth'):
    snapshot_path_2 = "../model/{}".format(args.checkpoint_2_base_dir)
    checkpoint_path_2 = os.path.join(snapshot_path_2,args.checkpoint_2)
    model2.load_state_dict(torch.load(checkpoint_path_2))
    print('Loading checkpoint at epoch 50th for model 2...!')
else:
    # model2 = kaiming_normal_init_weight(model2)
    print('Using pre-trained weights for model 2...!')
print('-'*100)
print('Loading datasets................!', '\n')



def train_STDASeg(args, snapshot_path, model1, model2, src_trainloader,  trg_trainloader, valloader):

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
    def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

    
    # Specify threshold to generate pseudo-labels
    if args.thresholded_pseudo:
        print('Pseudo labels with be thresholded with a median value','\n')
    else:
        print('Pseudo labels will be thresholded at 0.5','\n')
        
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


    model1.train()
    model2.train()

    # optimizers
    optimizer1 = optim.SGD(model1.parameters(), lr=args.base_lr_1,
                          momentum=0.9, weight_decay=args.weight_decay_1)
    optimizer2 = optim.SGD(model2.parameters(), lr=args.base_lr_2,
                          momentum=0.9, weight_decay=args.weight_decay_2)
    
    # loss functions
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)
    iw_msl = IW_MaxSquareloss(num_class=args.num_classes, ratio=args.relaxed_weight_factor)


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(src_trainloader)))

    iter_num = 0
    max_epoch = args.max_iterations // len(src_trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=100)
    for epoch_num in iterator:
        for (i_batch, src_sampled_batch), (i_trg_batch, trg_sampled_batch) in zip(enumerate(src_trainloader), enumerate(trg_trainloader)):
            i_batch, src_sampled_batch = (i_batch, src_sampled_batch)
            i_trg_batch, trg_sampled_batch = (i_trg_batch, trg_sampled_batch)

            src_img_batch, src_label_batch = src_sampled_batch['image'], src_sampled_batch['mask']
            src_img_batch, src_label_batch = src_img_batch.cuda(), src_label_batch.cuda()

            trg_img_batch, trg_label_batch = trg_sampled_batch['image'], trg_sampled_batch['mask']
            trg_img_batch, trg_label_batch = trg_img_batch.cuda(), trg_label_batch.cuda()

            iter_num = iter_num + 1
            
            if args.is_stage_1:
                print('Stage 1 is involved during training.')

                # First update on model 1, dont accumulate grads in model 2
                for param in model1.parameters():
                    param.requires_grad = True
                for param in model2.parameters():
                    param.requires_grad = False

                # Train model1 using src labeled data
                src_outputs1  = model1(src_img_batch)
                src_outputs_soft1 = torch.softmax(src_outputs1, dim=1)
                # Supervised loss for model1
                loss1 = 0.5*(ce_loss(src_outputs1, src_label_batch.long()) + dice_loss(
                        src_outputs_soft1, src_label_batch.unsqueeze(1)))

                # Train model1 using trg unlabeled data
                trg_outputs1  = model1(trg_img_batch)
                trg_outputs_soft1 = torch.softmax(trg_outputs1, dim=1)

                # For entropy minimization with prior class weights
                pred_1 = trg_outputs1
                prob_1 = trg_outputs_soft1
                iw_msl_1 = iw_msl(pred_1, prob_1)

                model1_loss = loss1 + args.ent_weight*iw_msl_1

                optimizer1.zero_grad()
                model1_loss.backward() # call gradient
                optimizer1.step() # update trainable weights

                logging.info('iteration %d - first update: model1 ce : %f model1 ent : %f --> model1 loss : %f' 
                             % (iter_num, loss1.item(), iw_msl_1.item(), model1_loss.item()))

                # First update on model 2, dont accumulate grads in model 1
                for param in model1.parameters():
                    param.requires_grad = False
                for param in model2.parameters():
                    param.requires_grad = True

                # Train model2 using src labeled data
                src_outputs2  = model2(src_img_batch)
                src_outputs_soft2 = torch.softmax(src_outputs2, dim=1)
                # Supervised loss for model2
                loss2 = 0.5*(ce_loss(src_outputs2, src_label_batch.long()) + dice_loss(
                        src_outputs_soft2, src_label_batch.unsqueeze(1)))

                # Train model2 using trg unlabeled data
                trg_outputs2  = model2(trg_img_batch)
                trg_outputs_soft2 = torch.softmax(trg_outputs2, dim=1)
                pred_2 = trg_outputs2
                prob_2 = trg_outputs_soft2
                iw_msl_2 = iw_msl(pred_2, prob_2)

                model2_loss = loss2 + args.ent_weight*iw_msl_2

                optimizer2.zero_grad()
                model2_loss.backward() # call gradient
                optimizer2.step() # update trainable weights

                logging.info('iteration %d - first update: model2 ce : %f model2 ent : %f --> model2 loss : %f' 
                             % (iter_num, loss2.item(), iw_msl_2.item(), model2_loss.item()))

            # Second update using Cross pseudo supervision
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            for param in model1.parameters():
                param.requires_grad = True
            for param in model2.parameters():
                param.requires_grad = True

            # Train model1 using src labeled data
            src_outputs1  = model1(src_img_batch)
            src_outputs_soft1 = torch.softmax(src_outputs1, dim=1)
            # Supervised loss for model1
            loss1 = 0.5 * (ce_loss(src_outputs1, src_label_batch.long()) + dice_loss(
                    src_outputs_soft1, src_label_batch.unsqueeze(1)))

            # Train model2 using src labeled data
            src_outputs2  = model2(src_img_batch)
            src_outputs_soft2 = torch.softmax(src_outputs2, dim=1)
            # Supervised loss for model2
            loss2 = 0.5 * (ce_loss(src_outputs2, src_label_batch.long()) + dice_loss(
                    src_outputs_soft2, src_label_batch.unsqueeze(1)))


            # Train model1 using trg unlabeled data
            trg_outputs1  = model1(trg_img_batch)
            trg_outputs_soft1 = torch.softmax(trg_outputs1, dim=1)

            # Train model2 using trg unlabeled data
            trg_outputs2  = model2(trg_img_batch)
            trg_outputs_soft2 = torch.softmax(trg_outputs2, dim=1)

            # cdd loss
            loss_cdd = 0.5*(discrepancy_calc(trg_outputs_soft1, trg_outputs_soft2) + discrepancy_calc(src_outputs_soft1, src_outputs_soft2))

            pseudo_outputs1 = torch.argmax(trg_outputs_soft1.detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(trg_outputs_soft2.detach(), dim=1, keepdim=False)

            # Cross teaching using hard pseudo label (at fixed threshold = 0.5)
            h_pseudo_label1_thr = pseudo_outputs2
            h_pseudo_label2_thr = pseudo_outputs1
            
            
            # Save samples of pseudo labels
            # if iter_num > 0 and iter_num % args.valid_checkpoint == 0: #
            if args.save_pseudo and iter_num in [100, 1000, 5000, 10000, 20000]: # 
                print("Saving pseudo labels to check...!")
                
                pl_path = f"{snapshot_path}/pseudo_labels_{iter_num}"
                if not os.path.exists(pl_path):
                        os.makedirs(pl_path)

                for i_val_batch, val_sampled_batch in enumerate(valloader):
                    val_img_batch, val_label_batch, name = val_sampled_batch['image'], val_sampled_batch['mask'], val_sampled_batch['name']
                    val_img_batch, val_label_batch = val_img_batch.cuda(), val_label_batch.cuda()

                    # Train model1 using trg unlabeled data
                    val_outputs1  = model1(val_img_batch)
                    val_outputs_soft1 = torch.softmax(val_outputs1, dim=1)

                    # Train model2 using trg unlabeled data
                    val_outputs2  = model2(val_img_batch)
                    val_outputs_soft2 = torch.softmax(val_outputs2, dim=1)
                    
                    # Cross teaching using hard pseudo label (at fixed threshold = 0.5)

                    val_pseudo_outputs1 = torch.argmax(val_outputs_soft1.detach(), dim=1, keepdim=False)
                    val_pseudo_outputs2 = torch.argmax(val_outputs_soft2.detach(), dim=1, keepdim=False)

                    val_h_pseudo_label1_thr = val_pseudo_outputs2
                    val_h_pseudo_label2_thr = val_pseudo_outputs1

                    img_titles = ['img', 'gt', 'Model2_pseudo_outputs', 'h_pseudo_label1_thr', 'Model1_pseudo_outputs1', 'h_pseudo_label2_thr']

                    fig, axs = plt.subplots(len(val_h_pseudo_label1_thr),6,figsize=(30,5), squeeze=True)

                    for i in range(len(val_h_pseudo_label1_thr)):
                        axs[0].imshow(torch.movedim(denormalize(val_img_batch.cpu()[i]),0,-1))
                        axs[1].imshow(val_label_batch.cpu()[i])
                        # axs[2].imshow(pseudo_outputs2[i].cpu())
                        axs[2].imshow(val_outputs_soft2.detach().cpu()[i][1])
                        axs[3].imshow(val_h_pseudo_label1_thr[i].cpu())
                        # axs[4].imshow(pseudo_outputs1[i].cpu())
                        axs[4].imshow(val_outputs_soft1.detach().cpu()[i][1])
                        axs[5].imshow(val_h_pseudo_label2_thr[i].cpu())
                    for i in range(0, 6):
                        axs[i].set_axis_off()
                        axs[i].set_title(img_titles[i], loc='center', y=1.05)
                    plt.savefig(f"{pl_path}/pseudo05_{name[0]}.jpg", bbox_inches='tight')
                    plt.close()

            pseudo_supervision1 = ce_loss(trg_outputs1, h_pseudo_label1_thr.long()) + dice_loss(trg_outputs_soft1, h_pseudo_label1_thr.unsqueeze(1))
            pseudo_supervision2 = ce_loss(trg_outputs2, h_pseudo_label2_thr.long()) + dice_loss(trg_outputs_soft2, h_pseudo_label2_thr.unsqueeze(1))

            consistency_weight = get_current_consistency_weight(iter_num // len(src_trainloader))

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss + args.cdd_weight*loss_cdd


            loss.backward()

            optimizer1.step()
            optimizer2.step()


            lr_1 = args.base_lr_1 * (1.0 - iter_num / args.max_iterations) ** 0.9
            lr_2 = args.base_lr_2 * (1.0 - iter_num / args.max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_1
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_2



            writer.add_scalar('lr_1', lr_1, iter_num)
            writer.add_scalar('lr_2', lr_2, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('loss/cdd_loss',
                              loss_cdd, iter_num)
            writer.add_scalar('loss/total_loss',
                              loss, iter_num)
            logging.info('iteration %d - second update: model1 loss : %f model2 loss : %f cdd loss : %f --> total loss : %f \n' 
                         % (iter_num, model1_loss.item(), model2_loss.item(), loss_cdd.item(), loss.item()))

            # if iter_num % 50 == 0:
            if iter_num % 100 == 0:
                image = trg_img_batch[0, :, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    trg_outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[0, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    trg_outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[0, ...] * 50, iter_num)
                labs = trg_label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
            # if iter_num > 0 and iter_num % 200 == 0:
            if iter_num > 0 and iter_num % args.valid_checkpoint == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_mod(
                        sampled_batch["image"], sampled_batch["mask"], model1, classes=args.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(valloader)
                for class_i in range(args.num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                          metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    if args.save_best_ckpt_with_iter:
                        save_mode_path = os.path.join(snapshot_path,
                                                      'model1_iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance1, 4)))
                        torch.save(model1.state_dict(), save_mode_path)
                    
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.cnn_model))
                    torch.save(model1.state_dict(), save_best)
                    print('Saved best checkpoint at iter {}!'.format(iter_num))

                logging.info(
                    'iteration %d : model1_mean_dice : %f' % (iter_num, performance1))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_mod(
                        sampled_batch["image"], sampled_batch["mask"], model2, classes=args.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(valloader)
                for class_i in range(args.num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    if args.save_best_ckpt_with_iter:
                        save_mode_path = os.path.join(snapshot_path,
                                                      'model2_iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance2, 4)))
                        torch.save(model2.state_dict(), save_mode_path)
                    
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.trans_model))
                    torch.save(model2.state_dict(), save_best)
                    
                    print('Saved best checkpoint at iter {}!'.format(iter_num))

                logging.info(
                    'iteration %d : model2_mean_dice : %f' % (iter_num, performance2))
                model2.train()
            # if iter_num % 3000 == 0:
            if args.save_ckpt_every_5000_inters and iter_num % 5000 == 0: # save model every 20 epochs
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= args.max_iterations:
                    break
            time1 = time.time()
        if iter_num >= args.max_iterations:
            iterator.close()
            break
    writer.close()
    
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

    snapshot_path = "../model/{}_{}/{}_{}".format(
        args.exp, args.labeled_num, args.cnn_model, args.trans_model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    pseudo_path = f"{snapshot_path}/pseudo_labels"
    if not os.path.exists(pseudo_path):
        os.makedirs(pseudo_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    print(f"Start training on synthesized data using {args.syn_method} method...!")
    
    train_STDASeg(args, snapshot_path, model1, model2, src_trainloader,  trg_trainloader, valloader)
        
    
    torch.cuda.empty_cache()
