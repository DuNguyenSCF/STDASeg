import os
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
def add_generate_syndata_STDASeg_args(arg_parser):
    arg_parser.add_argument('--root_path', type=str,
                    default='/data/gpfs/projects/punim1699/data_dunguyen/Chundata_bridgecrack', help='Path to data')

    arg_parser.add_argument('--syn_concrete_data', action='store_true', help="Synthesize concrete data or not. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--syn_pav_data', action='store_true', help="Synthesize pavement data or not. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--saved_syn_samples', type=int,
                        default=100, help='The number of synthesized samples being saved to check')
    arg_parser.add_argument('--save_syn_path', type=str,
                    default='../syn_samples', help='Path to data')

    arg_parser.add_argument('--train_syn_src_filenames', type=str,
                        default='train_src_concrete_aug_crop512_split.txt', help='List of source concrete images for synthesizing new data')

    arg_parser.add_argument('--train_syn_trg_filenames', type=str,
                        default='train_syn_trg_concrete_noncrack_512_split.txt', help='List of target non-crack images for synthesizing new data')

    arg_parser.add_argument('--syn_method', type=str,
                        default='fpie', help='Name of synthesis method: cutpaste, dove, fpie')


    arg_parser.add_argument('-l','--patch_size', nargs='+', type=int,  default=[512, 512],
                        help='Patch size of network input')
    arg_parser.add_argument('--seed', type=int,  default=1337, help='Random seed')
    arg_parser.add_argument('--num_classes', type=int,  default=2,
                        help='Output channel of network')
    
    arg_parser.add_argument('--load_pretrained_weights', action='store_true', help="Load pre-trained weights on large-scale dataset or not. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--deterministic', type=int,  default=1,
                        help='Whether use deterministic training')
    
    return arg_parser
        
def add_train_STDASeg_args(arg_parser):
    arg_parser.add_argument('--root_path', type=str,
                    default='/data/gpfs/projects/punim1699/data_dunguyen/Chundata_bridgecrack', help='Path to data')

    arg_parser.add_argument('--syn_concrete_data', action='store_true', help="Synthesize concrete data or not. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--syn_pav_data', action='store_true', help="Synthesize pavement data or not. Default value is False if not specifying in command line, otherwise, True")
    # arg_parser.add_argument('--saved_syn_samples', type=int,
    #                     default=100, help='The number of synthesized samples being saved to check')

    arg_parser.add_argument('--train_syn_src_filenames', type=str,
                        default='train_src_concrete_aug_crop512_split.txt', help='List of source concrete images for synthesizing new data')

    arg_parser.add_argument('--train_syn_trg_filenames', type=str,
                        default='train_syn_trg_concrete_noncrack_512_split.txt', help='List of target non-crack images for synthesizing new data')

    arg_parser.add_argument('--train_trg_filenames', type=str,
                        default='train_new_crop512_selected.txt', help='List of unlabeled images for training from the target domain')

    arg_parser.add_argument('--valid_on_trg_domain', action='store_true', help="Valid on target domain or not. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--val_trg_filenames', type=str,
                        default='val_new_512.txt', help='List of cropped target images for validation')
    
    # arg_parser.add_argument('--val_syn_filenames', type=str,
    #                     default='valid_new_fpie_c2c_512.txt', help='Valid Sample filenames')


    arg_parser.add_argument('--train_syn_transform', action='store_true', help="Augmentation or not. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--train_trg_transform', action='store_true', help="Augmentation or not. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--val_transform', action='store_true', help="Augmentation or not. Default value is False if not specifying in command line, otherwise, True")

    arg_parser.add_argument('--syn_method', type=str,
                        default='fpie', help='Name of synthesis method: cutpaste, dove, fpie')
    arg_parser.add_argument('--exp', type=str,
                        default='C2C/STDASeg_FPIE_50000iters_bs8', help='Experiment_name')
    arg_parser.add_argument('--cnn_model', type=str,
                        default='TransU', help='Model_name')
    arg_parser.add_argument('--trans_model', type=str,
                        default='SwinU', help='Model_name')

    arg_parser.add_argument('--max_iterations', type=int,
                        default=50000, help='Maximum epoch number to train')
    arg_parser.add_argument('--valid_checkpoint', type=int,
                        default=500, help='The number of iteration to check performance on validation set')
    arg_parser.add_argument('--save_best_ckpt_with_iter', action='store_true', help="Save the best checkpoit at the current iter. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--save_ckpt_every_5000_inters', action='store_true', help="Save checkpoit at every 5000 iterations. Default value is False if not specifying in command line, otherwise, True")
    
    arg_parser.add_argument('--deterministic', type=int,  default=1,
                        help='Whether use deterministic training')

    arg_parser.add_argument('--base_lr_1', type=float,  default=0.01,
                        help='Segmentation network 1 learning rate') 
    arg_parser.add_argument('--weight_decay_1', type=float,  default=1e-4,
                        help='Segmentation network 2 learning rate')
    arg_parser.add_argument('--base_lr_2', type=float,  default=0.01,
                        help='Segmentation network 2 learning rate')
    arg_parser.add_argument('--weight_decay_2', type=float,  default=1e-4,
                        help='Segmentation network 2 learning rate')

    arg_parser.add_argument('--DAN_lr', type=float,  default=0.0001,
                        help='DAN learning rate')

    arg_parser.add_argument('--relaxed_weight_factor', type=float,  default=0.2,
                        help='Hyper-parameter in calculating class balanced weighs')
    arg_parser.add_argument('--cdd_weight', type=float,  default=0.01,
                        help='Consistency weight of cdd loss')
    arg_parser.add_argument('--ent_weight', type=float,  default=0.1,
                        help='Consistency weight of ent loss')


    arg_parser.add_argument('-l','--patch_size', nargs='+', type=int,  default=[512, 512],
                        help='Patch size of network input')
    arg_parser.add_argument('--seed', type=int,  default=1337, help='Random seed')
    arg_parser.add_argument('--num_classes', type=int,  default=2,
                        help='Output channel of network')

    
    
    arg_parser.add_argument('--is_stage_1', action='store_true', help="Use stage 1 or not. Default value is False if not specifying in command line, otherwise, True")

    arg_parser.add_argument('--thresholded_pseudo', action='store_true', help="Threshold pseudo labels or not. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--lower_bound_thrh', type=float, default=0.5,
                        help='The lowest propability value when classifying a pixel into crack class')
    arg_parser.add_argument('--save_pseudo', action='store_true', help="Save pseudo labels or not. Default value is False if not specifying in command line, otherwise, True")

    # For ViT
    arg_parser.add_argument('--n_skip', type=int,
                        default=3, help='Using number of skip-connect, default is num')
    arg_parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='Select one vit model for TransUnet')
    arg_parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='ViT patche size, default is 16')
    
    arg_parser.add_argument('--load_pretrained_weights', action='store_true', help="Load pre-trained weights on large-scale dataset or not. Default value is False if not specifying in command line, otherwise, True")
    arg_parser.add_argument('--encoder_weights', type=str,
                        default='imagenet', help='Pre-trained weights for CNN models')

    # label and unlabel
    arg_parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per gpu')
    arg_parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Batch size per gpu')
    arg_parser.add_argument('--labeled_bs', type=int, default=4,
                        help='Labeled_batch_size per gpu')
    arg_parser.add_argument('--labeled_num', type=int, default=4032,
                        help='labeled data')

    # costs
    arg_parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    arg_parser.add_argument('--consistency_type', type=str,
                        default="mse", help='consistency_type')
    arg_parser.add_argument('--consistency', type=float,
                        default=1.5, help='consistency')
    arg_parser.add_argument('--consistency_rampup', type=float,
                        default=100, help='consistency_rampup_choosen_as_num_epochs')

    # pre-train models for initialization
    arg_parser.add_argument('--checkpoint_1_base_dir', type=str,
                        default='', help='Stored checkpoint 1 dir')
    arg_parser.add_argument('--checkpoint_1', type=str,
                        default='', help='Name of checkpoint 1')
    arg_parser.add_argument('--checkpoint_2_base_dir', type=str,
                        default='', help='Stored checkpoint 1 dir')
    arg_parser.add_argument('--checkpoint_2', type=str,
                        default='', help='Name of checkpoint 2')
    
    return arg_parser

def initialize_STDASeg_train_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_STDASeg_args(arg_parser)
    args = arg_parser.parse_args()
    return args

def add_test_STDASeg_args(test_arg_parser):
    test_arg_parser.add_argument('--root_path', type=str,
                    default='/data/gpfs/projects/punim1699/data_dunguyen/Chundata_bridgecrack', help='Path to data')
    test_arg_parser.add_argument('--fullsize_gt', type=str,
                        default='Testset/new_masks_3072x5120', help='Path to full size mask')
    test_arg_parser.add_argument('--crop_gt', type=str,
                        default='Testset/crack_masks_512x512', help='Path to cropped size mask')
    test_arg_parser.add_argument('--trg_fullsize_filenames', type=str,
                        default='test_new_fullsize_3072x5120.txt', help='Test Fullsize Sample filenames')
    test_arg_parser.add_argument('--trg_crop_filenames', type=str,
                        default='test_new_crop512.txt', help='Test Crop Sample filenames')

    # parser.add_argument('--load_pretrained_weights', action='store_true', help="Load pretrained weights or not. Default value is False if not specifying in command line, otherwise, True")

    test_arg_parser.add_argument('--preprocess', action='store_true', help="Preprocess testing data or not. Default value is False if not specifying in command line, otherwise, True")

    test_arg_parser.add_argument('--exp', type=str,
                        default='C2C_STDASeg/STDASeg_FPIE_50000iters_bs8', help='experiment_name')
    
    # For ViT
    test_arg_parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    test_arg_parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model for TransUnet')
    test_arg_parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    
    test_arg_parser.add_argument('--cnn_model', type=str,
                    default='TransU', help='model_name')
    test_arg_parser.add_argument('--trans_model', type=str,
                    default='SwinU', help='model_name')
    

    test_arg_parser.add_argument('--max_iterations', type=int,
                        default=50000, help='maximum epoch number to train')
    test_arg_parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size per gpu')
    test_arg_parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    test_arg_parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    test_arg_parser.add_argument('-l','--patch_size', nargs='+', type=int,  default=[512, 512],
                    help='patch size of network input')
    test_arg_parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    test_arg_parser.add_argument('--num_classes', type=int,  default=2,
                        help='output channel of network')

    # label and unlabel
    test_arg_parser.add_argument('--labeled_num', type=int, default=4032,
                        help='labeled data')

    # test
    test_arg_parser.add_argument('--test_on_model1', action='store_true', help="test on model 1. Default value is False if not specifying in command line, otherwise, True")
    test_arg_parser.add_argument('--checkpoint_1', type=str,
                        default='TransU_best_model1.pth', help='name of checkpoint1')
    test_arg_parser.add_argument('--test_on_model2', action='store_true', help="test on model 2. Default value is False if not specifying in command line, otherwise, True")
    test_arg_parser.add_argument('--checkpoint_2', type=str,
                        default='SwinU_best_model2.pth', help='name of checkpoint2')
    
    test_arg_parser.add_argument('--test_full', action='store_true', help="test on full or cropped images. Default value is False if not specifying in command line, otherwise, True")

    test_arg_parser.add_argument('--overlap_h', type=int,  default=11,
                        help='number of vertically overlaped patches')
    test_arg_parser.add_argument('--overlap_w', type=int,  default=19,
                        help='number of horizontally overlaped patches')
    test_arg_parser.add_argument('--overlap_pixels', type=int,  default=256,
                        help='number of overlaped pixels')
    test_arg_parser.add_argument('--fullsize_h', type=int,  default=3072,
                        help='number of vertically overlaped patches')
    test_arg_parser.add_argument('--fullsize_w', type=int,  default=5120,
                        help='number of horizontally overlaped patches')
    test_arg_parser.add_argument('--method', type=str,
                        default='C2C_STDASeg_FPIE', help='method to create train data')
    test_arg_parser.add_argument('--iters_1', type=str,
                    default='', help='best checkpoint or at specific iterations')
    test_arg_parser.add_argument('--iters_2', type=str,
                    default='', help='best checkpoint or at specific iterations')
    test_arg_parser.add_argument('--results_dir', type=str,
                        default='../Sum_Results/Results_512', help='where to save results')
    
    return test_arg_parser

# def initialize_STDASeg_test_args():
#     test_arg_parser = argparse.ArgumentParser()
#     test_arg_parser = add_test_STDASeg_args(test_arg_parser)
#     test_args = test_arg_parser.parse_args()
#     return test_args