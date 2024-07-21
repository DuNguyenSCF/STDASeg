import os
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
def add_train_Supervised_args(s_arg_parser):
    s_arg_parser.add_argument('--root_path', type=str,
                    default='/data/gpfs/projects/punim1699/data_dunguyen/Chundata_bridgecrack', help='Path to data')
    s_arg_parser.add_argument('--train_filenames', type=str,
                        default='train_src_concrete_aug_crop512_split.txt', help='Train Sample filenames')
    s_arg_parser.add_argument('--val_filenames', type=str,
                        default='val_new_512.txt', help='Valid Sample filenames')
    s_arg_parser.add_argument('--train_transform', action='store_true', help="Augmentation or not. Default value is False if not specifying in command line, otherwise, True")
    s_arg_parser.add_argument('--val_transform', action='store_true', help="Augmentation or not. Default value is False if not specifying in command line, otherwise, True")

    s_arg_parser.add_argument('--exp', type=str,
                        default='C2C_src_512/Supervised_src_concrete_data_50000iters_bs8', help='experiment_name')
    s_arg_parser.add_argument('--model', type=str,
                        default='SwinU', help='model_name')
    s_arg_parser.add_argument('--max_iterations', type=int,
                        default=50000, help='maximum epoch number to train')
    s_arg_parser.add_argument('--save_checkpoint', type=int,
                        default=500, help='number of iters for checking performance')
    s_arg_parser.add_argument('--save_best_ckpt_with_iter', action='store_true', help="Save the best checkpoit at the current iter. Default value is False if not specifying in command line, otherwise, True")
    s_arg_parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    s_arg_parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    s_arg_parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    s_arg_parser.add_argument('-l','--patch_size', nargs='+', type=int,  default=[512, 512],
                        help='patch size of network input')
    s_arg_parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    s_arg_parser.add_argument('--num_classes', type=int,  default=2,
                        help='output channel of network')


    # For ViT
    s_arg_parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    s_arg_parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model for TransUnet')
    s_arg_parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')


    s_arg_parser.add_argument('--load_pretrained_weights', action='store_true', help="Load pre-trained weights on large-scale dataset or not. Default value is False if not specifying in command line, otherwise, True")
    s_arg_parser.add_argument('--encoder_weights', type=str,
                        default='imagenet', help='pre-trained weights for CNN models')




    # label and unlabel
    s_arg_parser.add_argument('--labeled_num', type=int, default=4008,
                        help='labeled data')
    
    return s_arg_parser

def initialize_Supervised_train_args():
    s_arg_parser = argparse.ArgumentParser()
    s_arg_parser = add_train_Supervised_args(s_arg_parser)
    s_args = s_arg_parser.parse_args()
    return s_args

def add_test_args(test_arg_parser):
    test_arg_parser.add_argument('--root_path', type=str,
                    default='/data/gpfs/projects/punim1699/data_dunguyen/Chundata_bridgecrack', help='Path to data')
    test_arg_parser.add_argument('--fullsize_gt', type=str,
                        default='Testset/new_masks_3072x5120', help='Name of Experiment')
    test_arg_parser.add_argument('--crop_gt', type=str,
                        default='Testset/crack_masks_512x512', help='Name of Experiment')
    test_arg_parser.add_argument('--trg_fullsize_filenames', type=str,
                        default='test_new_fullsize_3072x5120.txt', help='Test Fullsize Sample filenames')
    test_arg_parser.add_argument('--trg_crop_filenames', type=str,
                        default='test_new_crop512.txt', help='Test Crop Sample filenames')

    # parser.add_argument('--load_pretrained_weights', action='store_true', help="Load pretrained weights or not. Default value is False if not specifying in command line, otherwise, True")

    test_arg_parser.add_argument('--preprocess', action='store_true', help="Preprocess testing data or not. Default value is False if not specifying in command line, otherwise, True")

    test_arg_parser.add_argument('--exp', type=str,
                        default='C2C_src_512/Supervised_src_concrete_data_50000iters_bs8', help='experiment_name')
    
    # For ViT
    test_arg_parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    test_arg_parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model for TransUnet')
    test_arg_parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    
    test_arg_parser.add_argument('--model', type=str,
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
    test_arg_parser.add_argument('--labeled_num', type=int, default=4008,
                        help='labeled data')

    # test
    test_arg_parser.add_argument('--checkpoint', type=str,
                        default='SwinU_best_model.pth', help='name of checkpoint')
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
                        default='C2C_src_512', help='method to create train data')
    test_arg_parser.add_argument('--iters', type=str,
                        default='', help='best checkpoint or at specific iterations')
    test_arg_parser.add_argument('--results_dir', type=str,
                        default='../Sum_Results/Results_512', help='where to save results')
    
    return test_arg_parser