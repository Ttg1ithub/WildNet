"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import random
import pdb

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR50V3PlusD',
                    help='Network architecture.')
parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes, mapillary, gtav, bdd100k, synthia')
#跨多源域统一采集图像
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['synthia'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--wild_dataset', nargs='*', type=str, default=['imagenet'],
                    help='a list consists of imagenet')
#交叉验证方式
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
#指定要从不同源域均匀抽样的图像比例的百分比
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
#用于类均匀抽样的瓷砖大小。在执行类均匀抽样时，数据可能会被分成块状区域，以便更有效地处理和采样数据
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
#使用粗略注释来增强具有特定类别的细粒度数据。
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

#类平衡加权损失
parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

#边界放松
parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

#混合精度训练（mixed precision training）和分布式训练
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

#优化器选择
parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

#冻结网络主干、挖掘硬样本
parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet-50',
                    help='trunk model, can be: resnet-50 (default)')
parser.add_argument('--max_epoch', type=int, default=256)
parser.add_argument('--max_iter', type=int, default=32768)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
#裁剪图像时是否填充
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
#裁剪尺寸
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
#模型恢复
parser.add_argument('--snapshot', type=str, default='/mnt/backup/gcw-yhj/wildnet/logs/ckpt/07-11-prot/V1/07_11_15/last_None_epoch_296_mean-iu_0.00000.pth')
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='/mnt/backup/gcw-yhj/wildnet/logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='/mnt/backup/gcw-yhj/wildnet/logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
#转储增强后的图像以进行健全性检查
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')

#Feature stylization的层数
parser.add_argument('--fs_layer', nargs='*', type=int, default=[0,0,0,0,0],
                    help='0: None, 1: AdaIN')
parser.add_argument('--lambda_cel', type=float, default=0.0,
                    help='lambda for content extension learning loss')
parser.add_argument('--lambda_sel', type=float, default=0.0,
                    help='lambda for style extension learning loss')
parser.add_argument('--lambda_scr', type=float, default=0.0,
                    help='lambda for semantic consistency regularization loss')
parser.add_argument('--cont_proj_head', type=int, default=0, 
                    help='number of output channels of content projection head')
parser.add_argument('--wild_cont_dict_size', type=int, default=0,
                    help='wild-content dictionary size')

parser.add_argument('--use_fs', action='store_true', default=False,
                    help='Automatic setting from fs_layer. feature stylization with wild dataset')
parser.add_argument('--use_scr', action='store_true', default=False,
                    help='Automatic setting from lambda_scr')
parser.add_argument('--use_sel', action='store_true', default=False,
                    help='Automatic setting from lambda_sel')
parser.add_argument('--use_cel', action='store_true', default=False,
                    help='Automatic setting from lambda_cel')

args = parser.parse_args()

random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                    init_method='env://',
                                    world_size=args.world_size,
                                    rank=args.local_rank)
# torch.distributed.init_process_group(backend='nccl',
#                                     init_method=args.dist_url,
#                                     world_size=args.world_size,
#                                     rank=args.local_rank)

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    prep_experiment(args, parser)
    writer = None

    _, val_loaders, _, _, extra_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux, args.cont_proj_head, args.wild_cont_dict_size)

    optim, scheduler = optimizer.get_optimizer(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)

    print("#### iteration", i)
    torch.cuda.empty_cache()

    for dataset, val_loader in val_loaders.items():
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):

        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            output = net(inputs)

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    return val_loss.avg


if __name__ == '__main__':
    main()
