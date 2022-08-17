# DAMA pre-training code
# References: MAE https://github.com/facebookresearch/mae

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from functools import partial
import numpy as np
from PIL import Image
from typing import Iterable

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import timm
import timm.optim.optim_factory as optim_factory

import DAMA.vision_transformer_sampling as vits
import DAMA.builder_sampling as builder
import utils
from utils import NativeScalerWithGradNormCount as NativeScaler

from data_transform import *
import tifffile

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['main_vit_tiny', 'main_vit_base'] + torchvision_archs

def get_args_parser():
    parser = argparse.ArgumentParser('DAMA', add_help=False)

    # Training comments
    parser.add_argument('--train_comment', default='', type=str,
                        help='Training comments')

    # Model params
    parser.add_argument('--arch', default='main_vit_base', type=str, metavar='ARCH',
                        choices=model_names, help='Name of model to train')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='images input size')
    parser.add_argument('--in_chans', default=7, type=int,
                        help='input channels')
    parser.add_argument('--img_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_overlap_ratio', default=0.5, type=float,
                        help='Masking overlap ratio between input student and teacher.')
    parser.add_argument('--mask_sampling', default='random', type=str,
                        help='Masks sampling strategy: random, etc.')
    parser.add_argument('--loss_beta', default=2.0, type=float,
                        help='Beta for smooth L1 loss.')
    parser.add_argument('--loss_alpha', default=1.0, type=float,
                        help='Scale between features loss and reconstruction loss.')
    parser.add_argument('--last_k_blocks', default=6, type=int,
                        help='Last K blocks of transformer to get the output for features target.')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss.')
    parser.set_defaults(norm_pix_loss=True)

    # Training params
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Optimizer params
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument("--warmup_epochs", default=40, type=int,
        help='Number of epochs for the linear learning-rate warm up.')

    parser.add_argument("--stable_epoch", default=0, type=int,
        help='Number of epochs for stable the training by set if stable_epoch < epoch: loss = recons_loss')

    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw'], help='Type of optimizer. We recommend using adamw with ViTs.')

    parser.add_argument('--main_m', default=0.99, type=float,
                        help='masked information momentum of updating momentum model (default: 0.99).')
    parser.add_argument('--main_m_cos', action='store_true',
                        help='gradually increase moco momentum to 1 with a half-cycle cosine schedule.')

    # Dataset params
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed params
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed) # fix the seed for reproducibility

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    # simple augmentation
    if args.in_chans != 7:
        transform_train = transforms.Compose([
                transforms.CenterCrop(50),
                transforms.RandomResizedCrop(args.img_size, scale=(0.5, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    elif args.in_chans == 7:
        def my_tiff_loader(filename):
            return tifffile.imread(filename)

        transform_train = Compose([
            RandomResizedCrop(args.img_size // 2, args.img_size),  # 3 is bicubic
            RandomRotation(90),
            RandomShift(0.3),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()])
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), loader=my_tiff_loader,
                                             transform=transform_train)

    print(dataset_train)
    print(f"Data loaded: there are {len(dataset_train)} images.")

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model = builder.DAMA(
            partial(vits.__dict__[args.arch], patch_size=args.patch_size, mask_ratio=args.mask_ratio,
                    mask_overlap_ratio=args.mask_overlap_ratio, img_size=args.img_size, in_chans=args.in_chans),
            loss_beta=args.loss_beta, last_k_blocks=args.last_k_blocks, loss_alpha=args.loss_alpha,
                    norm_pix_loss=args.norm_pix_loss, in_chans=args.in_chans)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()

    # infer learning rate before changing batch size
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256  # blr : base learning rate

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    utils.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.stable_epoch > 0:
        print('Training with stable_epoch {}'.format(args.stable_epoch))
    else:
        print('Not training with stable_epoch')

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 30 == 0 or epoch + 1 == args.epochs):
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def adjust_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.main_m)
    return m

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_info', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_recons_txt', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_recons_img', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    main_m = args.main_m

    print('len data_loader: %s' % len(data_loader))
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # adjust learning rate and momentum coefficient per iteration
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, epoch + data_iter_step / len(data_loader), args)

        if args.main_m_cos:
            main_m = adjust_momentum(epoch + data_iter_step / len(data_loader), args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            info_loss, recons_img, recons_txt, _, _, _, _ = model(samples, main_m)

            # train with reconstruction loss to get the training stable
            if args.stable_epoch > 0 and epoch < args.stable_epoch:
                if epoch == 0:
                    print('Begin of stable_epoch {}'.format(args.stable_epoch))
                loss = recons_txt + recons_img
            else:
                if epoch == args.stable_epoch:
                    print('End of stable_epoch {}'.format(args.stable_epoch))
                loss = info_loss + recons_txt + recons_img

        loss_value = loss.item()
        loss_info = info_loss.item()
        loss_recons_txt = recons_txt.item()
        loss_recons_img = recons_img.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("loss_info is {}, stopping training".format(loss_info))
            print("loss_recons_txt is {}, stopping training".format(loss_recons_txt))
            print("loss_recons_img is {}, stopping training".format(loss_recons_img))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_info=loss_info)
        metric_logger.update(loss_recons_txt=loss_recons_txt)
        metric_logger.update(loss_recons_img=loss_recons_img)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = utils.all_reduce_mean(loss_value)
        loss_info_reduce = utils.all_reduce_mean(loss_info)
        loss_recons_txt_reduce = utils.all_reduce_mean(loss_recons_txt)
        loss_recons_img_reduce = utils.all_reduce_mean(loss_recons_img)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('loss_info', loss_info_reduce, epoch_1000x)
            log_writer.add_scalar('loss_recons_txt', loss_recons_txt_reduce, epoch_1000x)
            log_writer.add_scalar('loss_recons_img', loss_recons_img_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def copy_folder(src, dst):
    import shutil
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAMA', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args)
    start_train = time.time()
    main(args)
    end_train = time.time()
    print('Total training time is: %s hours' %((end_train - start_train)/3600))
