import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from data.util import *

from apex.optimizers import FusedAdam
from apex import amp
from apex.fp16_utils import FP16_Optimizer
from dist_utils import SimpleDistributedDataParallel

def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

def switch_schedule(schedule, mult, switch):
    """ Apply LR multiplier before iteration "switch" """
    def f(e):
        s = schedule(e)
        if e < switch:
            return s * mult
        return s 
    return f

def linear_schedule(args):
    def f(e):
        if e <= args.warmup:
            return e / args.warmup 
        return max((e - args.iterations) / (args.warmup - args.iterations), 0)
    return f

def create_optimizers(model, args, lr_schedule, prev_optimizer=None, prev_scheduler=None):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = FusedAdam(params, lr=args.lr)
    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True, verbose=False)

    if prev_optimizer is not None:
        optimizer.load_state_dict(prev_optimizer.state_dict())

    if args.warmup < 0:
        print('No learning rate schedule used.')
    else:
        print('Using learning rate schedule.')
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_schedule)
        if prev_scheduler is not None:
            # Continue LR schedule from previous scheduler
            scheduler.load_state_dict(prev_scheduler.state_dict())

    loss_model = SimpleDistributedDataParallel(model, args.world_size)
    return loss_model, optimizer, scheduler if args.warmup > 0 else None