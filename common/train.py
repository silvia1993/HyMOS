from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset

from common.common import parse_args
import models.classifier as C
from datasets.datasets import get_dataset_2, get_style_dataset, get_datasets_for_test, BalancedMultiSourceRandomSampler
from utils.utils import load_checkpoint
import sys
import os

P = parse_args()
P.iterative = True
P.mode = "HyMOS_st"
P.model = "resnet50_imagenet"
P.resize_factor = 0.08
P.simclr_dim = 128
P.iterations = 40000
P.its_breakpoints = [20000,25000,30000,35000]
P.im_mean = [0.485, 0.456, 0.406]
P.im_std = [0.229, 0.224, 0.225]
P.optimizer = "lars"
P.lr_scheduler = "cosine"
P.warmup = 2500

# batch_K -> number of known classes
# batch_p -> number of source domains
# they are necessary to build balanced batches.

if P.dataset == "OfficeHome":
    P.batch_K = 45
    P.batch_p = 3
elif P.dataset == "DomainNet":
    P.batch_K = 100
    P.batch_p = 2
elif P.dataset == "Office31":
    P.batch_K = 20
    P.batch_p = 2
else:
    raise NotImplementedError(f"Unknown dataset {P.dataset}")

### Set torch device ###
if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()

import apex
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np

P.multi_gpu = True
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
)

# get local rank and world size from torch distributed
P.local_rank=int(os.environ['RANK'])
P.n_gpus=torch.distributed.get_world_size()
print(f"Num of GPUS {P.n_gpus}")

### Initialize dataset ###
train_sets, n_classes = get_dataset_2(P)

# we have a list of ConcatDatasets, one for each class
P.image_size = (224, 224, 3)
P.n_classes = n_classes
kwargs = {'pin_memory': False, 'num_workers': 2, 'drop_last':True}

assert P.batch_K%P.n_gpus == 0, "batch_K has to be divisible by world size!!"
single_GPU_batch_K = P.batch_K/P.n_gpus
single_GPU_batch_size = int(P.batch_p*single_GPU_batch_K)
whole_source = ConcatDataset([train_sets[idx] for idx in range(n_classes)])
my_sampler = BalancedMultiSourceRandomSampler(whole_source, P.batch_p, P.local_rank, P.n_gpus)
print(f"{P.local_rank} sampler_size: {len(my_sampler)}. Dataset_size: {len(whole_source)}")
train_loader = DataLoader(whole_source, sampler=my_sampler, batch_size=single_GPU_batch_size, **kwargs)

### Test datasets and data loaders
source_ds, target_ds, _, = get_datasets_for_test(P)
    
source_test_sampler = DistributedSampler(source_ds, num_replicas=P.n_gpus, rank=P.local_rank)
target_test_sampler = DistributedSampler(target_ds, num_replicas=P.n_gpus, rank=P.local_rank)
source_test_loader = DataLoader(source_ds, sampler=source_test_sampler, shuffle=False, batch_size=1, num_workers=4)
target_test_loader = DataLoader(target_ds, sampler=target_test_sampler, shuffle=False, batch_size=1, num_workers=4)

### Initialize model ###
# define transformations for SimCLR augmentation
simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)

# generate model (includes backbone + classification head)
model = C.get_classifier(P.model, n_classes=P.n_classes, pretrain=P.pretrain).to(device)

# modify normalize params if necessary
mean = torch.tensor(P.im_mean).to(device)
std = torch.tensor(P.im_std).to(device)
model.normalize.mean=mean
model.normalize.std=std

criterion = nn.CrossEntropyLoss().to(device)

if P.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
elif P.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=P.lr_init, betas=(.9, .999), weight_decay=P.weight_decay)
    lr_decay_gamma = 0.3
elif P.optimizer == 'lars':
    from torchlars import LARS
    # wrap SGD in LARS for multi-gpu optimization
    base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init*10.0, momentum=0.9, weight_decay=P.weight_decay)
    optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    lr_decay_gamma = 0.1
else:
    raise NotImplementedError()

if P.lr_scheduler == 'cosine': # default choice
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.iterations)
elif P.lr_scheduler == 'step_decay':
    milestones = [int(0.5 * P.iterations), int(0.75 * P.iterations)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
else:
    raise NotImplementedError()

# a warmup scheduler is used in the first iterations, then substituted with the scheduler defined above
from training.scheduler import GradualWarmupScheduler
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=P.warmup, after_scheduler=scheduler)

resume = False
start_its = 1

if "st" in P.mode: 
    assert P.adain_ckpt is not None, "You need to pass adain ckpt path using --adain_ckpt"

    # first we load style transfer model 
    from models.adain import AdaIN 
    adain_model = AdaIN()
    adain_model.load_state_dict(torch.load(P.adain_ckpt, map_location=lambda storage, loc: storage))
    print('Adain augmentation setup: loaded checkpoint {}.'.format(P.adain_ckpt))
    P.adain_model = adain_model

    # now we also need to create a data loader for style images
    style_dataset = get_style_dataset(P)

    if P.multi_gpu:
        style_sampler = DistributedSampler(style_dataset, num_replicas=P.n_gpus, rank=P.local_rank)
        style_loader = DataLoader(style_dataset, sampler=style_sampler, batch_size=P.batch_size, **kwargs)
    else:
        style_loader = DataLoader(style_dataset, shuffle=True, batch_size=P.batch_size, **kwargs)
    P.style_loader = style_loader

    # for images on which we apply style transfer we don't want to have also 
    # color jittering and grey scale among simclr augmentations
    P.simclr_aug_st = C.get_simclr_augmentation_crop_only(P, image_size=P.image_size).to(device)

if P.multi_gpu:
    simclr_aug = apex.parallel.DistributedDataParallel(simclr_aug, delay_allreduce=True)
    model = apex.parallel.convert_syncbn_model(model)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
