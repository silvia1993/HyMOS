import time
import numpy as np
from torchvision import datasets, transforms

import torch.optim

import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix_padded, get_similarity_matrix, Supervised_NT_xent_padded, Supervised_NT_xent
from utils.utils import AverageMeter, normalize, apply_simclr_aug
import torch.distributed as dist
import diffdist.functional as distops
from datasets.datasets import BalancedMultiSourceRandomSampler
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)

def train(P, its, model, criterion, optimizer, scheduler, data_loader, data_iter, logger=None, simclr_aug=None):

    assert simclr_aug is not None
    
    try: 
        ims, lbls, path = next(data_iter)
    except StopIteration:
        my_sampler = BalancedMultiSourceRandomSampler(data_loader.dataset, P.batch_p, P.local_rank, P.n_gpus)
        loader_kwargs = {'pin_memory': False, 'num_workers': 1, 'drop_last':True}
        data_loader = DataLoader(data_loader.dataset, sampler=my_sampler, batch_size=data_loader.batch_size, **loader_kwargs)
        data_iter = iter(data_loader)
        ims, lbls, path = next(data_iter)
    images1 = ims[0]
    images2 = ims[1]
    labels = lbls 
    images1 = images1.to(device)
    images2 = images2.to(device)
    labels = labels.to(device)

    try:
        style_images, _ = next(P.style_iter)
    except StopIteration:
        P.style_iter = iter(P.style_loader)
        style_images, _ = next(P.style_iter)

    images_pair = torch.cat([images1, images2], dim=0)  # 2B

    images_pair = apply_simclr_aug(P, simclr_aug, images_pair, style_images)  # simclr augmentation

    # perform forward
    _, outputs_aux = model(images_pair, simclr=True, penultimate=True)

    # normalize output
    simclr = normalize(outputs_aux['simclr'])  # normalize
    # compute similarities

    sim_matrix = get_similarity_matrix(simclr,multi_gpu=P.multi_gpu)

    # obtain simclr (supclr) loss
    temperature = P.temperature
    loss_sim = Supervised_NT_xent(sim_matrix, labels=labels, temperature=temperature, multi_gpu=P.multi_gpu)

    ### total loss ###
    loss = loss_sim

    # perform backward and step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    scheduler.step()

    ### Post-processing stuffs ###
    simclr_norm = outputs_aux['simclr'].norm(dim=1).mean() # compute avg norm of output features 

    return loss_sim, simclr_norm, data_loader, data_iter
