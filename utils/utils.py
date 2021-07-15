import os
import pickle
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from torch.nn.functional import interpolate
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, ask=True, local_rank=0):
        self.local_rank = local_rank
        if self.local_rank == 0:
            if not os.path.exists("./logs/"):
                os.mkdir("./logs/")

            logdir = self._make_dir(fn)
            if not os.path.exists(logdir):
                os.mkdir(logdir)

            if len(os.listdir(logdir)) != 0 and ask:
                ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
                if ans in ['y', 'Y']:
                    shutil.rmtree(logdir)
                else:
                    exit(1)

            self.set_dir(logdir)

    def _make_dir(self, fn):
        today = datetime.today().strftime("%y%m%d")
        logdir = 'logs/' + fn
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        if self.local_rank == 0:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.local_rank == 0:
            self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
            self.log_file.flush()

            print('%s (%s)' % (string, self.logdir))
            sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.local_rank == 0:
            self.writer.add_scalar(tag, value, step)

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        if self.local_rank == 0:
            self.writer.add_images(tag, images, step)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        if self.local_rank == 0:
            self.writer.add_histogram(tag, values, step, bins='auto')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def load_checkpoint(logdir, mode='last'):
    if mode == 'last':
        model_path = os.path.join(logdir, 'last.model')
        optim_path = os.path.join(logdir, 'last.optim')
        config_path = os.path.join(logdir, 'last.config')
    elif mode == 'best':
        model_path = os.path.join(logdir, 'best.model')
        optim_path = os.path.join(logdir, 'best.optim')
        config_path = os.path.join(logdir, 'best.config')

    else:
        raise NotImplementedError()

    print("=> Loading checkpoint from '{}'".format(logdir))
    if os.path.exists(model_path):
        model_state = torch.load(model_path)
        optim_state = torch.load(optim_path)
        with open(config_path, 'rb') as handle:
            cfg = pickle.load(handle)
    else:
        return None, None, None

    return model_state, optim_state, cfg


def save_checkpoint(its, model_state, optim_state, logdir):
    last_model = os.path.join(logdir, 'last.model')
    last_optim = os.path.join(logdir, 'last.optim')
    last_config = os.path.join(logdir, 'last.config')

    opt = {
        'its': its,
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_checkpoint_name(model_state, logdir, prefix):
    last_model = os.path.join(logdir, f'{prefix}.model')
    torch.save(model_state, last_model)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def normalize_images(P, inputs):
    mean = torch.tensor(P.im_mean).to(inputs.device)
    std = torch.tensor(P.im_std).to(inputs.device)
    return  ((inputs.permute(0,2,3,1)-mean)/std).permute(0,3,1,2)

def denormalize_images(P, inputs):
    mean = torch.tensor(P.im_mean).to(inputs.device)
    std = torch.tensor(P.im_std).to(inputs.device)
    return  (inputs.permute(0,2,3,1)*std+mean).permute(0,3,1,2)

def apply_simclr_aug(P, simclr_aug, images, style_images):
    cpu_device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = len(images)
    adain_mask = torch.zeros((len(images)), dtype=torch.bool)

    selected_content_images = []
    selected_style_images = []
    cnt_ids = []
    for i in range(len(images)):
        adain_mask[i] = random.random() < P.adain_probability
        if adain_mask[i]:
            selected_content_images.append(images[i])
            selected_style_images.append(style_images[int(random.random()*len(style_images))])
            cnt_ids.append(i)

    no_adain_mask = ~adain_mask

    adain_model = P.adain_model.to(device)

    if len(selected_content_images) > 0:
        selected_content_images = normalize_images(P,torch.stack(selected_content_images).to(device))
        selected_style_images = normalize_images(P,torch.stack(selected_style_images).to(device))
        with torch.no_grad():
            output_images = adain_model.generate(
                    selected_content_images,
                    selected_style_images,
                    P.adain_alpha)

        images[adain_mask] = P.simclr_aug_st(denormalize_images(P,interpolate(output_images, images[0].shape[1:][::-1])))
        del output_images 
        del selected_content_images
        del selected_style_images

    if len(images[no_adain_mask]) > 0:
        images[no_adain_mask] = simclr_aug(images[no_adain_mask])

    adain_model = adain_model.to(cpu_device)

    return images


