import torch.nn as nn
import torch

from models.resnet_imagenet import resnet50
import models.transform_layers as TL

def get_simclr_augmentation(P, image_size):

    # parameter for resizecrop
    resize_scale = (P.resize_factor, 1.0) # resize scaling factor

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # disable resize_crop
    resize_crop = nn.Identity()

    if P.dataset == 'imagenet': # Using RandomResizedCrop at PIL transform
        transform = nn.Sequential(
            color_jitter,
            color_gray,
        )
    else:
        transform = nn.Sequential(
            color_jitter,
            color_gray,
            resize_crop,
        )

    return transform

def get_simclr_augmentation_crop_only(P, image_size):
    # parameter for resizecrop
    resize_scale = (P.resize_factor, 1.0) # resize scaling factor

    # Align augmentation
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # disable resize crop
    resize_crop = nn.Identity()
    # Transform define #
    transform = nn.Sequential(resize_crop)
    return transform

def get_classifier(mode, n_classes=10, pretrain=None):
    if mode == 'resnet50_imagenet':
        classifier = resnet50(num_classes=n_classes)
        if not pretrain is None:
            ckpt = torch.load(pretrain)
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            missing, unexpected = classifier.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from {pretrain}")
            print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
    else:
        raise NotImplementedError()

    return classifier

