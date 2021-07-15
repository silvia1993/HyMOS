from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import parse_args
import models.classifier as C
from datasets.datasets import get_datasets_for_test

P = parse_args()
P.mode = "openset_eval"
P.model = "resnet50_imagenet"
P.im_mean = [0.485, 0.456, 0.406]
P.im_std = [0.229, 0.224, 0.225]


### Set torch device ###

P.n_gpus = torch.cuda.device_count()
assert P.n_gpus <= 1  # no multi GPU
P.multi_gpu = False

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_ds, target_ds, n_classes = get_datasets_for_test(P)
P.image_size = (224, 224, 3)
P.n_classes = n_classes

source_test_loader = DataLoader(source_ds, shuffle=False, batch_size=1, num_workers=4)
target_test_loader = DataLoader(target_ds, shuffle=False, batch_size=1, num_workers=4)

### Initialize model ###

model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)

criterion = nn.CrossEntropyLoss().to(device)

# set normalize params
mean = torch.tensor(P.im_mean).to(device)
std = torch.tensor(P.im_std).to(device)
model.normalize.mean=mean
model.normalize.std=std

assert P.load_path is not None, "You need to pass checkpoint path using --load_path"

checkpoint = torch.load(P.load_path)

missing, unexpected = model.load_state_dict(checkpoint, strict=not P.no_strict) 
print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

