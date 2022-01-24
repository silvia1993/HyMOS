# HyMOS (Hyperspherical classification for Multi-source Open-Set domain adaptation)

PyTorch official implementation of  "[Distance-based Hyperspherical Classification for Multi-source Open-Set Domain Adaptation](https://arxiv.org/abs/2107.02067)".
Video presentation available [here](https://www.youtube.com/watch?v=lleB3utdv9A). 
![Test Image 1](HyMOS.gif)

_Vision systems trained in close-world scenarios will inevitably fail when presented with new environmental conditions, new data distributions and novel classes at deployment time. How to move towards open-world learning is along standing research question, but the existing solutions mainly focus on specific aspects of the problem (single domain open-set, multi-domain closed-set), or propose complex strategies which combine multiple losses and manually tuned hyperparameters. In this work we tackle multi-source open-set domain adaptation by introducing HyMOS: a straightforward supervised model that exploits the power of contrastive learning and the properties of its hyperspherical feature space to correctly predict known labels on the target, while rejecting samples belonging to any unknown class. HyMOS  includes a tailored data balancing to enforce cross-source alignment and introduces style transfer among the instance transformations of contrastive learning for source-target adaptation, avoiding the risk of negative transfer. Finally a self-training strategy refines the model without the need for handcrafted thresholds. We validate our method over three challenging datasets and provide an extensive quantitative and qualitative experimental analysis. The obtained results show that HyMOS outperforms several open-set and universal domain adaptation approaches, defining the new state-of-the-art._

**Office-31 (HOS)**

|  | D,A -> W | W,A -> D | W,D -> A | Avg. |
| :---| :---: | :---: | :---: |---: |
| **HyMOS** | 90.2| 89.9| 60.8 | 80.3 |

**Office-Home (HOS)**

|  | Ar,Pr,Cl→Rw | Ar,Pr,Rw→Cl | Cl,Pr,Rw→Ar | Cl,Ar,Rw→Pr | Avg. |
| :---| :---: | :---: | :---: | :---: | ---: |
| **HyMOS** | 71.0 | 64.6 | 62.2 | 71.1 | 67.2 |

**DomainNet (HOS)**

|  | I,P -> S | I,P -> C | Avg. |
| :---| :---: | :---: | ---: 
| **HyMOS** | 57.5| 61.0| 59.3 | 


## Code

### 1. Requirements

#### Packages

The code requires these packages:
- python 3.6+
- torch 1.6+
- torchvision 0.7+
- CUDA 10.1+
- scikit-learn 0.22+
- tensorboardX 
- tqdm
- [torchlars](https://github.com/kakaobrain/torchlars) == 0.1.2 
- [apex](https://github.com/NVIDIA/apex) == 0.1
- [diffdist](https://github.com/ag14774/diffdist) == 0.1 

#### Datasets 

Datasets OfficeHome, Office31 and DomainNet should be placed in `~/data`. They can be downloaded
from official sites:

 - [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
 - [Office31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
 - [DomainNet](http://ai.bu.edu/M3SDA/)

Make sure that `~/data/OfficeHome/<domain>` points to the correct domain directory for all the domains. It may be possible that you need to rename *Real World* folder to remove the space.
Similarly you should check `~/data/DomainNet/<domain>` and `~/data/Office31/<domain>`.

#### Pretrained model 

We use ResNet50 pretrained via SupCLR, taken from [official github repository](https://github.com/google-research/google-research/tree/master/supcon).
We converted the checkpoint to pytorch format using this guide: [here](https://github.com/google-research/simclr#model-convertion-to-pytorch-format). 
The converted model is available [here](https://drive.google.com/file/d/1w-IdsYwCScbHTlCUDGCDxCPhY9VH6hRl/view?usp=sharing). 

#### AdaIN model

We use a freely available PyTorch based AdaIN implementation that can be found [here](https://github.com/irasin/Pytorch_AdaIN). Follow the instructions to train a model. Put source data
in `train_content_dir` and target data in `train_style_dir`. We also included a model trained by us
for the Office31 Dslr,Webcam -> Amazon shift together with this code. The file is named
`Amazon_adain.pth`.

### 2. Training

In the examples below the training is performed on multiple GPUs. It is possible to use more or
less by changing the value in `--nproc_per_node=2` and setting `CUDA_VISIBLE_DEVICES` appropriately.
In order to obtain domain and class-balance in each training mini batch the number of known classes
of the datasets has to be divisible by the number of GPUs used. For example in the case of
OfficeHome we use 3 GPUs because there are 45 known classes.

We use 'test_domain' to refer to target domain.
Train output, with saved models and log files, is stored in `logs/` directory.

#### Office31

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10001 train.py --dataset Office31 \
    --test_domain <test_domain> --pretrain <path_to_resnet50_supclr_pretrained.pth> --adain_ckpt <path_to_adain_checkpoint.pth>
```

Test domain should be one of "Amazon", "Webcam", "Dslr".

For example to train for the experiment having Amazon as target using the provided AdaIN model and a SupCLR pretrained ResNet50 model
we use:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10001 train.py --dataset Office31 \
    --test_domain Amazon --pretrain pretrained/resnet50_SupCLR.pth --adain_ckpt Amazon_adain.pth
```

Use a different port if 10001 is already taken. 

#### OfficeHome

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=10001 train.py --dataset OfficeHome \
    --test_domain <test_domain> --pretrain <path_to_resnet50_supclr_pretrained.pth> --adain_ckpt <path_to_adain_checkpoint.pth>
```

Test domain should be one of "Art", "Clipart", "Product", "RealWorld".

#### DomainNet

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10001 train.py --dataset DomainNet \
    --test_domain <test_domain> --pretrain <path_to_resnet50_supclr_pretrained.pth> --adain_ckpt <path_to_adain_checkpoint.pth>
```

Test domain should be one of "ipc", "ips".

### 3. Evaluation

Periodic evaluation is performed during training. The final model can be tested using: 

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset <dataset> --test_domain <target_domain> --load_path <path_to_last.model>
```

For example to test the model trained for Office31 shift having Amazon as target we use:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset Office31 --test_domain Amazon --load_path logs/Dataset-Office31_Target-Amazon_Mode-HyMOS_st_batchK-20_batchP-2_iterative_ProbST-0.5/last.model
```

## Citation

To cite, please use the following reference: 
```
@inproceedings{bucci2022distance,
  title={Distance-based Hyperspherical Classification for Multi-source Open-Set Domain Adaptation},
  author={Silvia Bucci, Francesco Cappio Borlino, Barbara Caputo, Tatiana Tommasi},
  booktitle={Winter Conference on Applications of Computer Vision},
  year={2022}
} 
```
