import time
import itertools
import math

import diffdist.functional as distops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import models.transform_layers as TL
from utils.temperature_scaling import _ECELoss
from utils.utils import AverageMeter, set_random_seed, normalize
from tqdm import tqdm
import sys
from utils.dist_utils import synchronize, all_gather
np.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

def get_features(P, model, test_loaders, normalize=True, layer="penultimate", distributed=False):
    model.eval()

    # we have to use penultimate layer as some models (like baseline naive) do not have the simclr projection head
    feats = []
    labels = []
    ids = []
    out_dict = {}
    
    for loader in test_loaders:
        for batch in tqdm(loader):
            images, lbl, img_id = batch
            labels.append(lbl.item())
            images = images.to(device)
            with torch.no_grad():
                _, output_aux = model(images, penultimate=True,simclr=True)
            feat = output_aux[layer]
            if normalize:
                feat = feat/feat.norm()
            cpu_feat = feat.to(cpu_device)
            feats.append(cpu_feat)
            out_dict[img_id.item()] = {'feat': cpu_feat, 'label': lbl.item()}

    feats=torch.cat(feats)

    if distributed:
        all_dicts = all_gather(out_dict)
        predictions = {}
        for dic in all_dicts:
            predictions.update(dic)
        image_ids = list(sorted(predictions.keys()))

        all_feats = []
        all_labels = []
        for img_id in image_ids:
            all_feats.append(predictions[img_id]['feat'])
            all_labels.append(predictions[img_id]['label'])

        all_feats = torch.cat(all_feats)
        feats = all_feats

        all_labels = torch.tensor(all_labels)
        labels = all_labels.numpy()
        synchronize()
    return feats, np.array(labels)

def rescale_cosine_similarity(similarities):
    return (similarities+1)/2

def compute_source_prototypes(P, model, source_loader, eval_layer, distributed=False):
    
    # we extract features for all source samples
    source_feats, source_gt_labels = get_features(P,model, [source_loader], layer=eval_layer, normalize=True,distributed=distributed)

    # we compute prototypes for source classes
    labels_set = set(source_gt_labels.tolist())
    prototypes = {}
    for label in labels_set:
        lbl_mask = source_gt_labels == label
        source_this_label = source_feats[lbl_mask]
        prototypes[label] = source_this_label.mean(dim=0)

    # let's move prototypes to the hypersphere. We also compute average cluster compactness
    # we will need a threshold which will be based on this value 
    hyp_prototypes = []
    cls_compactness_tot = 0
    for cls in prototypes.keys():
        cls_prototype = prototypes[cls]
        norm = cls_prototype.norm()
        hyp_prototype = cls_prototype/norm
        hyp_prototypes.append(hyp_prototype)    
        
        source_this_label = source_feats[source_gt_labels == cls]
        tot_similarities = 0
        for src_feat in source_this_label:
            similarity = (src_feat*hyp_prototype).sum()
            similarity = rescale_cosine_similarity(similarity)
            tot_similarities += similarity.item()
        avg_cls_similarity = tot_similarities/len(source_this_label)
        cls_compactness_tot += avg_cls_similarity
    cls_compactness_avg = cls_compactness_tot/len(labels_set)

    hyp_prototypes = np.stack(hyp_prototypes)

    # we also compute average distance between nearest prototypes
    topk_sims = np.zeros((len(hyp_prototypes)))
    for idx, hyp_prt in enumerate(hyp_prototypes):
        similarities = (hyp_prt*hyp_prototypes).sum(1)
        similarities = rescale_cosine_similarity(similarities)
        similarities.sort()
        topk_val = similarities[-2]
        topk_sims[idx] = topk_val

    return source_feats, hyp_prototypes, cls_compactness_avg, topk_sims.mean()

def compute_threshold_multiplier(avg_compactness, avg_cls_dist):
    y = (1-avg_cls_dist)
    x = (1-avg_compactness)
    z = y/(2*x)
    return math.log(z) + 1

def compute_confident_known_mask(P, model, source_loader, target_loader, logger, eval_layer="simclr"):

    distributed = isinstance(source_loader.sampler, torch.utils.data.distributed.DistributedSampler)
    model.eval()

    # compute source prototypes and compactness for known classes
    _, hyp_prototypes, cls_compactness_avg, avg_min_sim = compute_source_prototypes(P, model, source_loader, eval_layer, distributed=distributed)
    
    # now we get features for target samples 
    target_feats, target_gt_labels = get_features(P, model, [target_loader], layer=eval_layer, normalize=True, distributed=distributed)

    # for each target sample we measure distance from nearest prototype. If ditance is lower than a threshold we select 
    #this sample and the prototype label as pseudo label 

    threshold_multiplier = compute_threshold_multiplier(cls_compactness_avg, avg_min_sim)/2
    known_threshold = (1-cls_compactness_avg)*threshold_multiplier

    known_mask = np.zeros((len(target_feats)), dtype=np.bool)
    known_pseudo_labels = P.n_classes*np.ones((len(target_feats)), dtype=np.uint32)
    known_gt_labels = P.n_classes*np.ones((len(target_feats)), dtype=np.uint32)
    for idx, (tgt_feat, tgt_gt_label) in enumerate(zip(target_feats, target_gt_labels)): 

        similarities = (tgt_feat*hyp_prototypes).sum(dim=1)
        similarities = rescale_cosine_similarity(similarities)

        highest = similarities.max()
        cls_id = similarities.argmax()

        # check whether it is near enough to nearest prototype to be considered known
        if highest >= (1 - known_threshold):
            known_mask[idx] = True
            known_pseudo_labels[idx] = cls_id

        known_gt_labels[idx] = tgt_gt_label.item()
        if tgt_gt_label > P.n_classes: # unknown class
            known_gt_labels[idx] = P.n_classes

    return known_mask, known_pseudo_labels, known_gt_labels

def openset_eval(P, model, source_loader, target_loader, logger, eval_layer="simclr"):

    distributed = isinstance(source_loader.sampler, torch.utils.data.distributed.DistributedSampler)

    model.eval()
    source_feats, hyp_prototypes, cls_compactness_avg, avg_min_sim = compute_source_prototypes(P, model, source_loader, eval_layer, distributed=distributed)
    print(f"Class compactness avg: {cls_compactness_avg}, avg_min_sim: {avg_min_sim}")
    
    # now we get features for target samples 
    target_feats, target_gt_labels = get_features(P, model, [target_loader], layer=eval_layer, normalize=True, distributed=distributed)

    # define counters we need for openset eval
    samples_per_class = np.zeros(P.n_classes + 1) #46
    correct_pred_per_class = np.zeros(P.n_classes + 1) #46

    # for each target sample we have to make a predictions. So we compare it with all the prototypes. 
    # the sample is associated with the class of the nearest prototype if its similarity with this prototype 
    # is higher than a certain threshold
    threshold_multiplier = compute_threshold_multiplier(cls_compactness_avg, avg_min_sim)
    normality_threshold = (1-cls_compactness_avg)*threshold_multiplier
    for tgt_feat, tgt_gt_label in zip(target_feats, target_gt_labels): 

        similarities = (tgt_feat*hyp_prototypes).sum(dim=1)
        similarities = rescale_cosine_similarity(similarities)

        highest = similarities.max()
        cls_id = similarities.argmax()

        # check whether it is near enough to nearest prototype to be considered known
        if highest < (1 - normality_threshold):
            # this is the unknown cls_id
            cls_id = P.n_classes

        # accumulate prediction
        if tgt_gt_label > P.n_classes:
            tgt_gt_label = P.n_classes
        samples_per_class[tgt_gt_label] += 1
        if cls_id == tgt_gt_label:
            correct_pred_per_class[cls_id] += 1

    acc_os_star = np.mean(correct_pred_per_class[0:len(correct_pred_per_class)-1] / samples_per_class[0:len(correct_pred_per_class)-1])
    acc_unknown = (correct_pred_per_class[-1] / samples_per_class[-1])
    acc_hos = 2 * (acc_os_star * acc_unknown) / (acc_os_star + acc_unknown)
    acc_os = np.mean(correct_pred_per_class/ samples_per_class)

    acc_os *= 100
    acc_os_star *= 100
    acc_unknown *= 100
    acc_hos *= 100

    x = (1 - cls_compactness_avg)
    y = (1 - avg_min_sim)

    if logger is None: 
        print('[OS %6f]' %(acc_os))
        print('[OS* %6f]' % (acc_os_star))
        print('[UNK %6f]' % (acc_unknown))
        print('[HOS %6f]' % (acc_hos))
    else:
        logger.log('[OS %6f]' %(acc_os))
        logger.log('[OS* %6f]' % (acc_os_star))
        logger.log('[UNK %6f]' % (acc_unknown))
        logger.log('[HOS %6f]' % (acc_hos))

