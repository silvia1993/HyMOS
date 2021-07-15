from utils.utils import Logger
from utils.utils import save_checkpoint, save_checkpoint_name
from utils.utils import AverageMeter
from evals.evals import compute_confident_known_mask, openset_eval
from datasets.datasets import get_dataset_2
import time
import sys
from sklearn.metrics import accuracy_score

from common.train import *
import torch
import resource

from training.sup import setup
# setup training routine
train, fname = setup(P)

logger = Logger(fname, ask=not resume, local_rank=P.local_rank)
logger.log(P)
logger.log(model)

# Run experiments
losses = dict()
losses['sim'] = AverageMeter()
losses['norm'] = AverageMeter()
losses['time'] = AverageMeter()
check = time.time()

P.style_iter = iter(P.style_loader)

data_iter = iter(train_loader)

for its in range(start_its, P.iterations + 1):
    model.train()

    kwargs = {}
    kwargs['simclr_aug'] = simclr_aug

    # train one iteration
    sim_loss, simclr_norm, train_loader, data_iter = train(P, its, model, criterion, optimizer, scheduler_warmup, train_loader, data_iter, logger=logger, **kwargs)

    model.eval()

    if its == P.iterations and P.local_rank == 0:
        if P.multi_gpu:
            save_states = model.module.state_dict()
        else:
            save_states = model.state_dict()
        save_checkpoint(its, save_states, optimizer.state_dict(), logger.logdir)
        save_checkpoint_name(save_states, logger.logdir, f"{its}")
        logger.log("[Saving checkpoint]")

    # log
    losses['sim'].update(sim_loss, 1)
    losses['norm'].update(simclr_norm, 1)
    losses['time'].update(time.time()-check, 1)

    check = time.time()
    if its%10 == 0 :
        eta_sec = (P.iterations - its)*losses['time'].average
        hour = eta_sec // 3600
        eta_sec = eta_sec % 3600
        min = eta_sec // 60
        eta_sec = eta_sec % 60

        lr = optimizer.param_groups[0]['lr']

        logger.log('[Iteration %3d] [Loss_Sim %5.2f] [SimclrNorm %f] [LR %f] [Avg time %.2fs] [ETA %dh%dm%ds]' % (its, losses['sim'].average, 
            losses['norm'].average, lr, losses['time'].average, hour, min, eta_sec))

        losses['sim'] = AverageMeter()
        losses['norm'] = AverageMeter()

    if P.iterative and its in P.its_breakpoints:
        logger.log(f"Reached breakpoint iteration {its}. Start computing pseudo labels for new known samples")

        known_mask, known_pseudo_labels, known_gt_labels = compute_confident_known_mask(P, model, source_test_loader, target_test_loader, logger=None)

        acc = accuracy_score(known_gt_labels[known_mask], known_pseudo_labels[known_mask])
        logger.log("Selected {} target samples as known. Classification accuracy for selected samples: {:.4f}".format(len(known_mask.nonzero()[0]), acc))
        del known_gt_labels
        
        if len(known_mask.nonzero()[0]) > 0:
            train_sets, n_classes = get_dataset_2(P, train=True, target_known_mask=known_mask, target_known_pseudo_labels = known_pseudo_labels)
            whole_source = ConcatDataset([train_sets[idx] for idx in range(n_classes)])
            my_sampler = BalancedMultiSourceRandomSampler(whole_source, P.batch_p, P.local_rank, P.n_gpus)
            print(f"{P.local_rank} sampler_size: {len(my_sampler)}")
            loader_kwargs = {'pin_memory': False, 'num_workers': 1, 'drop_last':True}
            train_loader = DataLoader(whole_source, sampler=my_sampler, batch_size=single_GPU_batch_size, **loader_kwargs)
            data_iter = iter(train_loader)


    if its%5000 == 0 or its == P.iterations:

        logger.log("Running openset eval")
        openset_eval(P, model, source_test_loader, target_test_loader, logger=logger)

sys.exit(0)
