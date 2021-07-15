from common.eval import *
from tqdm import tqdm
import numpy as np

model.eval()


if P.mode == "openset_eval":
    from evals.evals import openset_eval

    with torch.no_grad():
        openset_eval(P, model, source_test_loader, target_test_loader, logger=None)

elif P.mode == "eval_known_selection":
    from evals.evals import compute_confident_known_mask

    with torch.no_grad():
        known_mask, known_pseudo_labels, known_gt_labels = compute_confident_known_mask(P, model, source_test_loader, target_test_loader, logger=None)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(known_gt_labels[known_mask], known_pseudo_labels[known_mask])

    gt_known = 0

    known_gt_lbls = known_gt_labels[known_mask]
    number_real_known = len(known_gt_lbls[known_gt_lbls < P.n_classes])
    percentage_true_known = number_real_known/len(known_mask.nonzero()[0])
    print("Selected {} target samples as known. Classification accuracy: {:.4f}. Percentage of gt known: {:.4f}".format(len(known_mask.nonzero()[0]), acc, percentage_true_known))
else:
    raise NotImplementedError()



