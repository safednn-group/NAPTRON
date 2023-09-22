from sklearn.metrics import roc_curve, auc
import numpy as np


def auroc(fps, tps, reverse=False):
    if reverse:
        fps, tps = tps, fps
    labels = np.concatenate((np.zeros_like(fps), np.ones_like(tps)))
    preds = np.concatenate((fps, tps))
    fpr, tpr, _ = roc_curve(labels, preds)
    return auc(fpr, tpr)


def fpr_at_95_tpr(fps, tps, reverse=False):
    if reverse:
        fps, tps = tps, fps
    labels = np.concatenate((np.zeros_like(fps), np.ones_like(tps)))
    preds = np.concatenate((fps, tps))
    fpr, tpr, _ = roc_curve(labels, preds)
    if all(tpr < 0.95):
        return 0
    elif all(tpr >= 0.95):
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        return np.interp(0.95, tpr, fpr)