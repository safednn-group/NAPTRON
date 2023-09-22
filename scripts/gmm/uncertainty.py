"""
* Based on https://github.com/dimitymiller/openset_detection/blob/main/mmdetection/get_results.py
* BSD 3-Clause License
* Copyright (c) 2021, Dimity Miller
* Copyright (c) SafeDNN group 2023
"""

import numpy as np
import argparse
import tqdm
import mmcv
import os
from pathlib import Path

from utils import fit_gmms, gmm_uncertainty

num_classes = 20
scoreThresh = 0.7
iouThresh = 0.6


def compute_auroc(tps, fps):
    from sklearn.metrics import roc_curve, auc
    labels = np.concatenate((np.zeros_like(fps), np.ones_like(tps)))
    preds = np.concatenate((fps, tps))
    fpr, tpr, _ = roc_curve(labels, preds)
    return auc(fpr, tpr)


def get_gmms(args):
    if args.use_fit and os.path.isfile(args.gmms):
        return mmcv.load(args.gmms)
    trainData = mmcv.load(args.train_data)

    valData = mmcv.load(args.val_data)

    trainLogits = np.array(trainData['logits'])
    trainLabels = np.array(trainData['labels'])
    trainScores = np.array(trainData['scores'])
    trainIoUs = np.array(trainData['ious'])

    valLogits = np.array(valData['logits'])
    valTypes = np.array(valData['type'])

    # fit distance-based models

    # find the number of components that gives best performance on validation data, unless numComp argument specified

    allAurocs = []
    nComps = [nI for nI in range(3, 16)]
    print('Finding optimal component number for the GMM')
    for nComp in tqdm.tqdm(nComps, total=len(nComps)):
        gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, scoreThresh, iouThresh,
                        num_classes, components=nComp)

        gmmScores = gmm_uncertainty(valLogits, gmms)
        valTP = gmmScores[valTypes == 0]
        valFP = gmmScores[valTypes == 1]
        auroc = compute_auroc(valTP, valFP)
        allAurocs += [auroc]

    allAurocs = np.array(allAurocs)
    bestIdx = np.argmax(allAurocs)
    preferredComp = nComps[bestIdx]

    print(f'Testing GMM with {preferredComp} optimal components')
    gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, scoreThresh, iouThresh,
                    num_classes, components=preferredComp)

    if args.gmms:
        path = Path(args.gmms)
        if not os.path.exists(str(path.parent)):
            os.makedirs(str(path.parent))
        mmcv.dump(gmms, args.gmms)
    return gmms


def append_uncertainty(bboxes, gmms):
    bbox_counter = 0
    for i in tqdm.tqdm(range(len(bboxes))):
        for j in range(len(bboxes[i])):
            num_bboxes = bboxes[i][j].shape[0]
            if num_bboxes:
                logits = bboxes[i][j][:, 5:]
                uncertainty = gmm_uncertainty(logits, gmms)
                bboxes[i][j] = np.column_stack((bboxes[i][j][:, :5], uncertainty))
                bbox_counter += num_bboxes
    return bboxes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute metrics')
    parser.add_argument('train_data', help='associated train data')
    parser.add_argument('val_data', help='associated val data')
    parser.add_argument('test_data', help='raw test data')
    parser.add_argument('-use-fit', action='store_true', help='use fit gmms')
    parser.add_argument('gmms', help='fit gmms')
    args = parser.parse_args()
    return args

def load_outputs(outputs_file):
    import os
    path = os.path.dirname(outputs_file)
    all_files = [os.path.join(path, f) for f in os.listdir(path) if
                 os.path.isfile(os.path.join(path, f))]
    patched = []
    for file in sorted(all_files):
        if file.find(os.path.basename(outputs_file.split('.')[0])) >= 0 and file.find('chunk') >=0:
            chunk = mmcv.load(file)
            patched.extend(chunk)
    return patched if patched else mmcv.load(outputs_file)

def main():
    args = parse_args()
    gmms = get_gmms(args)
    outputs = load_outputs(args.test_data)
    bboxes = append_uncertainty(outputs, gmms)
    save_path = args.test_data.split('.')[0] + '_uncertainty.pkl'
    mmcv.dump(bboxes, save_path)


if __name__ == '__main__':
    main()
