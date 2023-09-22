"""
* Based on https://github.com/dimitymiller/openset_detection/blob/main/mmdetection/associate_data.py
* BSD 3-Clause License
* Copyright (c) 2021, Dimity Miller
* Copyright (c) SafeDNN group 2023
"""

import json
import numpy as np
import tqdm
import os
import argparse

from mmdet.datasets import build_dataset
from mmcv import Config
import mmcv

# iou threshold for object to be associated with detection
iouThresh = 0.5
# score threshold for detection to be considered valid
scoreThresh = 0.2


# function used to calculate IoU between boxes, taken from: https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
def iouCalc(boxes1, boxes2):
    def run(bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou

    return run(boxes1, boxes2)


def associate_train_data(args):
    print('Associating training data')
    allLogits = []
    allLabels = []
    allScores = []
    allIoUs = []
    cfg = Config.fromfile(args.config)
    trainDataset = build_dataset(cfg.data.test)
    trainDatasets = [trainDataset]

    trainDets = mmcv.load(args.outputs)

    allTrainDets = [trainDets]

    for tIdx, trainDataset in tqdm.tqdm(enumerate(trainDatasets)):
        trainDets = allTrainDets[tIdx]
        lenDataset = len(trainDataset)
        for imIdx in range(lenDataset):
            imName = trainDataset.data_infos[imIdx]['filename']

            # detData = np.asarray(trainDets[imName])
            detData = [np.insert(c, c.shape[1], c_id, axis=1) for c_id, c in enumerate(trainDets[imIdx]) if c.shape[0]]

            # detData = np.asarray(trainDets[imIdx])
            # continue if no detections made
            if len(detData) == 0:
                continue
            detData = np.concatenate(detData)
            gtData = trainDataset.get_ann_info(imIdx)

            detPred = detData[:, -1]
            detLogits = detData[:, 5:-1]
            detPredict = np.argmax(detLogits, axis=1)
            mask = detPred == detPredict
            detBoxes = detData[:, :4][mask]
            detScores = detData[:, 4][mask]
            detLogits = detData[:, 5:-1][mask]
            detPredict = np.argmax(detLogits, axis=1)
            # if args.dType == 'retinanet':
            # 	if 'Ensembles' not in args.saveNm:
            # 		newDetLogits = np.log(detLogits/(1-detLogits))
            # 	else:
            # 		newDetLogits = detLogits
            # 		detLogits = 1/(1+np.exp(-newDetLogits))
            # 	mask = np.max(newDetLogits, axis = 1) > 100
            # 	if np.sum(mask) > 0:
            # 		if np.sum(mask) > 1:
            # 			print("ISSUE")
            # 			exit()

            # 		idxes = np.where(detLogits == 1)
            # 		idx1 = idxes[0][0]
            # 		idx2 = idxes[1][0]
            # 		newDetLogits[idx1, idx2] = 25
            # 	detLogits = newDetLogits

            gtBoxes = gtData['bboxes']
            gtLabels = gtData['labels']

            ious = iouCalc(detBoxes, gtBoxes)
            for detIdx, guess in enumerate(detPredict):
                iou = ious[detIdx]
                mask = iou > iouThresh

                trueClasses = gtLabels[mask]
                gtMatches = np.where(guess == trueClasses)[0]

                if len(gtMatches) > 0:
                    allLogits += [detLogits[detIdx].tolist()]
                    allLabels += [int(guess)]
                    allScores += [detScores[detIdx]]

                    maxIoU = np.max(iou[mask][gtMatches])
                    allIoUs += [maxIoU]

    allLogits = list(allLogits)
    allLabels = list(allLabels)
    allScores = list(allScores)
    allIoUs = list(allIoUs)

    trainDict = {'logits': allLogits, 'labels': allLabels, 'scores': allScores, 'ious': allIoUs}

    save_path = args.outputs.split('.')[0] + '_associated.pkl'
    mmcv.dump(trainDict, save_path)


def associate_val_data(args):
    cfg = Config.fromfile(args.config)
    testDataset = build_dataset(cfg.data.test)
    nm = 'VAL'
    print(f'Associating {nm} data')
    allData = {'scores': [], 'type': [], 'logits': [], 'ious': []}
    lenDataset = len(testDataset)

    testDets = mmcv.load(args.outputs)

    for imIdx in tqdm.tqdm(range(lenDataset)):
        gtData = testDataset.get_ann_info(imIdx)

        detData = [np.insert(c, c.shape[1], c_id, axis=1) for c_id, c in enumerate(testDets[imIdx]) if c.shape[0]]

        # continue if no detections made
        if len(detData) == 0:
            continue
        detData = np.concatenate(detData)

        detPred = detData[:, -1]
        detLogits = detData[:, 5:-1]
        detPredict = np.argmax(detLogits, axis=1)
        mask = detPred == detPredict
        detBoxes = detData[:, :4][mask]
        detScores = detData[:, 4][mask]
        detLogits = detData[:, 5:-1][mask]
        detPredict = np.argmax(detLogits, axis=1)

        # if args.dType == 'retinanet':
        # 	if 'Ensembles' not in args.saveNm and 'Ens' not in args.saveNm:
        # 		newDetLogits = np.log(detLogits/(1-detLogits))
        # 	else:
        # 		newDetLogits = detLogits
        # 		detLogits = 1/(1+np.exp(-newDetLogits))

        # 	mask = np.max(newDetLogits, axis = 1) > 25
        # 	if np.sum(mask) > 0:
        # 		if np.sum(mask) > 1:
        # 			print("ISSUE")
        # 			exit()

        # 		idxes = np.where(detLogits == 1)
        # 		idx1 = idxes[0][0]
        # 		idx2 = idxes[1][0]
        # 		newDetLogits[idx1, idx2] = 25
        # 	detLogits = newDetLogits

        # only consider detections that meet the score threshold
        mask = detScores >= scoreThresh
        detScores = detScores[mask]
        detBoxes = detBoxes[mask]
        detLogits = detLogits[mask]
        detPredict = detPredict[mask]

        allDetsIm = {'predictions': detPredict, 'scores': detScores, 'boxes': detBoxes, 'logits': detLogits}
        # associate detections to objects
        allData = associate_detections(allData, allDetsIm, gtData)

    save_path = args.outputs.split('.')[0] + '_associated.pkl'
    mmcv.dump(allData, save_path)


# used to associate detections either as background, known class correctly predicted, known class incorrectly predicted, unknown class
def associate_detections(dataHolder, dets, gt, clsCutoff=21):
    gtBoxes = gt['bboxes']
    gtLabels = gt['labels']
    detPredict = dets['predictions']
    detBoxes = dets['boxes']
    detScores = dets['scores']
    detLogits = dets['logits']

    knownBoxes = gtBoxes[gtLabels < clsCutoff]
    knownLabels = gtLabels[gtLabels < clsCutoff]
    unknownBoxes = gtBoxes[gtLabels > clsCutoff]

    # sort from most confident to least
    sorted_scores = np.sort(detScores)[::-1]
    sorted_idxes = np.argsort(detScores)[::-1]

    detAssociated = [0] * len(detScores)
    gtKnownAssociated = [0] * len(knownBoxes)

    # first, we check if the detection has fallen on a known class
    # if an IoU > iouThresh with a known class --> it is detecting that known class
    if len(knownBoxes) > 0:
        knownIous = iouCalc(detBoxes, knownBoxes)

        for idx, score in enumerate(sorted_scores):
            # if all gt have been associated, move on
            if np.sum(gtKnownAssociated) == len(gtKnownAssociated):
                break

            detIdx = sorted_idxes[idx]
            ious = knownIous[detIdx]
            # sort from greatest to lowest overlap
            sorted_iouIdxs = np.argsort(ious)[::-1]

            for iouIdx in sorted_iouIdxs:
                # check this gt object hasn't already been detected
                if gtKnownAssociated[iouIdx] == 1:
                    continue

                if ious[iouIdx] >= iouThresh:
                    # associating this detection and gt object
                    gtKnownAssociated[iouIdx] = 1
                    detAssociated[detIdx] = 1

                    gtLbl = knownLabels[iouIdx]
                    dataHolder['ious'] += [ious[iouIdx]]
                    # known class was classified correctly
                    if detPredict[detIdx] == gtLbl:
                        dataHolder['scores'] += [score]
                        dataHolder['logits'] += [list(detLogits[detIdx])]
                        dataHolder['type'] += [0]
                    # known class was misclassified
                    else:
                        dataHolder['scores'] += [score]
                        dataHolder['logits'] += [list(detLogits[detIdx])]
                        dataHolder['type'] += [1]
                    break
                else:
                    # doesn't have an iou greater than 0.5 with anything
                    break

    # all detections have been associated
    if np.sum(detAssociated) == len(detAssociated):
        return dataHolder

    ### Next, check if the detection overlaps an ignored gt known object - these detections are ignored
    # also check ignored gt known objects
    if len(gt['bboxes_ignore']) > 0:
        igBoxes = gt['bboxes_ignore']
        igIous = iouCalc(detBoxes, igBoxes)
        for idx, score in enumerate(sorted_scores):
            detIdx = sorted_idxes[idx]
            if detAssociated[detIdx] == 1:
                continue

            ious = igIous[detIdx]

            # sort from greatest to lowest overlap
            sorted_iouIdxs = np.argsort(ious)[::-1]

            for iouIdx in sorted_iouIdxs:
                if ious[iouIdx] >= iouThresh:
                    # associating this detection and gt object
                    detAssociated[detIdx] = 1
                break

    # all detections have been associated
    if np.sum(detAssociated) == len(detAssociated):
        return dataHolder

    # if an IoU > 0.5 with an unknown class (but not any known classes) --> it is detecting the unknown class
    newDetAssociated = detAssociated
    if len(unknownBoxes) > 0:
        unknownIous = iouCalc(detBoxes, unknownBoxes)

        for idx, score in enumerate(sorted_scores):
            detIdx = sorted_idxes[idx]

            # if the detection has already been associated, skip it
            if detAssociated[detIdx] == 1:
                continue

            ious = unknownIous[detIdx]

            # sort from greatest to lowest overlap
            sorted_iouIdxs = np.argsort(ious)[::-1]
            for iouIdx in sorted_iouIdxs:
                if ious[iouIdx] >= iouThresh:
                    newDetAssociated[detIdx] = 1

                    dataHolder['scores'] += [score]
                    dataHolder['logits'] += [list(detLogits[detIdx])]
                    dataHolder['type'] += [2]
                    dataHolder['ious'] += [ious[iouIdx]]
                    break
                else:
                    # no overlap greater than 0.5
                    break

    detAssociated = newDetAssociated

    if np.sum(detAssociated) == len(detAssociated):
        return dataHolder

    # otherwise remaining detections are all background detections
    for detIdx, assoc in enumerate(detAssociated):
        if not assoc:
            score = detScores[detIdx]
            dataHolder['scores'] += [score]
            dataHolder['type'] += [3]
            dataHolder['logits'] += [list(detLogits[detIdx])]
            dataHolder['ious'] += [0]
            detAssociated[detIdx] = 1

    if np.sum(detAssociated) != len(detAssociated):
        print("THERE IS A BIG ASSOCIATION PROBLEM")
        exit()

    return dataHolder


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute metrics')
    parser.add_argument('outputs', help='model outputs')
    parser.add_argument('config', help='test config')
    parser.add_argument('-train', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.train:
        associate_train_data(args)
    else:
        associate_val_data(args)


if __name__ == '__main__':
    main()
