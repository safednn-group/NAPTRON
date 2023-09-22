"""
* Based on https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
* Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
* Copyright (c) SafeDNN group 2023
"""
import numpy as np
import datetime
import time
from collections import defaultdict
from pycocotools.cocoeval import COCOeval


class CocoEvalOOD(COCOeval):
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]

    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', ignore_ood_gts=False, known_category_names=None,
                 class_agnostic=False, localization_uncertainty=False):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        *SA*
        usage: localization - class_agnostic=True, to obtain stats only for certain class(-es) eg.
        ID classes ignore_ood_gts=True, known_category_names=list(ID classes)
        ID stats --||-- class agnostic=False
        '''
        super().__init__(cocoGt, cocoDt, iouType)
        self.ignore_ood_gts = ignore_ood_gts
        self.params.useCats = int(not class_agnostic)
        self.known_category_ids = None
        self.localization_uncertainty = localization_uncertainty
        if not cocoGt is None:
            if known_category_names:
                self.known_category_ids = sorted(cocoGt.getCatIds(catNms=known_category_names))
                assert self.known_category_ids

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        ***SAFEDNN MODIFICATION***
        allow indentifying OOD groundtruth
        :return: None
        '''

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
            if self.known_category_ids:
                gt['ood'] = gt['category_id'] not in self.known_category_ids
                if self.ignore_ood_gts:
                    gt['ignore'] = gt['category_id'] not in self.known_category_ids

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        ***SAFEDNN MODIFICATION***
        allow indentifying of detected bboxes that match OOD groundtruth
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        gtm_ood = np.zeros((T, G))
        dtm = np.zeros((T, D))
        dtm_ood = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        gt_ood = np.array([g['ood'] if 'ood' in g and not g['_ignore'] else 0 for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    dtm_ood[tind, dind] = gt[m]['id'] if 'ood' in gt[m] and gt[m]['ood'] else 0
                    gtm[tind, m] = d['id']
                    gtm_ood[tind, m] = d['id'] if 'ood' in gt[m] and gt[m]['ood'] else 0
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'dtUncertaintyScores': [d['uncertainty_score'] if 'uncertainty_score' in d else 0. for d in dt],
            'dtBBoxId': [d['bbox_id'] if 'bbox_id' in d else -1 for d in dt],
            'dtLabel': [d['label'] if 'label' in d else -1 for d in dt],
            'dtImgId': [d['img_id'] if 'img_id' in d else -1 for d in dt],
            'gtIgnore': gtIg,
            'gtOOD': gt_ood,
            'dtIgnore': dtIg,
            'dtMatchesOOD': dtm_ood,
            'gtMatchesOOD': gtm_ood,
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        ***SAFEDNN MODIFICATION***
        allow accumulating certainty scores of false positives; true positives; and OOD detections
        fix recall computation

        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        detected_objects = -np.ones((T, K, A, M))
        detected_objects_ood = -np.ones((T, K, A, M))
        objects = -np.ones((K, A, M))
        objects_ood = -np.ones((K, A, M))
        scores = -np.ones((T, R, K, A, M))
        uncertainty_scores = dict()
        for t in range(T):
            uncertainty_scores[t] = dict()

        bbox_ids = dict()
        for t in range(T):
            bbox_ids[t] = dict()

        bbox_label = dict()
        for t in range(T):
            bbox_label[t] = dict()

        bbox_img_id = dict()
        for t in range(T):
            bbox_img_id[t] = dict()

        logit_scores = dict()
        for t in range(T):
            logit_scores[t] = dict()


        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
                    dtUncertaintyScores = np.concatenate([e['dtUncertaintyScores'][0:maxDet] for e in E])
                    dtBBoxId = np.concatenate([e['dtBBoxId'][0:maxDet] for e in E])
                    dtLabel = np.concatenate([e['dtLabel'][0:maxDet] for e in E])
                    dtImgId = np.concatenate([e['dtImgId'][0:maxDet] for e in E])
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]
                    dtUncertaintyScoresSorted = dtUncertaintyScores[inds]
                    dtBBoxId = dtBBoxId[inds]
                    dtLabel = dtLabel[inds]
                    dtImgId = dtImgId[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtm = np.concatenate([e['gtMatches'][:, 0:maxDet] for e in E], axis=1)
                    gtm_ood = np.concatenate([e['gtMatchesOOD'][:, 0:maxDet] for e in E], axis=1)
                    dtm_ood = np.concatenate([e['dtMatchesOOD'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gt_ood = np.concatenate([e['gtOOD'] for e in E])
                    n_ood = np.count_nonzero(gt_ood != 0)
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)

                    objects_ood[k, a, m] = n_ood
                    objects[k, a, m] = npig

                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    tps_g = np.logical_and(gtm, np.logical_not(gtIg))
                    ood_tps_g = np.logical_and(tps_g, gtm_ood)
                    ood_tps = np.logical_and(tps, dtm_ood)
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    tp_sum_g = np.cumsum(tps_g, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    ood_tps_g_sum = np.cumsum(ood_tps_g, axis=1).astype(dtype=float)

                    if m == len(m_list) - 1:
                        for t in range(T):
                            if _pe.useCats == 0 and self.known_category_ids:
                                # uncertainty_scores[t, k, a, m] = np.nanmean(dtScoresSorted[ood_tps[t, :]])
                                u_s = dtUncertaintyScoresSorted[ood_tps[t, :]]
                                u_s = u_s[~np.isnan(u_s)]
                                uncertainty_scores[t][a] = u_s

                                b_s = dtBBoxId[ood_tps[t, :]]
                                i_s = dtImgId[ood_tps[t, :]]
                                c_s = dtLabel[ood_tps[t, :]]
                                bbox_ids[t][a] = b_s
                                bbox_img_id[t][a] = i_s
                                bbox_label[t][a] = c_s

                                l_s = dtScoresSorted[ood_tps[t, :]]
                                l_s = l_s[~np.isnan(l_s)]
                                logit_scores[t][a] = l_s
                                detected_objects_ood[t, k, a, m] = ood_tps_g_sum[t, -1]
                            elif _pe.useCats == 0 and self.localization_uncertainty:
                                u_s = dtUncertaintyScoresSorted[tps[t, :]]
                                u_s = u_s[~np.isnan(u_s)]
                                uncertainty_scores[t][a] = u_s

                                b_s = dtBBoxId[tps[t, :]]
                                i_s = dtImgId[tps[t, :]]
                                c_s = dtLabel[tps[t, :]]
                                bbox_ids[t][a] = b_s
                                bbox_img_id[t][a] = i_s
                                bbox_label[t][a] = c_s


                                l_s = dtScoresSorted[tps[t, :]]
                                l_s = l_s[~np.isnan(l_s)]
                                logit_scores[t][a] = l_s
                                detected_objects[t, k, a, m] = tp_sum_g[t, -1]
                                # uncertainty_scores[t, k, a, m] = np.nanmean(dtScoresSorted[fps[t, :]])
                            elif _pe.useCats == 0:
                                u_s = dtUncertaintyScoresSorted[fps[t, :]]
                                u_s = u_s[~np.isnan(u_s)]
                                uncertainty_scores[t][a] = u_s

                                b_s = dtBBoxId[fps[t, :]]
                                i_s = dtImgId[fps[t, :]]
                                c_s = dtLabel[fps[t, :]]
                                bbox_ids[t][a] = b_s
                                bbox_img_id[t][a] = i_s
                                bbox_label[t][a] = c_s

                                l_s = dtScoresSorted[fps[t, :]]
                                l_s = l_s[~np.isnan(l_s)]
                                logit_scores[t][a] = l_s
                            else:
                                u_s = dtUncertaintyScoresSorted[tps[t, :]]
                                u_s = u_s[~np.isnan(u_s)]

                                b_s = dtBBoxId[tps[t, :]]
                                i_s = dtImgId[tps[t, :]]
                                c_s = dtLabel[tps[t, :]]

                                l_s = dtScoresSorted[tps[t, :]]
                                l_s = l_s[~np.isnan(l_s)]

                                detected_objects[t, k, a, m] = tp_sum_g[t, 1]
                                if isinstance(uncertainty_scores[t].get(a), np.ndarray):
                                    uncertainty_scores[t][a] = np.concatenate((uncertainty_scores[t].get(a), u_s))

                                    bbox_ids[t][a] = np.concatenate((bbox_ids[t].get(a), b_s))
                                    bbox_img_id[t][a] = np.concatenate((bbox_img_id[t].get(a), i_s))
                                    bbox_label[t][a] = np.concatenate((bbox_label[t].get(a), c_s))
                                    logit_scores[t][a] = np.concatenate((logit_scores[t].get(a), l_s))
                                else:
                                    uncertainty_scores[t][a] = u_s
                                    bbox_ids[t][a] = b_s
                                    bbox_img_id[t][a] = i_s
                                    bbox_label[t][a] = c_s

                                    logit_scores[t][a] = l_s
                                # uncertainty_scores[t, k, a, m] = np.nanmean(dtScoresSorted[tps[t, :]])

                    for t, (tp, fp, tp_g) in enumerate(zip(tp_sum, fp_sum, tp_sum_g)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))
                        # fixed recall computation
                        tp_g = np.array(tp_g)
                        rc_g = tp_g / npig
                        if nd:
                            # recall[t, k, a, m] = rc[-1]
                            recall[t, k, a, m] = rc_g[-1]
                        else:
                            recall[t, k, a, m] = 0
                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist();
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
            'uncertainty_scores': uncertainty_scores,
            'bbox_ids': bbox_ids,
            'bbox_img_ids': bbox_img_id,
            'bbox_labels': bbox_label,
            'logit_scores': logit_scores,
            'detected_objects_count': detected_objects,
            'detected_ood_objects_count': detected_objects_ood,
            'objects_count': objects,
            'ood_objects_count': objects_ood,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self, uncertainty=False):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if uncertainty:
                u = self.eval['uncertainty_scores']
                l = self.eval['logit_scores']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    u = u[t[0]][aind[0]]
                    mean_u = np.nanmean(u)
                    l = l[t[0]][aind[0]]
                    mean_l = np.nanmean(l)
                else:
                    means = []
                    for t in u:
                        means.append(np.nanmean(u[t][aind[0]]))
                    mean_u = np.nanmean(means)
                    means_l = []
                    for t in l:
                        means_l.append(np.nanmean(l[t][aind[0]]))
                    mean_l = np.nanmean(means_l)
                print(iStr.format('Mean uncertainty', '', iouStr, areaRng, maxDets, mean_u))
                print(iStr.format('Mean logit uncertainty', '', iouStr, areaRng, maxDets, mean_l))
                return mean_l
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        def _summarizeUnc():
            stats = np.zeros((1,))
            stats[0] = _summarize()
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        elif uncertainty:
            summarize = _summarizeUnc()
        self.stats = summarize()

    def __str__(self):
        self.summarize()
