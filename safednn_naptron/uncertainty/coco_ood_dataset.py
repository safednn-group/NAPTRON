"""
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
import contextlib
import io
import itertools
import logging
from collections import OrderedDict

import mmcv
import pandas as pd
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.datasets import CocoDataset, DATASETS
from mmdet.datasets.api_wrappers import COCO
from .coco_eval_ood import CocoEvalOOD
from safednn.utils.metrics import auroc, fpr_at_95_tpr


@DATASETS.register_module()
class CocoOODDataset(CocoDataset):

    # mapping of cocoEval.stats
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@100': 6,
        'AR@300': 7,
        'AR@1000': 8,
        'AR_s@1000': 9,
        'AR_m@1000': 10,
        'AR_l@1000': 11
    }

    def __init__(self,
                 is_class_agnostic=False,
                 filter_unknown_imgs=False,
                 **kwargs):
        self.is_class_agnostic = is_class_agnostic
        self.filter_unknown_imgs = filter_unknown_imgs
        self.KNOWN_CLASSES = None
        super(CocoOODDataset, self).__init__(**kwargs)
        if filter_unknown_imgs:
            valid_inds = self._filter_imgs()
            if self.is_class_agnostic:
                self.CLASSES = ('known',)
            self.data_infos = [self.data_infos[i] for i in valid_inds]

    def get_cats(self):
        cats = self.coco.dataset['categories']
        cats = cats if len(self.CLASSES) == 0 else [cat for cat in cats if cat['name'] in self.CLASSES]
        return cats

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        if self.is_class_agnostic:
            self.cat2label = {cat_id: 0 for cat_id in self.cat_ids}
            if not self.filter_unknown_imgs:
                self.CLASSES = ('known',)
        else:
            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.img_ids = self.coco.get_img_ids()
        self.label2cat = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        if self.filter_unknown_imgs:
            for cat_id in self.coco.cats:
                if self.coco.cats[cat_id]['name'] not in self.CLASSES:
                    ids_in_cat -= set(self.coco.cat_img_map[cat_id])

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['bbox_id'] = i
                    data['img_id'] = idx
                    data['label'] = label
                    if bboxes[i].shape[0] == 6:
                        data['uncertainty_score'] = float(bboxes[i][5])
                    data['category_id'] = self.label2cat[label]
                    json_results.append(data)
        return json_results

    def evaluate_ood(self,
                     results,
                     metric='all',
                     logger=None,
                     jsonfile_prefix=None,
                     classwise=False,
                     metric_items=None,
                     dump_certainties=False):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        if metric == 'all':
            metrics = ['localization', 'localization_known', 'localization_unknown', 'bbox_known', 'uncertainty']
        else:
            metrics = metric if isinstance(metric, list) else [metric]
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if 'bbox' not in result_files:
                raise KeyError(f'{metric} bbox is not in results')
            try:
                predictions = mmcv.load(result_files['bbox'])
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in self.coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric.startswith('localization'):
                eval_results = self._evaluate_localization(predictions, metric, logger, eval_results)
            elif metric == 'bbox_known':
                eval_results = self._evaluate_bbox_known(predictions, metric, logger, eval_results, classwise)
            else:
                eval_results = self._evaluate_uncertainty(predictions, eval_results, logger, dump_certainties)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def _evaluate_localization(self, predictions, metric, logger, eval_results):
        cocoGt = self.coco
        cocoDt = cocoGt.loadRes(predictions)
        iou_type = 'bbox'
        proposal_nums = (100, 300, 1000)
        metric_items = [
            'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
            'AR_m@1000', 'AR_l@1000'
        ]
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric == 'localization':
            cocoEval = CocoEvalOOD(cocoGt, cocoDt, iou_type, class_agnostic=True)
        elif metric == 'localization_known':
            cocoEval = CocoEvalOOD(cocoGt, cocoDt, iou_type, class_agnostic=True, ignore_ood_gts=True,
                                   known_category_names=self.KNOWN_CLASSES)
        else:
            if not set(self.CLASSES) - set(self.KNOWN_CLASSES):
                raise ValueError(
                    f'Cannot perform metric localization_unknown, because there are no OOD classes')
            cocoEval = CocoEvalOOD(cocoGt, cocoDt, iou_type, class_agnostic=True, ignore_ood_gts=True,
                                   known_category_names=(set(self.CLASSES) - set(self.KNOWN_CLASSES)))
        cocoEval = self._eval(cocoEval, proposal_nums, iou_thrs, logger)

        for item in metric_items:
            key = f'localization_{metric}_{item}'
            val = float(
                f'{cocoEval.stats[self.coco_metric_names[item]]:.3f}')
            eval_results[key] = val

        ap = cocoEval.stats[:6]
        eval_results[f'localization_{metric}_mAP_copypaste'] = (
            f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
            f'{ap[4]:.3f} {ap[5]:.3f}')
        return eval_results

    def _evaluate_bbox_known(self, predictions, metric, logger, eval_results, classwise):
        cocoGt = self.coco
        cocoDt = cocoGt.loadRes(predictions)
        iou_type = 'bbox'
        proposal_nums = (100, 300, 1000)
        metric_items = [
            'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
            'AR_m@1000', 'AR_l@1000'
        ]
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        cocoEval = CocoEvalOOD(cocoGt, cocoDt, iou_type, ignore_ood_gts=True,
                               known_category_names=self.KNOWN_CLASSES)
        cocoEval = self._eval(cocoEval, proposal_nums, iou_thrs, logger)
        if classwise:  # Compute per-category AP
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = cocoEval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            assert len(self.cat_ids) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(self.cat_ids):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = self.coco.loadCats(catId)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                results_per_category.append(
                    (f'{nm["name"]}', f'{float(ap):0.3f}'))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print_log('\n' + table.table, logger=logger)

        for metric_item in metric_items:
            key = f'bbox_known_{metric}_{metric_item}'
            val = float(
                f'{cocoEval.stats[self.coco_metric_names[metric_item]]:.3f}'
            )
            eval_results[key] = val
        ap = cocoEval.stats[:6]
        eval_results[f'bbox_known_{metric}_mAP_copypaste'] = (
            f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
            f'{ap[4]:.3f} {ap[5]:.3f}')
        return eval_results

    def _evaluate_uncertainty(self, predictions, eval_results, logger, dump_certainties=False):
        cocoGt = self.coco
        cocoDt = cocoGt.loadRes(predictions)
        iou_type = 'bbox'
        proposal_nums = (100, 300, 1000)
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        certainties = dict()
        gt_counts = dict()
        cocoEval = CocoEvalOOD(cocoGt, cocoDt, iou_type, class_agnostic=True)
        print_log('False positives certainty', logger=logger)
        cocoEval = self._eval(cocoEval, proposal_nums, iou_thrs, logger, uncertainty=True)
        certainties['fp_certainty'] = cocoEval.eval['uncertainty_scores']
        certainties['fp_img_id'] = cocoEval.eval['bbox_img_ids']
        certainties['fp_bbox_id'] = cocoEval.eval['bbox_ids']
        certainties['fp_bbox_label'] = cocoEval.eval['bbox_labels']
        certainties['fp_logit_certainty'] = cocoEval.eval['logit_scores']
        cocoEval = CocoEvalOOD(cocoGt, cocoDt, iou_type, class_agnostic=False)
        print_log('True positives certainty', logger=logger)
        cocoEval = self._eval(cocoEval, proposal_nums, iou_thrs, logger, uncertainty=True)
        certainties['tp_certainty'] = cocoEval.eval['uncertainty_scores']
        certainties['tp_img_id'] = cocoEval.eval['bbox_img_ids']
        certainties['tp_bbox_id'] = cocoEval.eval['bbox_ids']
        certainties['tp_bbox_label'] = cocoEval.eval['bbox_labels']
        certainties['tp_logit_certainty'] = cocoEval.eval['logit_scores']
        cocoEval = CocoEvalOOD(cocoGt, cocoDt, iou_type, class_agnostic=True, localization_uncertainty=True)
        print_log('True positive localization certainty', logger=logger)
        cocoEval = self._eval(cocoEval, proposal_nums, iou_thrs, logger, uncertainty=True)
        certainties['tpl_certainty'] = cocoEval.eval['uncertainty_scores']
        certainties['tpl_img_id'] = cocoEval.eval['bbox_img_ids']
        certainties['tpl_bbox_id'] = cocoEval.eval['bbox_ids']
        certainties['tpl_bbox_label'] = cocoEval.eval['bbox_labels']
        certainties['tpl_logit_certainty'] = cocoEval.eval['logit_scores']
        if not self.KNOWN_CLASSES:
            self.KNOWN_CLASSES = self.CLASSES
        cocoEval = CocoEvalOOD(cocoGt, cocoDt, iou_type, class_agnostic=True,
                               known_category_names=set(self.KNOWN_CLASSES))
        print_log('OOD detections certainty', logger=logger)
        cocoEval = self._eval(cocoEval, proposal_nums, iou_thrs, logger, uncertainty=True)
        certainties['ood_certainty'] = cocoEval.eval['uncertainty_scores']
        certainties['ood_logit_certainty'] = cocoEval.eval['logit_scores']
        certainties['ood_img_id'] = cocoEval.eval['bbox_img_ids']
        certainties['ood_bbox_id'] = cocoEval.eval['bbox_ids']
        certainties['ood_bbox_label'] = cocoEval.eval['bbox_labels']
        if dump_certainties:
            mmcv.dump(certainties, 'certainties.pkl')

        gt_counts['gt_count'] = cocoEval.eval['objects_count']
        gt_counts['gt_ood_count'] = cocoEval.eval['ood_objects_count']
        eval_results['uncertainty'] = self._auroc_fpr95(certainties, gt_counts, logger)
        return eval_results

    def _eval(self, cocoEval, proposal_nums, iou_thrs, logger, uncertainty=False):
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        cocoEval.evaluate()
        cocoEval.accumulate()

        # Save coco summarize print information to logger
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize(uncertainty=uncertainty)
        print_log('\n' + redirect_string.getvalue(), logger=logger)
        return cocoEval

    def _filter_certainty(self, certainty, iou_thr_id, area_rng_id):
        certainty_filtered = certainty[iou_thr_id][area_rng_id]
        return certainty_filtered[~np.isnan(certainty_filtered)]

    def _compute_auroc(self, certainties_filtered):
        auroc_results = dict()
        auroc_results['fp_tp_logit'] = auroc(certainties_filtered['fp_logit_certainty'],
                                        certainties_filtered['tp_logit_certainty'])

        auroc_results['ood_tp_logit'] = auroc(certainties_filtered['ood_logit_certainty'],
                                         certainties_filtered['tp_logit_certainty'])
        auroc_results['fp_tp'] = auroc(certainties_filtered['fp_certainty'], certainties_filtered['tp_certainty'])
        auroc_results['ood_tp'] = auroc(certainties_filtered['ood_certainty'], certainties_filtered['tp_certainty'])
        return auroc_results

    def _compute_fpr95(self, certainties_filtered):
        fpr95_results = dict()
        fpr95_results['fp_tp_logit'] = fpr_at_95_tpr(certainties_filtered['fp_logit_certainty'],
                                                   certainties_filtered['tp_logit_certainty'])

        fpr95_results['ood_tp_logit'] = fpr_at_95_tpr(certainties_filtered['ood_logit_certainty'],
                                                    certainties_filtered['tp_logit_certainty'])

        fpr95_results['fp_tp'] = fpr_at_95_tpr(certainties_filtered['fp_certainty'], certainties_filtered['tp_certainty'])
        fpr95_results['ood_tp'] = fpr_at_95_tpr(certainties_filtered['ood_certainty'],
                                              certainties_filtered['tp_certainty'])
        return fpr95_results

    def _auroc_fpr95(self, certainties, gt_counts, logger):
        eval_results = OrderedDict()
        areaRngLbl = ['all', 'small', 'medium', 'large']
        iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        for iou in [.5, .75, .9]:
            t = np.where(np.isclose(iou, iouThrs))[0][0]
            eval_results[str(iou)] = OrderedDict()
            for i, aRng in enumerate(areaRngLbl):
                certainties_filtered = {k: self._filter_certainty(v, t, i) for k, v in certainties.items()}

                fpr95_results = self._compute_fpr95(certainties_filtered)
                auroc_results = self._compute_auroc(certainties_filtered)

                gt_counts_ = {k: int(v[:, i, -1].sum()) for k, v in gt_counts.items()}
                gt_counts_['gt_id_count'] = gt_counts_['gt_count'] - gt_counts_['gt_ood_count']
                # self._log_results(certainties_filtered, fpr95_results, auroc_results, gt_counts_, iou,
                #                           aRng, logger)
                eval_results[str(iou)][aRng] = self._insert_results( certainties_filtered, fpr95_results, auroc_results, gt_counts_)
        return eval_results

    def _insert_results(self, certainties_filtered, fpr95_results, auroc_results, gt_counts_):
        eval_results = OrderedDict()
        eval_results['fps'] = certainties_filtered['fp_certainty'].shape[0]
        eval_results['tps'] = certainties_filtered["tp_certainty"].shape[0]
        eval_results['ood'] = certainties_filtered["ood_certainty"].shape[0]
        eval_results['miscl'] = int(certainties_filtered["tpl_certainty"].shape[0] - eval_results['tps'] - eval_results['ood'])
        eval_results['known_gt'] = gt_counts_["gt_id_count"]
        eval_results['unknown_gt'] = gt_counts_["gt_ood_count"]
        eval_results['auroc_fp_tp'] = auroc_results['fp_tp']
        eval_results['auroc_ood_tp'] = auroc_results['ood_tp']
        eval_results['auroc_fp_tp_logit'] = auroc_results['fp_tp_logit']
        eval_results['auroc_ood_tp_logit'] = auroc_results['ood_tp_logit']
        eval_results['fpr95_fp_tp'] = fpr95_results['fp_tp']
        eval_results['fpr95_ood_tp'] = fpr95_results['ood_tp']
        eval_results['fpr95_fp_tp_logit'] = fpr95_results['fp_tp_logit']
        eval_results['fpr95_ood_tp_logit'] = fpr95_results['ood_tp_logit']
        return eval_results

    def _log_results(self, certainties, fpr95_results, auroc_results, gt_counts_, iou, area, logger):
        print_log(f'IOU: {iou} area: {area} num_fps: {certainties["fp_certainty"].shape} known groundtruths: {gt_counts_["gt_id_count"]}'
                  f' unknown groundtruths: {gt_counts_["gt_ood_count"]}'
                  f' num_tps: {certainties["tp_certainty"].shape} num_ood: {certainties["ood_certainty"].shape}', logger=logger)
        print_log(f'False positives vs true positives auroc {auroc_results["fp_tp"]}', logger=logger)
        print_log(f'False positives vs true positives FPR at 95 TPR {fpr95_results["fp_tp"]}', logger=logger)
        print_log(f'OOD vs true positives auroc {auroc_results["ood_tp"]}', logger=logger)
        print_log(f'OOD vs true positives FPR at 95 TPR {fpr95_results["ood_tp"]}', logger=logger)
        print_log(f'LOGIT False positives vs true positives auroc {auroc_results["fp_tp_logit"]}', logger=logger)
        print_log(f'LOGIT False positives vs true positives FPR at 95 TPR {fpr95_results["fp_tp_logit"]}', logger=logger)
        print_log(f'LOGIT OOD vs true positives auroc {auroc_results["ood_tp_logit"]}', logger=logger)
        print_log(f'LOGIT OOD vs true positives FPR at 95 TPR {fpr95_results["ood_tp_logit"]}', logger=logger)

