import os
from collections import OrderedDict
import fnmatch
import mmcv
import numpy as np
from safednn_naptron.utils.metrics import auroc, fpr_at_95_tpr
import matplotlib.pyplot as plt


def load_all_results(results_dir='results/uncertainty/all_thresholds', pattern='*faster*'):
    all_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if
                 os.path.isfile(os.path.join(results_dir, f))]
    results = OrderedDict()
    for file in all_files:
        if fnmatch.fnmatch(file, pattern):
            results[file.split('/')[-1]] = mmcv.load(file)
    return results


def ood_ranking(architecture="retinanet", dataset="voc2coco"):
    assert architecture in {"retinanet", "fcos", "faster_rcnn"}
    assert dataset in {"voc2coco", "bdd"}
    certainties = load_all_results(results_dir='./work_dirs/outputs_dump', pattern=f'*{architecture}*_{dataset}*_certainties.pkl')
    aurocs = dict()
    fprs = dict()
    aurocs_logit = dict()
    fprs_logit = dict()
    areaRngLbl = ['all']
    ascending_methods = {"naptron", "opendet"}
    for a, aRng in enumerate(areaRngLbl):
        for method_filename in certainties:

            tp_logit_key = 'tpl_logit_certainty' if 'oln' in method_filename else 'tp_logit_certainty'
            tp_key = 'tpl_certainty' if 'oln' in method_filename else 'tp_certainty'
            tp_label_key = 'tpl_bbox_label' if 'oln' in method_filename else 'tp_bbox_label'
            n_classes = 1 if 'oln' in method_filename else 20
            aurocs[method_filename] = []
            fprs[method_filename] = []
            aurocs_logit[method_filename] = []
            fprs_logit[method_filename] = []
            a_l = []
            a_l_l = []
            f_l = []
            f_l_l = []
            reverse_metrics = any(m in method_filename for m in ascending_methods)
            for label in range(n_classes):
                mask_ood = certainties[method_filename]['ood_bbox_label'][0][a] == label
                mask = certainties[method_filename][tp_label_key][0][a] == label

                a_l.append(auroc(certainties[method_filename]['ood_certainty'][0][a][mask_ood],
                                 certainties[method_filename][tp_key][0][a][mask], reverse_metrics))
                a_l_l.append(auroc(certainties[method_filename]['ood_logit_certainty'][0][a][mask_ood],
                                   certainties[method_filename][tp_logit_key][0][a][mask], reverse_metrics))
                f_l.append(fpr_at_95_tpr(certainties[method_filename]['ood_certainty'][0][a][mask_ood],
                                         certainties[method_filename][tp_key][0][a][mask], reverse_metrics))
                f_l_l.append(fpr_at_95_tpr(certainties[method_filename]['ood_logit_certainty'][0][a][mask_ood],
                                           certainties[method_filename][tp_logit_key][0][a][mask], reverse_metrics))

            aurocs_logit[method_filename].append(sum(a_l_l) / len(a_l_l))
            fprs_logit[method_filename].append(sum(f_l_l) / len(f_l_l))
            aurocs[method_filename].append(sum(a_l) / len(a_l))
            fprs[method_filename].append(sum(f_l) / len(f_l))

    ranking = dict()
    method_index_in_filename = 2 if architecture == "faster_rcnn" else 1
    for a in aurocs:
        s = sum(fprs[a]) / len(aurocs[a])
        s_l = sum(fprs_logit[a]) / len(aurocs[a])
        ranking[a.split('_')[method_index_in_filename]] = s if not np.isclose(s, 0.95) else s_l

    print("FPR@95TPR", ranking)
    sorted_keys = [key for (key, value) in sorted(ranking.items(), key=lambda x: x[1])]
    sorted_vals = [value for (key, value) in sorted(ranking.items(), key=lambda x: x[1])]
    plt.bar(sorted_keys, sorted_vals)
    plt.xlabel("Method name")
    plt.ylabel("FPR@95TPR")
    plt.show()

    ranking = dict()
    for a in aurocs:
        s = sum(aurocs[a]) / len(aurocs[a])
        s_l = sum(aurocs_logit[a]) / len(aurocs[a])
        ranking[a.split('_')[method_index_in_filename]] = s if not np.isclose(s, 0.5) else s_l

    print("AUROC", ranking)
    sorted_keys = [key for (key, value) in sorted(ranking.items(), key=lambda x: x[1])]
    sorted_vals = [value for (key, value) in sorted(ranking.items(), key=lambda x: x[1])]
    plt.bar(sorted_keys, sorted_vals)
    plt.xlabel("Method name")
    plt.ylabel("AUROC")
    plt.show()


if __name__ == '__main__':
    ood_ranking("retinanet", "voc2coco")
    ood_ranking("faster_rcnn", "voc2coco")
    ood_ranking("retinanet", "bdd")
